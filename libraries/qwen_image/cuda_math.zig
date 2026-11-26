const std = @import("std");
const zml = @import("zml/core");
const pjrt = @import("pjrt");
const stablehlo = @import("mlir/dialects/stablehlo");

const Tensor = zml.Tensor;
const Shape = zml.Shape;
// With this workspace budget cuBLASLt uses the same reduction as the FP32 reference.
const workspace_bytes = 1 << 20;
const attention_tile_rows = 1024;

const Buffer = extern struct {
    data: [*]u8,
    rank: usize,
    dims: [*]const i64,

    fn fromFfi(buffer: *const pjrt.ffi.Buffer) Buffer {
        return .{ .data = buffer.data, .rank = buffer.dims().len, .dims = buffer.dims().ptr };
    }
};

const Operation = enum(u32) {
    linear, layer_norm, rms, silu, gelu, phase, vector, scores, attend, softmax, tiled_attention,

    fn symbol(comptime op: Operation) [:0]const u8 {
        return "zml_qwen_cuda_" ++ @tagName(op);
    }

    fn arity(op: Operation) usize {
        return switch (op) {
            .tiled_attention => 4,
            .linear, .vector => 3,
            .rms, .scores, .attend => 2,
            else => 1,
        };
    }

    fn results(op: Operation) usize {
        return switch (op) {
            .linear, .tiled_attention => 2,
            else => 1,
        };
    }
};

const Launch = *const fn (u32, [*]const Buffer, [*]Buffer, *const pjrt.Stream) callconv(.c) c_int;
var launch: ?Launch = null;

pub const Library = struct {
    lib: std.DynLib,

    // The probe owns this library until all model executables have been destroyed.
    pub fn open(platform: *const zml.Platform, path: []const u8) !Library {
        if (platform.target != .cuda or platform.pjrt_api.ffi() == null) return error.CudaFfiUnavailable;
        if (launch != null) return error.CudaMathAlreadyLoaded;
        var lib = try std.DynLib.open(path);
        errdefer lib.close();
        const function = lib.lookup(Launch, "zml_qwen_cuda_math") orelse return error.InvalidCudaMathLibrary;
        inline for (std.enums.values(Operation)) |op| {
            try platform.registerFfi(.{ .name = op.symbol(), .handler = Handler(op).call });
        }
        launch = function;
        return .{ .lib = lib };
    }

    pub fn close(self: *Library) void {
        launch = null;
        self.lib.close();
    }
};

pub fn enabled() bool {
    return launch != null;
}

fn Handler(comptime op: Operation) type {
    return struct {
        fn call(frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
            if (frame.registeringHook()) return null;
            const function = launch orelse return pjrt.ffi.Error.create(frame.api, .unavailable, "Qwen CUDA math library is closed");
            const inputs = frame.args.buffers();
            const outputs = frame.results.buffers();
            if (inputs.len != op.arity() or outputs.len != op.results())
                return pjrt.ffi.Error.create(frame.api, .invalid_argument, "Invalid Qwen CUDA math buffers");
            var in: [4]Buffer = undefined;
            var out: [2]Buffer = undefined;
            for (inputs, 0..) |input, i| in[i] = .fromFfi(input);
            for (outputs, 0..) |output, i| out[i] = .fromFfi(output);
            const stream = frame.stream() orelse return pjrt.ffi.Error.create(frame.api, .unavailable, "Missing CUDA stream");
            const status = function(@intFromEnum(op), &in, &out, stream);
            if (status != 0) {
                std.log.err("Qwen CUDA {s} failed with status {d}", .{ @tagName(op), status });
                return pjrt.ffi.Error.create(frame.api, .internal, "Qwen CUDA math operation failed");
            }
            return null;
        }
    };
}

fn emit(comptime op: Operation, inputs: []const Tensor, output: Shape) Tensor {
    const compiler = zml.Compiler.current();
    const ctx = compiler.mlir_ctx;
    var operands: [4]@TypeOf(Tensor.value(undefined)) = undefined;
    for (inputs, 0..) |input, i| operands[i] = input.value();
    const workspace: Shape = switch (op) {
        .linear => .init(.{workspace_bytes}, .u8),
        .tiled_attention => .init(.{output.dim(0) * output.dim(1) * @min(output.dim(2), attention_tile_rows) * inputs[1].dim(2)}, .f32),
        else => output,
    };
    const result_types = [_]@TypeOf(zml.mlir.Type.rankedTensor(ctx, output)){
        zml.mlir.Type.rankedTensor(ctx, output),
        zml.mlir.Type.rankedTensor(ctx, workspace),
    };
    const operation = stablehlo.custom_call(ctx, operands[0..inputs.len], result_types[0..op.results()], .{
        .call_target_name = op.symbol(),
        .backend_config = .{ .typed_ffi = .dict(ctx, &.{}) },
    }, compiler.location).appendTo(compiler.currentScope().block);
    return Tensor._result(output, operation.result(0));
}

pub fn linear(x: Tensor, weight: Tensor, bias: Tensor) Tensor {
    const width = weight.dim(1);
    const rows = @divExact(@as(i64, @intCast(x.shape().count())), width);
    const cols = weight.dim(0);
    const src = x.reshape(.{ rows, width });
    const output = Shape.init(.{ rows, cols }, .f32);
    const dst = if (rows == 1) emit(.vector, &.{ src, weight, bias }, output) else emit(.linear, &.{ src, weight, bias }, output);
    return dst.reshape(x.shape().setDim(x.rank() - 1, cols));
}

pub fn layerNorm(x: Tensor) Tensor {
    return emit(.layer_norm, &.{x}, x.shape());
}

pub fn rms(x: Tensor, gain: Tensor) Tensor {
    return emit(.rms, &.{ x, gain }, x.shape());
}

pub fn silu(x: Tensor) Tensor {
    return emit(.silu, &.{x}, x.shape());
}

pub fn gelu(x: Tensor) Tensor {
    return emit(.gelu, &.{x}, x.shape());
}

pub fn phase(time: Tensor) Tensor {
    return emit(.phase, &.{time.reshape(.{1})}, .init(.{ 2, 256 }, .f32));
}

pub fn attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, tiled: bool) Tensor {
    const context = if (tiled) emit(.tiled_attention, &.{ q, k, v, mask }, q.shape()) else blk: {
        const scores = emit(.scores, &.{ q, k }, q.shape().setDim(3, k.dim(2)));
        const logits = scores.add(mask.broadcast(scores.shape(), &.{3}));
        const probabilities = emit(.softmax, &.{logits}, logits.shape());
        break :blk emit(.attend, &.{ probabilities, v }, q.shape());
    };
    return context.transpose(.{ 0, 2, 1, 3 }).reshape(.{ q.dim(0), q.dim(2), q.dim(1) * q.dim(3) });
}
