const std = @import("std");
const zml = @import("zml/core");
const rrdb = @import("rrdbnet.zig");
const runtime = @import("zml_lib/runtime");
const checkpoint = runtime.checkpoint;

pub const Variant = rrdb.Variant;
pub const trace_names = rrdb.trace_names;

pub const Size = struct {
    width: u32,
    height: u32,

    pub fn elements(self: Size) !usize {
        if (self.width == 0 or self.height == 0) return error.InvalidDimensions;
        const area = try std.math.mul(usize, self.width, self.height);
        const count = try std.math.mul(usize, area, 3);
        _ = try std.math.mul(usize, count, @sizeOf(f32));
        return count;
    }

    pub fn upscaled(self: Size) !Size {
        return self.scaled(4);
    }

    pub fn scaled(self: Size, scale: u32) !Size {
        return .{
            .width = try std.math.mul(u32, self.width, scale),
            .height = try std.math.mul(u32, self.height, scale),
        };
    }

    fn shape(self: Size) zml.Shape {
        return .init(.{ 1, 3, @as(i64, self.height), @as(i64, self.width) }, .f32);
    }
};

const Executable = struct {
    size: Size,
    tracing: bool,
    compiled: runtime.Compiled,

    fn init(model: *const Model, size: Size, tracing: bool) !Executable {
        const input = zml.Tensor.fromShape(size.shape());
        const parameters = .{ model.parameters.tensors, input };
        var compiled = if (tracing)
            try runtime.Compiled.init(model.session, rrdb.forwardWithTrace, parameters)
        else
            try runtime.Compiled.init(model.session, rrdb.forward, parameters);
        compiled.args.bake(model.parameters.buffers);
        return .{ .size = size, .tracing = tracing, .compiled = compiled };
    }

    fn deinit(self: *const Executable, allocator: std.mem.Allocator) void {
        self.compiled.deinit(allocator);
    }
};

pub const TraceWriter = struct {
    context: *anyopaque,
    write: *const fn (*anyopaque, []const u8, zml.Buffer) anyerror!void,
};

pub const Model = struct {
    session: *const runtime.Session,
    variant: Variant,
    parameters: checkpoint.Parameters,
    compiled: ?Executable = null,

    pub fn load(session: *const runtime.Session, variant: Variant, path: []const u8) !Model {
        return .{
            .session = session,
            .variant = variant,
            .parameters = try checkpoint.load(
                session.allocator,
                session.io,
                session.platform,
                rrdb.tensorCount(variant),
                variant,
                rrdb.WeightSpec.at,
                path,
            ),
        };
    }

    pub fn deinit(self: *Model) void {
        const allocator = self.session.allocator;
        if (self.compiled) |*compiled| compiled.deinit(allocator);
        self.parameters.deinit(allocator);
    }

    pub fn infer(
        self: *Model,
        size: Size,
        input: []const f32,
        output: []f32,
        trace: ?TraceWriter,
    ) !u32 {
        const out_size = try size.scaled(self.variant.scale());
        if (input.len != try size.elements() or output.len != try out_size.elements())
            return error.InvalidBufferSize;
        const session = self.session;
        const tracing = trace != null;
        const matches = if (self.compiled) |cached|
            cached.size.width == size.width and cached.size.height == size.height and cached.tracing == tracing
        else
            false;
        if (!matches) {
            const replacement = try Executable.init(self, size, tracing);
            if (self.compiled) |*old| old.deinit(session.allocator);
            self.compiled = replacement;
        }
        const compiled = &self.compiled.?.compiled;
        var input_buffer = try zml.Buffer.fromBytesOpts(
            session.io,
            session.platform,
            size.shape(),
            std.mem.sliceAsBytes(input),
            .{ .wait = true },
        );
        defer input_buffer.deinit();
        try compiled.run(.{input_buffer});
        const output_slice = zml.Slice.init(out_size.shape(), std.mem.sliceAsBytes(output));
        if (trace) |writer| {
            var results = compiled.results.get(rrdb.Trace(zml.Buffer));
            defer results.output.deinit();
            defer for (&results.layers) |*layer| layer.deinit();
            const count = try session.verifyPlacement(&results.output);
            try results.output.toSlice(session.io, output_slice);
            for (trace_names, results.layers) |name, layer| {
                _ = try session.verifyPlacement(&layer);
                try writer.write(writer.context, name, layer);
            }
            return count;
        } else {
            var result = compiled.results.get(zml.Buffer);
            defer result.deinit();
            const count = try session.verifyPlacement(&result);
            try result.toSlice(session.io, output_slice);
            return count;
        }
    }
};
