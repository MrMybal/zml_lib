const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = zml.Tensor;

pub const channels = 16;
pub const factor = 8;
pub const max_latent_size = 128;
pub const Mode = enum { encode, decode };

// Normalized latents are the contract between the VAE and the Qwen Image denoiser.
const mean = [channels]f32{ -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921 };
const deviation = [channels]f32{ 2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160 };
const encoder_widths = [_]i64{ 96, 192, 384, 384 };
const decoder_widths = [_]i64{ 384, 384, 192, 96 };

fn param(name: []const u8, dims: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn convolution(comptime name: []const u8, out: i64, in: i64, kernel: i64) [2]l.Parameter {
    return .{ param(name ++ ".weight", .{ out, in, kernel, kernel }), param(name ++ ".bias", .{out}) };
}

fn residualParams(comptime prefix: []const u8, comptime in: i64, comptime out: i64) [if (in == out) 6 else 8]l.Parameter {
    const base = [_]l.Parameter{param(prefix ++ ".residual.0.gamma", .{in})} ++
        convolution(prefix ++ ".residual.2", out, in, 3) ++
        [_]l.Parameter{param(prefix ++ ".residual.3.gamma", .{out})} ++
        convolution(prefix ++ ".residual.6", out, out, 3);
    return if (in == out) base else base ++ convolution(prefix ++ ".shortcut", out, in, 1);
}

fn makeSchema(comptime mode: Mode) [if (mode == .encode) 82 else 104]l.Parameter {
    @setEvalBranchQuota(60_000);
    const encode = mode == .encode;
    const prefix = if (encode) "encoder." else "decoder.";
    const widths = if (encode) encoder_widths else decoder_widths;
    const resample_list = if (encode) prefix ++ "downsamples." else prefix ++ "upsamples.";
    const res_blocks = if (encode) 2 else 3;
    var params: [if (encode) 82 else 104]l.Parameter = undefined;
    params[0..9].* = convolution(prefix ++ "conv1", widths[0], if (encode) 3 else channels, 3) ++
        convolution(if (encode) "conv1" else "conv2", if (encode) 2 * channels else channels, if (encode) 2 * channels else channels, 1) ++
        [_]l.Parameter{param(prefix ++ "middle.1.norm.gamma", .{384})} ++ convolution(prefix ++ "middle.1.to_qkv", 1152, 384, 1) ++
        convolution(prefix ++ "middle.1.proj", 384, 384, 1);
    params[9..24].* = residualParams(prefix ++ "middle.0", 384, 384) ++ residualParams(prefix ++ "middle.2", 384, 384) ++
        [_]l.Parameter{param(prefix ++ "head.0.gamma", .{widths[3]})} ++ convolution(prefix ++ "head.2", if (encode) 2 * channels else 3, widths[3], 3);
    comptime var offset: usize = 24;
    comptime var input_channels: i64 = widths[0];
    inline for (widths, 0..) |out, level| {
        inline for (0..res_blocks) |block| {
            const index = level * (res_blocks + 1) + block;
            const key = std.fmt.comptimePrint("{s}{d}", .{ resample_list, index });
            const specs = residualParams(key, input_channels, out);
            @memcpy(params[offset..][0..specs.len], &specs);
            offset += specs.len;
            input_channels = out;
        }
        if (level < 3) {
            const index = level * (res_blocks + 1) + res_blocks;
            const key = std.fmt.comptimePrint("{s}{d}.resample.1", .{ resample_list, index });
            input_channels = if (encode) out else @divExact(out, 2);
            params[offset..][0..2].* = convolution(key, input_channels, out, 3);
            offset += 2;
        }
    }
    std.debug.assert(offset == params.len);
    return params;
}

fn conv(w: l.Weights, name: []const u8, input: Tensor, kernel: i64) Tensor {
    return l.conv(w, name, input, 1, @splat(@divExact(kernel - 1, 2)), true);
}

fn norm(x: Tensor, gamma: Tensor) Tensor {
    const denominator = x.mul(x).sum(1).sqrt().maximum(Tensor.scalar(@as(f32, 1e-12), .f32));
    return x.div(denominator.broad(x.shape())).scale(@sqrt(@as(f32, @floatFromInt(x.dim(1)))))
        .mul(gamma.broadcast(x.shape(), &.{1}));
}

fn residual(w: l.Weights, prefix: []const u8, x: Tensor) Tensor {
    var key: [96]u8 = undefined;
    const a = norm(x, w.get(l.name(&key, "{s}.residual.0", .{prefix}), ".gamma")).silu();
    const b = conv(w, l.name(&key, "{s}.residual.2", .{prefix}), a, 3);
    const c = norm(b, w.get(l.name(&key, "{s}.residual.3", .{prefix}), ".gamma")).silu();
    const d = conv(w, l.name(&key, "{s}.residual.6", .{prefix}), c, 3);
    const skip = if (x.dim(1) != d.dim(1)) conv(w, l.name(&key, "{s}.shortcut", .{prefix}), x, 1) else x;
    return skip.add(d);
}

fn attention(w: l.Weights, comptime prefix: []const u8, x: Tensor) Tensor {
    const n = norm(x, w.get(prefix ++ "norm", ".gamma"));
    const points = x.dim(2) * x.dim(3);
    const d = x.dim(1);
    const qkv = conv(w, prefix ++ "to_qkv", n, 1).reshape(.{ 1, 3 * d, points }).transpose(.{ 0, 2, 1 });
    const values = t.attention(qkv.slice(2, .{ .end = d }), qkv.slice(2, .{ .start = d, .end = 2 * d }), qkv.slice(2, .{ .start = 2 * d }), 1).transpose(.{ 0, 2, 1 }).reshape(x.shape());
    return x.add(conv(w, prefix ++ "proj", values, 1));
}

fn normalize(x: Tensor, comptime encode: bool) Tensor {
    const offset = Tensor.constantTensor(.init(.{channels}, .f32), std.mem.asBytes(&mean)).broadcast(x.shape(), &.{1});
    const sigma = Tensor.constantTensor(.init(.{channels}, .f32), std.mem.asBytes(&deviation)).broadcast(x.shape(), &.{1});
    return if (encode) x.sub(offset).div(sigma) else x.mul(sigma).add(offset);
}

pub fn Autoencoder(comptime mode: Mode) type {
    const encode = mode == .encode;
    const prefix = if (encode) "encoder." else "decoder.";
    const list = if (encode) prefix ++ "downsamples." else prefix ++ "upsamples.";
    const res_blocks = if (encode) 2 else 3;
    return struct {
        const Self = @This();
        pub const schema = makeSchema(mode);
        graph: runtime.TensorModel,

        pub fn load(session: *const runtime.Session, path: []const u8) !Self {
            return .{ .graph = try .load(session, &schema, path) };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        fn stages(w: l.Weights, input: Tensor) Tensor {
            var x = input;
            var key: [64]u8 = undefined;
            for (0..4) |level| {
                for (0..res_blocks) |block| {
                    x = residual(w, l.name(&key, "{s}{d}", .{ list, level * (res_blocks + 1) + block }), x);
                }
                if (level != 3) {
                    const name = l.name(&key, "{s}{d}.resample.1", .{ list, level * (res_blocks + 1) + res_blocks });
                    x = if (encode) l.conv(w, name, x, 2, .{ 0, 1, 0, 1 }, true) else conv(w, name, l.nearest2x(x), 3);
                }
            }
            return x;
        }

        fn forward(parameters: []const Tensor, input: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
            const start = if (encode) input else conv(w, "conv2", normalize(input, false), 1);
            var x = conv(w, prefix ++ "conv1", start, 3);
            if (encode) x = stages(w, x);
            x = residual(w, prefix ++ "middle.0", x);
            x = attention(w, prefix ++ "middle.1.", x);
            x = residual(w, prefix ++ "middle.2", x);
            if (!encode) x = stages(w, x);
            x = conv(w, prefix ++ "head.2", norm(x, w.get(prefix ++ "head.0", ".gamma")).silu(), 3);
            if (encode) return normalize(conv(w, "conv1", x, 1).slice(1, .{ .end = channels }), true);
            return x.maximum(Tensor.scalar(@as(f32, -1), .f32)).minimum(Tensor.scalar(@as(f32, 1), .f32));
        }

        pub fn infer(self: *Self, height: usize, width: usize, input: []const f32, output: []f32) !u32 {
            if (height == 0 or width == 0 or height > max_latent_size or width > max_latent_size) return error.InvalidSize;
            const rgb: zml.Shape = .init(.{ 1, 3, height * factor, width * factor }, .f32);
            const latent: zml.Shape = .init(.{ 1, channels, height, width }, .f32);
            return self.graph.infer(forward, if (encode) rgb else latent, if (encode) latent else rgb, input, output);
        }
    };
}
