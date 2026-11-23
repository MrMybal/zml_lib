const std = @import("std");
const zml = @import("zml/core");
const stablehlo = @import("mlir/dialects/stablehlo");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const Tensor = l.Tensor;
const t = rt.transformer;

pub const channels = 8;
pub const max_latent_edge = 16;
const scale = 4;
pub const official = [3]i64{ 512, 128, 32 };
pub const fixture = [3]i64{ 8, 6, 4 };

fn parameter(name: []const u8, dims: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn affine(comptime name: []const u8, output: i64, dims: anytype) [2]l.Parameter {
    return .{ parameter(name ++ ".weight", dims), parameter(name ++ ".bias", .{output}) };
}

fn kernel(comptime name: []const u8, input: i64, output: i64) [2]l.Parameter {
    return affine(name, output, .{ output, input, 3, 3, 3 });
}

fn residualWeights(comptime name: []const u8, width: i64) [8]l.Parameter {
    return affine(name ++ ".norm1", width, .{width}) ++ kernel(name ++ ".conv1", width, width) ++
        affine(name ++ ".norm2", width, .{width}) ++ kernel(name ++ ".conv2", width, width);
}

pub fn schema(comptime widths: [3]i64) [74]l.Parameter {
    @setEvalBranchQuota(150_000);
    const a, const b, const c = widths;
    return kernel("input_layer", channels, a) ++
        residualWeights("middle_block.0", a) ++ residualWeights("middle_block.1", a) ++
        residualWeights("blocks.0", a) ++ residualWeights("blocks.1", a) ++ kernel("blocks.2.conv", a, b * 8) ++
        residualWeights("blocks.3", b) ++ residualWeights("blocks.4", b) ++ kernel("blocks.5.conv", b, c * 8) ++
        residualWeights("blocks.6", c) ++ residualWeights("blocks.7", c) ++
        affine("out_layer.0", c, .{c}) ++ kernel("out_layer.2", c, 1);
}

fn conv(w: l.Weights, name: []const u8, input: Tensor) Tensor {
    const spatial: []const i64 = &.{ 2, 3, 4 };
    const output = input.conv2d(w.get(name, ".weight"), .{
        .window_strides = &.{ 1, 1, 1 },
        .padding = &.{ 1, 1, 1, 1, 1, 1 },
        .lhs_dilation = &.{ 1, 1, 1 },
        .rhs_dilation = &.{ 1, 1, 1 },
        .window_reversal = &.{ false, false, false },
        .input_spatial_dimensions = spatial,
        .kernel_spatial_dimensions = spatial,
        .output_spatial_dimensions = spatial,
    });
    const ctx = zml.Compiler.current().mlir_ctx;
    const high = stablehlo.precisionAttr(ctx, .HIGHEST);
    output.value().owner().setAttributeByName("precision_config", .array(ctx, &.{ high, high }));
    return l.biasAdd(output, w.get(name, ".bias"), 1);
}

fn norm(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    const pixels = x.transpose(.{ 0, 2, 3, 4, 1 });
    return t.norm(w, name, pixels, 1e-5).transpose(.{ 0, 4, 1, 2, 3 });
}

fn residual(w: l.Weights, comptime name: []const u8, x: Tensor) Tensor {
    const inner = conv(w, name ++ ".conv1", norm(w, name ++ ".norm1", x).silu());
    return x.add(conv(w, name ++ ".conv2", norm(w, name ++ ".norm2", inner).silu()));
}

fn shuffle(x: Tensor) Tensor {
    const n, const cf, const d, const h, const w = x.dims()[0..5].*;
    const c = @divExact(cf, 8);
    return x.reshape(.{ n * c, 2, 2, 2, d, h, w }).transpose(.{ 0, 4, 1, 5, 2, 6, 3 })
        .reshape(.{ n, c, 2 * d, 2 * h, 2 * w });
}

pub const Volume = struct {
    d: i64,
    h: i64,
    w: i64,

    pub fn init(d: i64, h: i64, w: i64) !Volume {
        for ([_]i64{ d, h, w }) |dim| {
            if (dim < 1 or dim > max_latent_edge) return error.InvalidLatentVolume;
        }
        return .{ .d = d, .h = h, .w = w };
    }

    pub fn latentCount(self: Volume) usize {
        return @intCast(channels * self.d * self.h * self.w);
    }

    pub fn fieldCount(self: Volume) usize {
        return @intCast((scale * self.d) * (scale * self.h) * (scale * self.w));
    }
};

pub fn Decoder(comptime widths: [3]i64) type {
    return struct {
        const Self = @This();
        const params = schema(widths);
        graph: rt.TensorModel,

        pub fn load(session: *const rt.Session, path: []const u8) !Self {
            return .{ .graph = try .load(session, &params, path) };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        pub fn decode(self: *Self, size: Volume, latent: []const f32, logits: []f32) !u32 {
            _ = try Volume.init(size.d, size.h, size.w);
            return self.graph.infer(forward, .init(.{ 1, channels, size.d, size.h, size.w }, .f32), .init(.{ 1, 1, scale * size.d, scale * size.h, scale * size.w }, .f32), latent, logits);
        }

        fn forward(parameters: []const Tensor, x: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &params, .tensors = parameters };
            var y = conv(w, "input_layer", x);
            inline for (.{ "middle_block.0", "middle_block.1", "blocks.0", "blocks.1" }) |key| y = residual(w, key, y);
            y = shuffle(conv(w, "blocks.2.conv", y));
            y = residual(w, "blocks.4", residual(w, "blocks.3", y));
            y = shuffle(conv(w, "blocks.5.conv", y));
            y = residual(w, "blocks.7", residual(w, "blocks.6", y));
            return conv(w, "out_layer.2", norm(w, "out_layer.0", y).silu());
        }
    };
}
