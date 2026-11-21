const std = @import("std");
const zml = @import("zml/core");
const stablehlo = @import("mlir/dialects/stablehlo");
const spatial = @import("zml_lib/runtime").spatial;
const Tensor = zml.Tensor;

pub const Variant = enum(u32) {
    RealESRGAN_x4plus = 1,
    RealESRGAN_x4plus_anime_6B = 2,
    RealESRGAN_x2plus = 3,

    pub fn blocks(self: Variant) usize {
        return switch (self) {
            .RealESRGAN_x4plus, .RealESRGAN_x2plus => 23,
            .RealESRGAN_x4plus_anime_6B => 6,
        };
    }

    pub fn scale(self: Variant) u32 {
        return if (self == .RealESRGAN_x2plus) 2 else 4;
    }
};

const features = 64;
const growth = 32;
const dense_per_block = 3;
const convs_per_dense = 5;
const convs_per_block = dense_per_block * convs_per_dense;

const Layer = struct { name: []const u8, in: i64, out: i64 };
const tail = [_]Layer{
    .{ .name = "conv_body", .in = features, .out = features },
    .{ .name = "conv_up1", .in = features, .out = features },
    .{ .name = "conv_up2", .in = features, .out = features },
    .{ .name = "conv_hr", .in = features, .out = features },
    .{ .name = "conv_last", .in = features, .out = 3 },
};

pub fn tensorCount(variant: Variant) usize {
    return 2 * (1 + variant.blocks() * convs_per_block + tail.len);
}

pub const WeightSpec = struct {
    name: []const u8,
    shape: zml.Shape,

    pub fn at(variant: Variant, parameter: usize, name_buffer: []u8) WeightSpec {
        const layer_index = parameter / 2;
        const body_convs = variant.blocks() * convs_per_block;
        var layer_name: [40]u8 = undefined;
        const layer = if (layer_index == 0)
            Layer{ .name = "conv_first", .in = if (variant == .RealESRGAN_x2plus) 12 else 3, .out = features }
        else if (layer_index <= body_convs) blk: {
            const index = layer_index - 1;
            const conv = index % convs_per_dense;
            break :blk Layer{
                .name = std.fmt.bufPrint(&layer_name, "body.{d}.rdb{d}.conv{d}", .{
                    index / convs_per_block,
                    (index % convs_per_block) / convs_per_dense + 1,
                    conv + 1,
                }) catch unreachable,
                .in = features + @as(i64, @intCast(conv)) * growth,
                .out = if (conv == convs_per_dense - 1) features else growth,
            };
        } else tail[layer_index - body_convs - 1];
        const is_weight = parameter % 2 == 0;
        return .{
            .name = std.fmt.bufPrint(name_buffer, "{s}.{s}", .{
                layer.name, if (is_weight) "weight" else "bias",
            }) catch unreachable,
            .shape = if (is_weight) .init(.{ layer.out, layer.in, 3, 3 }, .f32) else .init(.{layer.out}, .f32),
        };
    }
};

fn convolve(input: Tensor, weights: []const Tensor, layer: usize) Tensor {
    const weight = weights[layer * 2];
    const bias = weights[layer * 2 + 1];
    const output = input.conv2d(weight, .{ .padding = &.{ 1, 1, 1, 1 } });
    // ZML's conv2d emits DEFAULT. HIGHEST requests full F32 on XLA GPU.
    const ctx = zml.Compiler.current().mlir_ctx;
    const highest = stablehlo.precisionAttr(ctx, .HIGHEST);
    output.value().owner().setAttributeByName("precision_config", .array(ctx, &.{ highest, highest }));
    return output.add(bias.broadcast(output.shape(), &.{1}));
}

fn activate(input: Tensor) Tensor {
    return input.maximum(input.scale(@as(f32, 0.2)));
}

fn dense(input: Tensor, weights: []const Tensor, start: usize) Tensor {
    const x1 = activate(convolve(input, weights, start));
    const x2 = activate(convolve(.concatenate(&.{ input, x1 }, 1), weights, start + 1));
    const x3 = activate(convolve(.concatenate(&.{ input, x1, x2 }, 1), weights, start + 2));
    const x4 = activate(convolve(.concatenate(&.{ input, x1, x2, x3 }, 1), weights, start + 3));
    const x5 = convolve(.concatenate(&.{ input, x1, x2, x3, x4 }, 1), weights, start + 4);
    return input.add(x5.scale(@as(f32, 0.2)));
}

fn upsample(input: Tensor) Tensor {
    const n = input.dim(0);
    const c = input.dim(1);
    const h = input.dim(2);
    const w = input.dim(3);
    // Broadcast each pixel across the inserted axes before merging them into H and W.
    return input.broadcast(.init(.{ n, c, h, 2, w, 2 }, input.dtype()), &.{ 0, 1, 2, 4 })
        .reshape(.{ n, c, h * 2, w * 2 });
}

pub const trace_names = [_][]const u8{
    "conv_first", "body.0.rdb1", "body.0", "conv_body", "conv_up1", "conv_up2",
};

pub fn Trace(comptime T: type) type {
    return struct { output: T, layers: [trace_names.len]T };
}

pub fn forward(weights: []const Tensor, input: Tensor) Tensor {
    return forwardWithTrace(weights, input).output;
}

pub fn forwardWithTrace(weights: []const Tensor, input: Tensor) Trace(Tensor) {
    const blocks = (weights.len / 2 - 1 - tail.len) / convs_per_block;
    const x2 = weights[0].dim(1) == 12;
    const height = input.dim(2);
    const width = input.dim(3);
    const prepared = if (x2) blk: {
        const padded = spatial.reflectEnd(spatial.reflectEnd(input, 2, @mod(height, 2)), 3, @mod(width, 2));
        const rows = @divExact(padded.dim(2), 2);
        const cols = @divExact(padded.dim(3), 2);
        break :blk padded.reshape(.{ 1, 3, rows, 2, cols, 2 }).transpose(.{ 0, 1, 3, 5, 2, 4 })
            .reshape(.{ 1, 12, rows, cols });
    } else input;
    const first = convolve(prepared, weights, 0);
    var trunk = first;
    var first_dense: Tensor = undefined;
    var first_block: Tensor = undefined;
    for (0..blocks) |block| {
        const start = 1 + block * convs_per_block;
        const d1 = dense(trunk, weights, start);
        const d2 = dense(d1, weights, start + convs_per_dense);
        const d3 = dense(d2, weights, start + 2 * convs_per_dense);
        trunk = trunk.add(d3.scale(@as(f32, 0.2)));
        if (block == 0) {
            first_dense = d1;
            first_block = trunk;
        }
    }
    const body_index = 1 + blocks * convs_per_block;
    const body = convolve(trunk, weights, body_index);
    const up1 = activate(convolve(upsample(first.add(body)), weights, body_index + 1));
    const up2 = activate(convolve(upsample(up1), weights, body_index + 2));
    const hr = activate(convolve(up2, weights, body_index + 3));
    const output = convolve(hr, weights, body_index + 4);
    return .{
        .output = if (x2) output.slice(2, .{ .end = 2 * height }).slice(3, .{ .end = 2 * width }) else output,
        .layers = .{ first, first_dense, first_block, body, up1, up2 },
    };
}
