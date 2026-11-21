const std = @import("std");
const zml = @import("zml/core");
const stablehlo = @import("mlir/dialects/stablehlo");
const Tensor = zml.Tensor;

pub const scale = 4;
const blocks = 28;
const channels = 64;
const partial_channels = 16;
const groups = 4;
pub const max_image_area = @divTrunc(std.math.maxInt(i32), groups);
const subpixels = scale * scale;
const projection_channels = 3 * subpixels;
const offset_channels = 2 * groups * subpixels;

const BlockLayer = struct { name: []const u8, out: i64, in: i64, kernel: i64 };
const block_layers = [_]BlockLayer{
    .{ .name = "channel_mixer.0", .out = 2 * channels, .in = channels, .kernel = 3 },
    .{ .name = "channel_mixer.2", .out = channels, .in = 2 * channels, .kernel = 3 },
    .{ .name = "lk.conv", .out = partial_channels, .in = partial_channels, .kernel = 17 },
    .{ .name = "attn.f.0", .out = channels, .in = channels, .kernel = 3 },
    .{ .name = "refine", .out = channels, .in = channels, .kernel = 1 },
    .{ .name = "norm", .out = channels, .in = 0, .kernel = 0 },
};
const tensors_per_block = block_layers.len * 2;
const project = 2 + blocks * tensors_per_block;
const init_pos = project + 2;
const end_conv = init_pos + 1;
const offset_conv = end_conv + 2;
const scope_conv = offset_conv + 2;
pub const tensor_count = scope_conv + 1;

const Spec = struct { name: []const u8, shape: zml.Shape };

pub fn weightAt(_: void, index: usize, buffer: []u8) Spec {
    const head_and_tail = struct {
        fn conv(name: []const u8, is_bias: bool, out: i64, in: i64, kernel: i64, buf: []u8) Spec {
            return .{
                .name = std.fmt.bufPrint(buf, "{s}.{s}", .{ name, if (is_bias) "bias" else "weight" }) catch unreachable,
                .shape = if (is_bias) .init(.{out}, .f32) else .init(.{ out, in, kernel, kernel }, .f32),
            };
        }
    };
    if (index < 2) return head_and_tail.conv("feats.0", index == 1, channels, 3, 3, buffer);
    if (index < project) {
        const offset = index - 2;
        const layer = block_layers[(offset % tensors_per_block) / 2];
        const name = std.fmt.bufPrint(buffer, "feats.{d}.{s}.{s}", .{
            offset / tensors_per_block + 1, layer.name, if (offset % 2 == 1) "bias" else "weight",
        }) catch unreachable;
        return .{
            .name = name,
            .shape = if (offset % 2 == 1 or layer.kernel == 0) .init(.{layer.out}, .f32) else .init(.{ layer.out, layer.in, layer.kernel, layer.kernel }, .f32),
        };
    }
    return switch (index) {
        project, project + 1 => head_and_tail.conv("feats.30", index != project, projection_channels, channels, 3, buffer),
        init_pos => .{ .name = "to_img.init_pos", .shape = .init(.{ 1, offset_channels, 1, 1 }, .f32) },
        end_conv, end_conv + 1 => head_and_tail.conv("to_img.end_conv", index != end_conv, 3, projection_channels, 1, buffer),
        offset_conv, offset_conv + 1 => head_and_tail.conv("to_img.offset", index != offset_conv, offset_channels, projection_channels, 1, buffer),
        scope_conv => head_and_tail.conv("to_img.scope", false, offset_channels, projection_channels, 1, buffer),
        else => unreachable,
    };
}

fn conv(input: Tensor, weight: Tensor, bias: ?Tensor) Tensor {
    const pad = @divExact(weight.dim(2) - 1, 2);
    const output = input.conv2d(weight, .{ .padding = &.{ pad, pad, pad, pad } });
    const ctx = zml.Compiler.current().mlir_ctx;
    const high = stablehlo.precisionAttr(ctx, .HIGHEST);
    output.value().owner().setAttributeByName("precision_config", .array(ctx, &.{ high, high }));
    return if (bias) |b| output.add(b.broadcast(output.shape(), &.{1})) else output;
}

fn mish(x: Tensor) Tensor {
    const softplus = x.relu().add(x.abs().negate().exp().addConstant(@as(f32, 1)).log());
    return x.mul(softplus.tanh());
}

fn groupNorm(x: Tensor, weight: Tensor, bias: Tensor) Tensor {
    const grouped = x.reshape(.{ 1, groups, @divExact(channels, groups) * x.dim(2) * x.dim(3) });
    const centered = grouped.sub(grouped.mean(2).broad(grouped.shape()));
    const variance = centered.mul(centered).mean(2);
    const inverse_std = variance.addConstant(@as(f32, 1e-5)).rsqrt().broad(grouped.shape());
    return centered.mul(inverse_std).reshape(x.shape())
        .mul(weight.broadcast(x.shape(), &.{1})).add(bias.broadcast(x.shape(), &.{1}));
}

fn residualBlock(x: Tensor, w: []const Tensor) Tensor {
    const mixed = conv(mish(conv(x, w[0], w[1])), w[2], w[3]);
    const partial = conv(mixed.slice(1, .{ .end = partial_channels }), w[4], w[5]);
    const features = Tensor.concatenate(&.{ partial, mixed.slice(1, .{ .start = partial_channels }) }, 1);
    const attended = features.mul(conv(features, w[6], w[7]).sigmoid());
    return x.add(groupNorm(conv(attended, w[8], w[9]), w[10], w[11]));
}

// NCHW pixel shuffle of the coordinate field for one component (x or y).
fn shuffleCoordinates(x: Tensor) Tensor {
    const h = x.dim(2);
    const w = x.dim(3);
    return x.reshape(.{ groups, scale, scale, h, w }).transpose(.{ 0, 3, 1, 4, 2 })
        .reshape(.{ groups, h * scale, w * scale });
}

fn sampleAt(input: Tensor, y: Tensor, x: Tensor, source_width: i64) Tensor {
    // Fold the group into the spatial index so the gather also serializes for older PJRT plugins.
    const group_stride: i32 = @intCast(@divExact(input.dim(.index), groups));
    const group_offset = Tensor.iota(y.shape(), 0).scale(group_stride);
    const index = y.convert(.i32).scale(@as(i32, @intCast(source_width))).add(x.convert(.i32)).add(group_offset)
        .withTags(.{ .group, .y, .x });
    return input.gather(.{ .index = index }, .{}).transpose(.{ 0, 3, 1, 2 })
        .reshape(.{ groups, @divExact(projection_channels, groups), y.dim(1), y.dim(2) });
}

fn interpolate(input: Tensor, x: Tensor, y: Tensor, h: i64, w: i64) Tensor {
    const x0 = x.floor();
    const y0 = y.floor();
    const one = @as(f32, 1);
    const x1 = x0.addConstant(one).minimum(Tensor.scalar(@as(f32, @floatFromInt(w - 1)), .f32).broad(x.shape()));
    const y1 = y0.addConstant(one).minimum(Tensor.scalar(@as(f32, @floatFromInt(h - 1)), .f32).broad(y.shape()));
    const upper_left = sampleAt(input, y0, x0, w);
    const upper_right = sampleAt(input, y0, x1, w);
    const lower_left = sampleAt(input, y1, x0, w);
    const lower_right = sampleAt(input, y1, x1, w);
    const dx = x.sub(x0).broadcast(upper_left.shape(), &.{ 0, 2, 3 });
    const dy = y.sub(y0).broadcast(upper_left.shape(), &.{ 0, 2, 3 });
    const top = upper_left.add(upper_right.sub(upper_left).mul(dx));
    const bottom = lower_left.add(lower_right.sub(lower_left).mul(dx));
    return top.add(bottom.sub(top).mul(dy));
}

fn dySample(x: Tensor, weights: []const Tensor) Tensor {
    const h = x.dim(2);
    const w = x.dim(3);
    const scope = conv(x, weights[scope_conv], null).sigmoid();
    const offsets = conv(x, weights[offset_conv], weights[offset_conv + 1]).mul(scope).scale(@as(f32, 0.5));
    const positioned = offsets.add(weights[init_pos].broad(offsets.shape()));
    const half = groups * subpixels;
    const x_offset = positioned.slice(1, .{ .end = half });
    const y_offset = positioned.slice(1, .{ .start = half });
    // The +0.5 and -0.5 of the align_corners=false transform cancel in pixel coordinates.
    const u = shuffleCoordinates(x_offset.add(Tensor.iota(x_offset.shape(), 3).convert(.f32)));
    const v = shuffleCoordinates(y_offset.add(Tensor.iota(y_offset.shape(), 2).convert(.f32)));
    const group_channels = @divExact(projection_channels, groups);
    const pixels = x.reshape(.{ groups, group_channels, h * w }).transpose(.{ 0, 2, 1 })
        .reshape(.{ .index = groups * h * w, .channel = group_channels });
    const xs = u.clamp(Tensor.scalar(@as(f32, 0), .f32), Tensor.scalar(@as(f32, @floatFromInt(w - 1)), .f32));
    const ys = v.clamp(Tensor.scalar(@as(f32, 0), .f32), Tensor.scalar(@as(f32, @floatFromInt(h - 1)), .f32));
    const sampled = interpolate(pixels, xs, ys, h, w).reshape(.{ 1, projection_channels, h * scale, w * scale });
    return conv(sampled, weights[end_conv], weights[end_conv + 1]);
}

pub fn forward(weights: []const Tensor, input: Tensor) Tensor {
    const h = input.dim(2);
    const w = input.dim(3);
    var features = conv(input, weights[0], weights[1]);
    for (0..blocks) |block| {
        const start = 2 + block * tensors_per_block;
        features = residualBlock(features, weights[start..][0..tensors_per_block]);
    }
    const skip = input.broadcast(.init(.{ 1, 3, subpixels, h, w }, .f32), &.{ 0, 1, 3, 4 })
        .reshape(.{ 1, projection_channels, h, w });
    return dySample(conv(features, weights[project], weights[project + 1]).add(skip), weights);
}
