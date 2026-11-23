const dinov2 = @import("zml_lib/dinov2");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const Tensor = l.Tensor;
const weights = @import("weights.zig");

fn residual(w: l.Weights, prefix: []const u8, x: Tensor) Tensor {
    var key: [80]u8 = undefined;
    const y = l.conv(w, l.name(&key, "{s}.conv1", .{prefix}), x.relu(), 1, @splat(1), true);
    return x.add(l.conv(w, l.name(&key, "{s}.conv2", .{prefix}), y.relu(), 1, @splat(1), true));
}

fn refine(w: l.Weights, prefix: []const u8, x: Tensor, skip: ?Tensor, height: i64, width: i64) Tensor {
    var key: [80]u8 = undefined;
    const merged = if (skip) |value|
        x.add(residual(w, l.name(&key, "{s}.resConfUnit1", .{prefix}), value))
    else
        x;
    const refined = residual(w, l.name(&key, "{s}.resConfUnit2", .{prefix}), merged);
    const resized = runtime.spatial.bilinearAlignCorners(refined, height, width);
    return l.conv(w, l.name(&key, "{s}.out_conv", .{prefix}), resized, 1, @splat(0), true);
}

pub fn forward(parameters: []const Tensor, blocks: [4]Tensor, image_height: i64, image_width: i64) Tensor {
    const w: l.Weights = .{ .schema = &weights.schema, .tensors = parameters };
    return graph(w, "depth_head", blocks, image_height, image_width, null);
}

pub fn graph(
    w: l.Weights,
    comptime prefix: []const u8,
    blocks: [4]Tensor,
    image_height: i64,
    image_width: i64,
    comptime motion: ?*const fn (l.Weights, usize, Tensor) Tensor,
) Tensor {
    const height = @divExact(image_height, dinov2.patch_size);
    const width = @divExact(image_width, dinov2.patch_size);
    var pyramid: [4]Tensor = undefined;
    for (blocks, 0..) |tokens, i| {
        var key: [64]u8 = undefined;
        const image = tokens.slice(1, .{ .start = 1 }).transpose(.{ 0, 2, 1 })
            .reshape(.{ tokens.dim(0), tokens.dim(2), height, width });
        const projected = l.conv(w, l.name(&key, prefix ++ ".projects.{d}", .{i}), image, 1, @splat(0), true);
        const resized = switch (i) {
            0 => runtime.transformer.deconvNonoverlap(w, prefix ++ ".resize_layers.0", projected, 4),
            1 => runtime.transformer.deconv2x(w, prefix ++ ".resize_layers.1", projected),
            2 => projected,
            3 => l.conv3x3Stride2(w, prefix ++ ".resize_layers.3", projected, true),
            else => unreachable,
        };
        const mixed = if (motion != null and i >= 2) motion.?(w, i - 2, resized) else resized;
        pyramid[i] = l.conv(w, l.name(&key, prefix ++ ".scratch.layer{d}_rn", .{i + 1}), mixed, 1, @splat(1), false);
    }

    const fourth = refine(w, prefix ++ ".scratch.refinenet4", pyramid[3], null, height, width);
    const path4 = if (motion) |mix| mix(w, 2, fourth) else fourth;
    const third = refine(w, prefix ++ ".scratch.refinenet3", path4, pyramid[2], 2 * height, 2 * width);
    const path3 = if (motion) |mix| mix(w, 3, third) else third;
    const path2 = refine(w, prefix ++ ".scratch.refinenet2", path3, pyramid[1], 4 * height, 4 * width);
    const path1 = refine(w, prefix ++ ".scratch.refinenet1", path2, pyramid[0], 8 * height, 8 * width);
    const reduced = l.conv(w, prefix ++ ".scratch.output_conv1", path1, 1, @splat(1), true);
    const restored = runtime.spatial.bilinearAlignCorners(reduced, image_height, image_width);
    const hidden = l.conv(w, prefix ++ ".scratch.output_conv2.0", restored, 1, @splat(1), true).relu();
    return l.conv(w, prefix ++ ".scratch.output_conv2.2", hidden, 1, @splat(0), true).relu()
        .reshape(.{ blocks[0].dim(0), image_height, image_width });
}
