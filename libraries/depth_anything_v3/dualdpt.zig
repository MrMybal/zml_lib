const rt = @import("zml_lib/runtime");
const wd = @import("weights.zig");
const l = rt.layers;
const t = rt.transformer;
const Tensor = l.Tensor;

fn axisCoordinates(x: Tensor, comptime axis: u3, span: f32) Tensor {
    const count: f32 = @floatFromInt(x.dim(axis));
    const shape = x.shape().setDim(1, 1);
    return Tensor.iota(shape.withDtype(.i32), axis).convert(.f32).addConstant(@as(f32, 0.5))
        .scale(2 / count).subConstant(@as(f32, 1)).scale(span);
}

fn uvEmbed(x: Tensor, h: i64, w: i64) Tensor {
    const height: f32 = @floatFromInt(h);
    const width: f32 = @floatFromInt(w);
    const diagonal = @sqrt(height * height + width * width);
    const shape = x.shape().setDim(1, @divExact(x.dim(1), 4));
    const inv = Tensor.iota(shape.withDtype(.i32), 1).convert(.f32)
        .scale(-@log(@as(f32, 100)) / @as(f32, @floatFromInt(shape.dim(1)))).exp();
    const u = axisCoordinates(x, 3, width / diagonal).broad(shape).mul(inv);
    const v = axisCoordinates(x, 2, height / diagonal).broad(shape).mul(inv);
    return x.add(Tensor.concatenate(&.{ u.sin(), u.cos(), v.sin(), v.cos() }, 1).scale(@as(f32, 0.1)));
}

fn residual(w: l.Weights, x: Tensor, scope: []const u8) Tensor {
    var key: [96]u8 = undefined;
    const mid = l.conv(w, l.name(&key, "{s}.conv1", .{scope}), x.relu(), 1, @splat(1), true);
    return x.add(l.conv(w, l.name(&key, "{s}.conv2", .{scope}), mid.relu(), 1, @splat(1), true));
}

fn fuse(w: l.Weights, features: [4]Tensor, comptime suffix: []const u8) Tensor {
    var scope: [64]u8 = undefined;
    var key: [96]u8 = undefined;
    var out = features[3];
    for (0..4) |reverse| {
        const level = 3 - reverse;
        const name = l.name(&scope, "{s}.refinenet{d}{s}", .{ wd.pyramid, level + 1, suffix });
        if (level != 3)
            out = out.add(residual(w, features[level], l.name(&key, "{s}.resConfUnit1", .{name})));
        out = residual(w, out, l.name(&key, "{s}.resConfUnit2", .{name}));
        const rows = if (level == 0) 2 * out.dim(2) else features[level - 1].dim(2);
        const cols = if (level == 0) 2 * out.dim(3) else features[level - 1].dim(3);
        out = l.conv(w, l.name(&key, "{s}.out_conv", .{name}), rt.spatial.bilinearAlignCorners(out, rows, cols), 1, @splat(0), true);
    }
    return out;
}

pub fn forward(w: l.Weights, tokens: [4]Tensor, height: i64, width: i64) struct { Tensor, Tensor } {
    const rows = @divExact(height, 14);
    const cols = @divExact(width, 14);
    var features: [4]Tensor = undefined;
    var key: [96]u8 = undefined;
    for (tokens, 0..) |level, i| {
        const x = t.norm(w, "model.head.norm", level, 1e-5).transpose(.{ 0, 2, 1 }).reshape(.{ level.dim(0), 1536, rows, cols });
        const projected = l.conv(w, l.name(&key, "model.head.projects.{d}", .{i}), x, 1, @splat(0), true);
        const positioned = uvEmbed(projected, height, width);
        const resizer = l.name(&key, "model.head.resize_layers.{d}", .{i});
        const aligned = switch (i) {
            0 => t.deconvNonoverlap(w, resizer, positioned, 4),
            1 => t.deconv2x(w, resizer, positioned),
            2 => positioned,
            else => l.conv3x3Stride2(w, resizer, positioned, true),
        };
        features[i] = l.conv(w, l.name(&key, "{s}.layer{d}_rn", .{ wd.pyramid, i + 1 }), aligned, 1, @splat(1), false);
    }

    const fused = l.conv(w, wd.pyramid ++ ".output_conv1", fuse(w, features, ""), 1, @splat(1), true);
    const at_resolution = rt.spatial.bilinearAlignCorners(fused, height, width);
    const main_1 = l.conv(w, wd.pyramid ++ ".output_conv2.0", uvEmbed(at_resolution, height, width), 1, @splat(1), true).relu();
    const logits = l.conv(w, wd.pyramid ++ ".output_conv2.2", main_1, 1, @splat(0), true);
    const depth = logits.slice(1, .{ .end = 1 }).exp();
    const confidence = logits.slice(1, .{ .start = 1 }).exp().addConstant(@as(f32, 1));

    var auxiliary = fuse(w, features, "_aux");
    for (0..5) |i|
        auxiliary = l.conv(w, l.name(&key, "{s}.output_conv1_aux.3.{d}", .{ wd.pyramid, i }), auxiliary, 1, @splat(1), true);
    const features_aux = l.conv(w, wd.pyramid ++ ".output_conv2_aux.3.0", uvEmbed(auxiliary, height, width), 1, @splat(1), true)
        .transpose(.{ 0, 2, 3, 1 });
    const normalized = t.norm(w, wd.pyramid ++ ".output_conv2_aux.0.2", features_aux, 1e-5).transpose(.{ 0, 3, 1, 2 }).relu();
    const rays = l.conv(w, wd.pyramid ++ ".output_conv2_aux.3.5", normalized, 1, @splat(0), true);
    const ray_conf = rays.slice(1, .{ .start = 6 }).exp().addConstant(@as(f32, 1));
    return .{
        Tensor.concatenate(&.{ depth, confidence }, 1),
        Tensor.concatenate(&.{ rays.slice(1, .{ .end = 6 }), ray_conf }, 1),
    };
}
