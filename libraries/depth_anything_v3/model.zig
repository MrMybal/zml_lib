const rt = @import("zml_lib/runtime");
const wd = @import("weights.zig");
const dual = @import("dualdpt.zig");
const l = rt.layers;
const t = rt.transformer;
const Tensor = l.Tensor;
const width = wd.width;
const backbone = wd.backbone;
pub const parameters = wd.schema;

fn tokens(w: l.Weights, image: Tensor) Tensor {
    const patches = l.conv(w, backbone ++ ".patch_embed.proj", image, 14, @splat(0), true);
    const rows = patches.dim(2);
    const cols = patches.dim(3);
    const views = patches.dim(0);
    const count = rows * cols;
    const patch_tokens = patches.transpose(.{ 0, 2, 3, 1 }).reshape(.{ views, count, width });
    const cls = w.get(backbone ++ ".cls_token", "").broad(.init(.{ views, 1, width }, .f32));
    const embedding = w.get(backbone ++ ".pos_embed", "");
    const positions = if (rows == 37 and cols == 37) embedding else positioned: {
        const plane = embedding.slice(1, .{ .start = 1 }).reshape(.{ 1, 37, 37, width }).transpose(.{ 0, 3, 1, 2 });
        const resized = rt.spatial.bicubicHalfPixel(plane, rows, cols, 0.1)
            .transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, count, width });
        break :positioned Tensor.concatenate(&.{ embedding.slice(1, .{ .end = 1 }), resized }, 1);
    };
    const with_cls = Tensor.concatenate(&.{ cls, patch_tokens }, 1);
    return with_cls.add(positions.broad(with_cls.shape()));
}

fn angles(rows: i64, cols: i64, views: i64, comptime global: bool) Tensor {
    const shape = @TypeOf(Tensor.scalar(0, .f32).shape()).init(.{ rows, cols }, .i32);
    const y = Tensor.iota(shape, 0).reshape(.{rows * cols});
    const x = Tensor.iota(shape, 1).reshape(.{rows * cols});
    const one = Tensor.scalar(@as(f32, 1), .f32).broad(.init(.{rows * cols}, .f32));
    const y_coord = if (global) one else y.convert(.f32).addConstant(@as(f32, 1));
    const x_coord = if (global) one else x.convert(.f32).addConstant(@as(f32, 1));
    const prefix = Tensor.zeroes(.init(.{1}, .f32));
    const ys = Tensor.concatenate(&.{ prefix, y_coord }, 0);
    const xs = Tensor.concatenate(&.{ prefix, x_coord }, 0);
    const freq = Tensor.iota(.init(.{16}, .i32), 0).convert(.f32).scale(-@log(@as(f32, 100)) / 16).exp();
    const dims = @TypeOf(shape).init(.{ 1 + rows * cols, 16 }, .f32);
    const a = ys.broadcast(dims, &.{0}).mul(freq.broadcast(dims, &.{1}));
    const b = xs.broadcast(dims, &.{0}).mul(freq.broadcast(dims, &.{1}));
    const phase = Tensor.concatenate(&.{ a, a, b, b }, 1);
    if (!global) return phase;
    return phase.broadcast(.init(.{ views, phase.dim(0), phase.dim(1) }, .f32), &.{ 1, 2 })
        .reshape(.{ views * phase.dim(0), phase.dim(1) });
}

fn rotateHeads(x: Tensor, sin: Tensor, cos: Tensor) Tensor {
    const quarter = @divExact(x.dim(3), 4);
    const a = x.slice(3, .{ .end = quarter });
    const b = x.slice(3, .{ .start = quarter, .end = 2 * quarter });
    const c = x.slice(3, .{ .start = 2 * quarter, .end = 3 * quarter });
    const d = x.slice(3, .{ .start = 3 * quarter });
    const rotated = Tensor.concatenate(&.{ b.scale(@as(f32, -1)), a, d.scale(@as(f32, -1)), c }, 3);
    return x.mul(cos.broadcast(x.shape(), &.{ 2, 3 })).add(rotated.mul(sin.broadcast(x.shape(), &.{ 2, 3 })));
}

fn block(w: l.Weights, i: usize, x: Tensor, sine: Tensor, cosine: Tensor) Tensor {
    var base: [64]u8 = undefined;
    var key: [96]u8 = undefined;
    const scope = l.name(&base, "{s}.blocks.{d}", .{ backbone, i });
    const normed = t.norm(w, l.name(&key, "{s}.norm1", .{scope}), x, 1e-6);
    const qkv = t.linear(w, l.name(&key, "{s}.attn.qkv", .{scope}), normed, true)
        .reshape(.{ x.dim(0), x.dim(1), 3, width });
    var q = t.splitHeads(qkv.slice(2, .single(0)), wd.head_count);
    var k = t.splitHeads(qkv.slice(2, .single(1)), wd.head_count);
    if (i >= 4) {
        q = rotateHeads(t.norm(w, l.name(&key, "{s}.attn.q_norm", .{scope}), q, 1e-5), sine, cosine);
        k = rotateHeads(t.norm(w, l.name(&key, "{s}.attn.k_norm", .{scope}), k, 1e-5), sine, cosine);
    }
    const context = t.attendHeads(q, k, t.splitHeads(qkv.slice(2, .single(2)), wd.head_count));
    const projected = t.linear(w, l.name(&key, "{s}.attn.proj", .{scope}), context, true);
    const hidden = x.add(projected.mul(w.get(scope, ".ls1.gamma").broadcast(x.shape(), &.{2})));
    const mlp_input = t.norm(w, l.name(&key, "{s}.norm2", .{scope}), hidden, 1e-6);
    const fc1 = t.gelu(t.linear(w, l.name(&key, "{s}.mlp.fc1", .{scope}), mlp_input, true));
    const fc2 = t.linear(w, l.name(&key, "{s}.mlp.fc2", .{scope}), fc1, true);
    return hidden.add(fc2.mul(w.get(scope, ".ls2.gamma").broadcast(hidden.shape(), &.{2})));
}

fn pose(w: l.Weights, camera: Tensor) Tensor {
    const first = t.linear(w, wd.decoder ++ ".backbone.0", camera, true).relu();
    const last = t.linear(w, wd.decoder ++ ".backbone.2", first, true).relu();
    return Tensor.concatenate(&.{
        t.linear(w, wd.decoder ++ ".fc_t", last, true),
        t.linear(w, wd.decoder ++ ".fc_qvec", last, true),
        t.linear(w, wd.decoder ++ ".fc_fov.0", last, true).relu(),
    }, 1);
}

pub fn elements(views: i64, height: i64, width_: i64) i64 {
    return views * (2 * height * width_ + 7 * @divExact(height, 14) * @divExact(width_, 14) * 64 + 9);
}

pub fn forward(params: []const Tensor, image: Tensor) Tensor {
    const views = image.dim(0);
    const rows = @divExact(image.dim(2), 14);
    const cols = @divExact(image.dim(3), 14);
    const count = 1 + rows * cols;
    const local_phase = angles(rows, cols, views, false);
    const global_phase = angles(rows, cols, views, true);
    const ls, const lc = .{ local_phase.sin(), local_phase.cos() };
    const gs, const gc = .{ global_phase.sin(), global_phase.cos() };
    const w: l.Weights = .{ .schema = &parameters, .tensors = params };
    var x = tokens(w, image);
    var local = x;
    var features: [4]Tensor = undefined;
    var camera: Tensor = undefined;
    for (0..12) |i| {
        if (i == 4) {
            const cls = w.get(backbone ++ ".camera_token", "");
            const first = cls.slice(1, .single(0)).reshape(.{ 1, 1, width });
            const other = cls.slice(1, .single(1)).reshape(.{ 1, 1, width }).broad(.init(.{ views - 1, 1, width }, .f32));
            x = Tensor.concatenate(&.{ Tensor.concatenate(&.{ first, other }, 0), x.slice(1, .{ .start = 1 }) }, 1);
        }
        if (i >= 4 and i % 2 != 0) {
            x = block(w, i, x.reshape(.{ 1, views * count, width }), gs, gc).reshape(.{ views, count, width });
        } else {
            x = block(w, i, x, ls, lc);
            local = x;
        }
        if (i >= 5 and i % 2 != 0) {
            const combined = Tensor.concatenate(&.{ local, t.norm(w, backbone ++ ".norm", x, 1e-5) }, 2);
            features[(i - 5) / 2] = combined.slice(1, .{ .start = 1 });
        }
        if (i == 11)
            camera = Tensor.concatenate(&.{ local.slice(1, .single(0)), x.slice(1, .single(0)) }, 1);
    }
    const depth, const rays = dual.forward(w, features, image.dim(2), image.dim(3));
    return Tensor.concatenate(&.{
        depth.transpose(.{ 0, 2, 3, 1 }).reshape(.{views * 2 * image.dim(2) * image.dim(3)}),
        rays.transpose(.{ 0, 2, 3, 1 }).reshape(.{views * 7 * rows * cols * 64}),
        pose(w, camera).reshape(.{views * 9}),
    }, 0);
}
