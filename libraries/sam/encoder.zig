const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const schema = @import("weights.zig").encoder;

const channels = 768;
const heads = 12;
const grid = 64;

fn relativePositions(weight: Tensor, length: i64) Tensor {
    const rows = Tensor.iota(.init(.{ length, length }, .i32), 0);
    const cols = Tensor.iota(rows.shape(), 1);
    const indices = rows.sub(cols).addConstant(@as(i32, @intCast(length - 1))).withTags(.{ .pos_q, .pos_k });
    return weight.withTags(.{ .relative, .channel }).gather(.{ .relative = indices }, .{})
        .reshape(.{ length, length, weight.dim(1) });
}

fn relativeScores(scores: Tensor, q: Tensor, w: l.Weights, prefix: []const u8, height: i64, width: i64) Tensor {
    var key: [64]u8 = undefined;
    const rh = relativePositions(w.get(l.name(&key, "{s}.attn", .{prefix}), ".rel_pos_h"), height);
    const rw = relativePositions(w.get(l.name(&key, "{s}.attn", .{prefix}), ".rel_pos_w"), width);
    const batches = q.dim(0);
    const q_grid = q.reshape(.{ batches, heads, height, width, @divExact(channels, heads) });
    const rel_h = q_grid.dotGeneral(rh, &.{.{ 4, 2 }}, &.{.{ 2, 0 }}).transpose(.{ 1, 2, 0, 3, 4 });
    const rel_w = q_grid.dotGeneral(rw, &.{.{ 4, 2 }}, &.{.{ 3, 0 }}).transpose(.{ 1, 2, 3, 0, 4 });
    const spatial = scores.reshape(.{ batches, heads, height, width, height, width });
    return spatial.add(rel_h.broadcast(spatial.shape(), &.{ 0, 1, 2, 3, 4 }))
        .add(rel_w.broadcast(spatial.shape(), &.{ 0, 1, 2, 3, 5 })).reshape(scores.shape());
}

fn attention(w: l.Weights, prefix: []const u8, x: Tensor) Tensor {
    var key: [64]u8 = undefined;
    const b, const h, const width, _ = x.dims()[0..4].*;
    const qkv = t.linear(w, l.name(&key, "{s}.attn.qkv", .{prefix}), x, true)
        .reshape(.{ b, h * width, 3, channels });
    const q = t.splitHeads(qkv.slice(2, .single(0)), heads);
    const k = t.splitHeads(qkv.slice(2, .single(1)), heads);
    const v = t.splitHeads(qkv.slice(2, .single(2)), heads);
    const logits = relativeScores(t.scores(q, k), q, w, prefix, h, width);
    const y = t.attend(logits, v).reshape(.{ b, h, width, channels });
    return t.linear(w, l.name(&key, "{s}.attn.proj", .{prefix}), y, true);
}

fn block(w: l.Weights, index: usize, x: Tensor) Tensor {
    var base: [48]u8 = undefined;
    var key: [64]u8 = undefined;
    const prefix = l.name(&base, "image_encoder.blocks.{d}", .{index});
    const normalized = t.norm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 1e-6);
    const global = index == 2 or index == 5 or index == 8 or index == 11;
    const attended = if (global)
        attention(w, prefix, normalized)
    else
        t.windowUnpartition(attention(w, prefix, t.windowPartition(normalized, 14)), grid, grid);
    const y = attended.add(x);
    const mlp_in = t.norm(w, l.name(&key, "{s}.norm2", .{prefix}), y, 1e-6);
    const hidden = t.gelu(t.linear(w, l.name(&key, "{s}.mlp.lin1", .{prefix}), mlp_in, true));
    return y.add(t.linear(w, l.name(&key, "{s}.mlp.lin2", .{prefix}), hidden, true));
}

pub fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    var x = l.conv(w, "image_encoder.patch_embed.proj", input, 16, @splat(0), true)
        .transpose(.{ 0, 2, 3, 1 }).add(w.get("image_encoder.pos_embed", ""));
    for (0..12) |i| x = block(w, i, x);
    var y = l.conv(w, "image_encoder.neck.0", x.transpose(.{ 0, 3, 1, 2 }), 1, @splat(0), false);
    y = t.normNchw(w, "image_encoder.neck.1", y, 1e-6);
    y = l.conv(w, "image_encoder.neck.2", y, 1, @splat(1), false);
    return t.normNchw(w, "image_encoder.neck.3", y, 1e-6);
}
