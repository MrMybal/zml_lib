const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const schema = @import("weights.zig").encoder;

fn maxPool2x(x: Tensor) Tensor {
    // Hiera pools projected queries and the projected skip at stage changes.
    const even = x.slice(1, .{ .end = x.dim(1) - 1, .step = 2 });
    const odd = x.slice(1, .{ .start = 1, .end = x.dim(1), .step = 2 });
    const row = even.maximum(odd);
    return row.slice(2, .{ .end = row.dim(2) - 1, .step = 2 })
        .maximum(row.slice(2, .{ .start = 1, .end = row.dim(2), .step = 2 }));
}

fn attention(w: l.Weights, prefix: []const u8, x: Tensor, heads: i64, pool: bool) Tensor {
    var key: [88]u8 = undefined;
    const b, const h, const width, _ = x.dims()[0..4].*;
    const joined = t.linear(w, l.name(&key, "{s}.attn.qkv", .{prefix}), x, true);
    const channels = @divExact(joined.dim(3), 3);
    const qkv = joined.reshape(.{ b, h * width, 3, channels });
    const query = qkv.slice(2, .single(0)).reshape(.{ b, h, width, channels });
    const q = if (pool) maxPool2x(query) else query;
    const k = qkv.slice(2, .single(1));
    const v = qkv.slice(2, .single(2));
    const y = t.attention(q.reshape(.{ b, q.dim(1) * q.dim(2), channels }), k, v, heads)
        .reshape(q.shape());
    return t.linear(w, l.name(&key, "{s}.attn.proj", .{prefix}), y, true);
}

const stage_ends = [_]usize{ 1, 4, 20, 23 };
const windows = [_]i64{ 8, 4, 14, 7 };

fn block(w: l.Weights, index: usize, window_stage: usize, x: Tensor, pool: bool) Tensor {
    var base: [64]u8 = undefined;
    var key: [88]u8 = undefined;
    const prefix = l.name(&base, "image_encoder.trunk.blocks.{d}", .{index});
    const window_size = if (index == 12 or index == 16 or index == 20) 0 else windows[window_stage];
    const stage = if (pool) window_stage + 1 else window_stage;
    const nheads: i64 = @as(i64, 2) << @as(u6, @intCast(stage));
    const normalized = t.norm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 1e-6);
    const skip = if (pool) maxPool2x(t.linear(w, l.name(&key, "{s}.proj", .{prefix}), normalized, true)) else x;
    const a = if (window_size == 0)
        attention(w, prefix, normalized, nheads, pool)
    else
        t.windowUnpartition(attention(w, prefix, t.windowPartition(normalized, window_size), nheads, pool), skip.dim(1), skip.dim(2));
    const y = a.add(skip);
    const z = t.norm(w, l.name(&key, "{s}.norm2", .{prefix}), y, 1e-6);
    const hidden = t.gelu(t.linear(w, l.name(&key, "{s}.mlp.layers.0", .{prefix}), z, true));
    return y.add(t.linear(w, l.name(&key, "{s}.mlp.layers.1", .{prefix}), hidden, true));
}

pub fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const projected = l.conv(w, "image_encoder.trunk.patch_embed.proj", input, 4, @splat(3), true);
    const p = w.get("image_encoder.trunk.pos_embed", "");
    const window = w.get("image_encoder.trunk.pos_embed_window", "")
        .broadcast(.init(.{ 1, 112, 32, 8, 32, 8 }, .f32), &.{ 0, 1, 3, 5 })
        .reshape(.{ 1, 112, 256, 256 });
    var x = projected.add(p.add(window)).transpose(.{ 0, 2, 3, 1 });
    var laterals: [4]Tensor = undefined;
    var stage: usize = 0;
    var key: [80]u8 = undefined;
    for (0..24) |i| {
        const transition = i != 0 and stage < 3 and i == stage_ends[stage] + 1;
        x = block(w, i, stage, x, transition);
        if (transition) stage += 1;
        if (i == stage_ends[stage]) laterals[stage] = l.conv(w, l.name(&key, "image_encoder.neck.convs.{d}.conv", .{3 - stage}), x.transpose(.{ 0, 3, 1, 2 }), 1, @splat(0), true);
    }
    const primary = laterals[2].add(l.nearest2x(laterals[3]));
    const s0 = l.conv(w, "sam_mask_decoder.conv_s0", laterals[0], 1, @splat(0), true);
    const s1 = l.conv(w, "sam_mask_decoder.conv_s1", laterals[1], 1, @splat(0), true);
    return Tensor.concatenate(&.{ primary.flatten(), s0.flatten(), s1.flatten() }, 0);
}
