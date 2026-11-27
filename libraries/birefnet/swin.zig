const rt = @import("zml_lib/runtime");
const wd = @import("weights.zig");
const l = rt.layers;
const t = rt.transformer;
const Tensor = l.Tensor;
const window = wd.window;

fn rotate(x: Tensor, comptime axis: u3, amount: i64) Tensor {
    const split = @mod(amount, x.dim(axis));
    if (split == 0) return x;
    const start = x.dim(axis) - split;
    return Tensor.concatenate(&.{ x.slice(axis, .{ .start = start }), x.slice(axis, .{ .end = start }) }, axis);
}

fn region(x: Tensor, comptime axis: u3, shift: i64) Tensor {
    const position = Tensor.iota(x.shape().setDim(3, 1).withDtype(.i32), axis);
    const end = x.dim(axis);
    return position.cmp(.GE, .scalar(end - window, .i32)).convert(.i32)
        .add(position.cmp(.GE, .scalar(end - shift, .i32)).convert(.i32));
}

fn mask(x: Tensor, shift: i64) Tensor {
    const labels = region(x, 1, shift).scale(@as(i32, 3)).add(region(x, 2, shift));
    const tiles = t.windowPartition(labels.convert(.f32), window).reshape(.{ @divExact(x.dim(1) * x.dim(2), window * window), window * window });
    const shape = @TypeOf(x.shape()).init(.{ tiles.dim(0), tiles.dim(1), tiles.dim(1) }, .f32);
    const rows = tiles.broadcast(shape, &.{ 0, 1 });
    const cols = tiles.broadcast(shape, &.{ 0, 2 });
    return rows.cmp(.NE, cols).convert(.f32).scale(@as(f32, -100));
}

fn block(weights: l.Weights, x: Tensor, level: usize, index: usize) Tensor {
    const height = x.dim(1);
    const width = x.dim(2);
    const channels = x.dim(3);
    const heads = @divExact(channels, 32);
    const length = window * window;
    const shift: i64 = if (index % 2 == 0) 0 else window / 2;
    var scope: [48]u8 = undefined;
    var key: [96]u8 = undefined;
    const name = l.name(&scope, "bb.layers.{d}.blocks.{d}", .{ level, index });
    const normalized = t.norm(weights, l.name(&key, "{s}.norm1", .{name}), x, 1e-5);
    const padded = normalized.withTags(.{ .batch, .row, .col, .channel }).pad(@as(f32, 0), .{
        .row = Tensor.Pad{ .high = @mod(-height, window) },
        .col = Tensor.Pad{ .high = @mod(-width, window) },
    });
    const shifted = rotate(rotate(padded, 1, -shift), 2, -shift);
    const tiled = t.windowPartition(shifted, window);
    const windows = tiled.reshape(.{ tiled.dim(0), length, channels });
    const qkv = t.linear(weights, l.name(&key, "{s}.attn.qkv", .{name}), windows, true)
        .reshape(.{ windows.dim(0), length, 3, channels });
    const q = t.splitHeads(qkv.slice(2, .single(0)), heads);
    const k = t.splitHeads(qkv.slice(2, .single(1)), heads);
    const v = t.splitHeads(qkv.slice(2, .single(2)), heads);
    var logits = t.scores(q, k);
    const bias = weights.get(name, ".attn.relative_bias");
    logits = logits.add(bias.broadcast(logits.shape(), &.{ 1, 2, 3 }));
    if (shift != 0)
        logits = logits.add(mask(padded, shift).broadcast(logits.shape(), &.{ 0, 2, 3 }));
    const projected = t.linear(weights, l.name(&key, "{s}.attn.proj", .{name}), t.attend(logits, v), true);
    const merged = t.windowUnpartition(projected.reshape(tiled.shape()), shifted.dim(1), shifted.dim(2));
    const restored = rotate(rotate(merged, 1, shift), 2, shift).slice(1, .{ .end = height }).slice(2, .{ .end = width });
    const shortcut = x.add(restored);
    const mlp = t.norm(weights, l.name(&key, "{s}.norm2", .{name}), shortcut, 1e-5);
    const a = t.gelu(t.linear(weights, l.name(&key, "{s}.mlp.fc1", .{name}), mlp, true));
    return shortcut.add(t.linear(weights, l.name(&key, "{s}.mlp.fc2", .{name}), a, true));
}

pub fn forward(weights: l.Weights, image: Tensor) [4]Tensor {
    const embedded = l.conv(weights, "bb.patch_embed.proj", image, 4, @splat(0), true).transpose(.{ 0, 2, 3, 1 });
    var x = t.norm(weights, "bb.patch_embed.norm", embedded, 1e-5);
    var out: [4]Tensor = undefined;
    var key: [80]u8 = undefined;
    for (wd.depths, 0..) |count, level| {
        for (0..count) |i| x = block(weights, x, level, i);
        out[level] = t.norm(weights, l.name(&key, "bb.norm{d}", .{level}), x, 1e-5).transpose(.{ 0, 3, 1, 2 });
        if (level == 3) break;
        const h = @divExact(x.dim(1), 2);
        const w = @divExact(x.dim(2), 2);
        const c = x.dim(3);
        const merged = x.reshape(.{ 1, h, 2, w, 2, c }).transpose(.{ 0, 1, 3, 4, 2, 5 }).reshape(.{ 1, h, w, 4 * c });
        const normed = t.norm(weights, l.name(&key, "bb.layers.{d}.downsample.norm", .{level}), merged, 1e-5);
        x = t.linear(weights, l.name(&key, "bb.layers.{d}.downsample.reduction", .{level}), normed, false);
    }
    return out;
}
