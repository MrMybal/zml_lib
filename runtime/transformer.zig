const std = @import("std");
const l = @import("layers.zig");
const Tensor = l.Tensor;

pub fn linearParameters(x: Tensor, weight: Tensor, bias: ?Tensor) Tensor {
    const axis = x.rank() - 1;
    const channels = x.dim(axis);
    const out_channels = weight.dim(0);
    const flattened = x.reshape(.{ @divExact(@as(i64, @intCast(x.shape().count())), channels), channels });
    const y = flattened.dotGeneral(weight, &.{.{ 1, 1 }}, &.{});
    return (if (bias) |b| l.biasAdd(y, b, 1) else y).reshape(x.shape().setDim(axis, out_channels));
}

pub fn linear(w: l.Weights, name: []const u8, x: Tensor, has_bias: bool) Tensor {
    return linearParameters(x, w.get(name, ".weight"), if (has_bias) w.get(name, ".bias") else null);
}

pub fn normAffineFree(x: Tensor, eps: f32) Tensor {
    const axis = x.rank() - 1;
    const centered = x.sub(x.mean(axis).broad(x.shape()));
    return centered.mul(centered.mul(centered).mean(axis).addConstant(eps).rsqrt().broad(x.shape()));
}

pub fn norm(w: l.Weights, name: []const u8, x: Tensor, eps: f32) Tensor {
    const channels = x.dim(x.rank() - 1);
    const rows = x.reshape(.{ @divExact(@as(i64, @intCast(x.shape().count())), channels), channels });
    const normalized = normAffineFree(rows, eps);
    return l.biasAdd(normalized.mul(w.get(name, ".weight").broadcast(rows.shape(), &.{1})), w.get(name, ".bias"), 1)
        .reshape(x.shape());
}

pub fn normNchw(w: l.Weights, name: []const u8, x: Tensor, eps: f32) Tensor {
    return norm(w, name, x.transpose(.{ 0, 2, 3, 1 }), eps).transpose(.{ 0, 3, 1, 2 });
}

pub fn gelu(x: Tensor) Tensor {
    return x.mul(x.scale(@as(f32, 0.7071067811865476)).erf().addConstant(@as(f32, 1))).scale(@as(f32, 0.5));
}

pub fn splitHeads(x: Tensor, heads: i64) Tensor {
    return x.reshape(.{ x.dim(0), x.dim(1), heads, @divExact(x.dim(2), heads) }).transpose(.{ 0, 2, 1, 3 });
}

pub fn attend(logits: Tensor, v: Tensor) Tensor {
    const y = logits.softmax(3).dotGeneral(v, &.{.{ 3, 2 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
    return y.transpose(.{ 0, 2, 1, 3 }).reshape(.{ y.dim(0), y.dim(2), y.dim(1) * y.dim(3) });
}

pub fn scores(q: Tensor, k: Tensor) Tensor {
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(q.dim(3))));
    return q.scale(scale).dotGeneral(k, &.{.{ 3, 3 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
}

pub fn attention(q: Tensor, k: Tensor, v: Tensor, heads: i64) Tensor {
    return attend(scores(splitHeads(q, heads), splitHeads(k, heads)), splitHeads(v, heads));
}

pub fn deconv2x(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    return deconvNonoverlap(w, name, x, 2);
}

pub fn deconvNonoverlap(w: l.Weights, name: []const u8, x: Tensor, stride: i64) Tensor {
    const n, const c, const h, const width = x.dims()[0..4].*;
    const kernel = w.get(name, ".weight");
    const out = kernel.dim(1);
    // Equal kernel size and stride produce an independent tile for each source pixel.
    std.debug.assert(kernel.dim(2) == stride and kernel.dim(3) == stride);
    const filters = kernel.transpose(.{ 1, 2, 3, 0 }).reshape(.{ out * stride * stride, c });
    const y = linearParameters(x.transpose(.{ 0, 2, 3, 1 }), filters, null);
    const expanded = y.reshape(.{ n, h, width, out, stride, stride }).transpose(.{ 0, 3, 1, 4, 2, 5 })
        .reshape(.{ n, out, h * stride, width * stride });
    return l.biasAdd(expanded, w.get(name, ".bias"), 1);
}

pub fn windowPartition(x: Tensor, win: i64) Tensor {
    const height = x.dim(1);
    const width = x.dim(2);
    const channels = x.dim(3);
    const rows = @divFloor(height + win - 1, win);
    const cols = @divFloor(width + win - 1, win);
    const y = x.withTags(.{ .batch, .y, .x, .channel }).pad(@as(f32, 0), .{
        .y = Tensor.Pad{ .high = rows * win - height },
        .x = Tensor.Pad{ .high = cols * win - width },
    });
    return y.reshape(.{ rows, win, cols, win, channels }).transpose(.{ 0, 2, 1, 3, 4 })
        .reshape(.{ rows * cols, win, win, channels });
}

pub fn windowUnpartition(x: Tensor, height: i64, width: i64) Tensor {
    const win = x.dim(1);
    const channels = x.dim(3);
    const rows = @divFloor(height + win - 1, win);
    const cols = @divFloor(width + win - 1, win);
    return x.reshape(.{ rows, cols, win, win, channels }).transpose(.{ 0, 2, 1, 3, 4 })
        .reshape(.{ 1, rows * win, cols * win, channels })
        .slice(1, .{ .end = height }).slice(2, .{ .end = width });
}
