const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;
const m = @import("post_math.zig");
const f = m.scalar;

pub const layers = 3;
pub const levels = layers + 3;
pub const initial_sigma = 1.6;
pub const minimum_side = 20;
pub const border = 5;

pub fn i(value: i32) Tensor {
    return .scalar(value, .i32);
}

pub fn ramp(size: i64) Tensor {
    return Tensor.iota(.init(.{size}, .i32), 0);
}

pub fn rowsAt(rows: Tensor, indices: Tensor) Tensor {
    return rows.withTags(.{ .query, .value }).gather(.{ .query = indices.withTags(.{.feature}) }, .{})
        .reshape(.{ indices.dim(0), rows.dim(1) });
}

fn upsample(image: Tensor, comptime axis: usize) Tensor {
    const size = image.dim(axis);
    const position = ramp(2 * size).convert(.f32).scale(@as(f32, 0.5)).subConstant(@as(f32, 0.25));
    const base = position.floor();
    const ratio = position.sub(base);
    const low = base.convert(.i32).clamp(i(0), i(@intCast(size - 1)));
    const high = base.convert(.i32).add(i(1)).clamp(i(0), i(@intCast(size - 1)));
    const tagged = image.withTags(.{ .row, .column });
    const a = if (axis == 0) tagged.gather(.{ .row = low }, .{}) else tagged.gather(.{ .column = low }, .{});
    const b = if (axis == 0) tagged.gather(.{ .row = high }, .{}) else tagged.gather(.{ .column = high }, .{});
    const shape = image.shape().setDim(axis, size * 2);
    const a_ = a.reshape(shape);
    return a_.add(b.reshape(shape).sub(a_).mul(ratio.broadcast(shape, &.{axis})));
}

fn gaussian(comptime sigma: f64) [2 * radius(sigma) + 1]f32 {
    var kernel: [2 * radius(sigma) + 1]f32 = undefined;
    var sum: f64 = 0;
    for (&kernel, 0..) |*output, index| {
        const x: f64 = @floatFromInt(@as(i64, @intCast(index)) - radius(sigma));
        const value = @exp(-x * x / (2 * sigma * sigma));
        output.* = @floatCast(value);
        sum += value;
    }
    for (&kernel) |*output| output.* /= @as(f32, @floatCast(sum));
    return kernel;
}

fn radius(comptime sigma: f64) i64 {
    return @intFromFloat(@ceil(4 * sigma));
}

fn blurAxis(image: Tensor, comptime axis: usize, comptime sigma: f64) Tensor {
    const n = radius(sigma);
    const weights = comptime gaussian(sigma);
    const length = image.dim(axis);
    const before = image.slice(axis, .{ .start = 1, .end = 1 + n }).reverse(.{axis});
    const after = image.slice(axis, .{ .start = length - 1 - n, .end = length - 1 }).reverse(.{axis});
    const padded = Tensor.concatenate(&.{ before, image, after }, axis);
    const kernel_shape: zml.Shape = if (axis == 0) .init(.{ 1, 1, weights.len, 1 }, .f32) else .init(.{ 1, 1, 1, weights.len }, .f32);
    const kernel = Tensor.constantTensor(kernel_shape, std.mem.asBytes(&weights));
    return padded.reshape(.{ 1, 1, padded.dim(0), padded.dim(1) }).conv2d(kernel, .{}).reshape(image.shape());
}

fn blur(image: Tensor, comptime sigma: f64) Tensor {
    return blurAxis(blurAxis(image, 1, sigma), 0, sigma);
}

pub fn first(image: Tensor) Tensor {
    const channels = image.convert(.f32);
    // The GVHMR matcher uses BGR luma coefficients on the decoded byte order.
    const gray = channels.slice(2, .single(0)).scale(@as(f32, 0.114))
        .add(channels.slice(2, .single(1)).scale(@as(f32, 0.587)))
        .add(channels.slice(2, .single(2)).scale(@as(f32, 0.299)));
    const pixels = gray.addConstant(@as(f32, 0.5)).floor().scale(@as(f32, 1.0 / 255.0));
    return blur(upsample(upsample(pixels, 1), 0), @sqrt(initial_sigma * initial_sigma - 1));
}

pub fn octave(base: Tensor) struct { Tensor, Tensor } {
    const h = base.dim(0);
    const w = base.dim(1);
    var gaussians: [levels]Tensor = undefined;
    gaussians[0] = base;
    comptime var last_sigma: f64 = initial_sigma;
    inline for (1..levels) |level| {
        const sigma = comptime initial_sigma * std.math.pow(f64, 2, @as(f64, @floatFromInt(level)) / layers);
        gaussians[level] = blur(gaussians[level - 1], @sqrt(sigma * sigma - last_sigma * last_sigma));
        last_sigma = sigma;
    }
    const next = gaussians[layers].slice(0, .{ .end = @divFloor(h, 2) * 2, .step = 2 })
        .slice(1, .{ .end = @divFloor(w, 2) * 2, .step = 2 });
    const space = Tensor.stack(&gaussians, 0, .scale).reshape(.{ levels, h, w });
    return .{ space, next };
}

pub fn planeIndex(x: Tensor, y: Tensor, scale: Tensor, width: i64, height: i64) Tensor {
    return scale.mul(i(@intCast(width * height))).add(y.mul(i(@intCast(width)))).add(x);
}

fn scan(values: Tensor, comptime axis: usize) Tensor {
    var prefix = values;
    const count = values.dim(axis);
    var distance: i64 = 1;
    while (distance < count) : (distance *= 2) {
        const zeros = i(0).broad(values.shape().setDim(axis, distance));
        prefix = prefix.add(Tensor.concatenate(&.{ zeros, prefix.slice(axis, .{ .end = count - distance }) }, axis));
    }
    return prefix;
}

// A block scan and a vectorized lower bound preserve order without atomics.
pub fn compactIndices(valid: Tensor, capacity: i64) struct { index: Tensor, valid: Tensor } {
    const count = valid.dim(0);
    const block = 128;
    const blocks = @divFloor(count + block - 1, block);
    const tail = blocks * block - count;
    const flags = valid.convert(.i32);
    const padded = if (tail == 0) flags else Tensor.concatenate(&.{ flags, i(0).broad(.init(.{tail}, .i32)) }, 0);
    const local = scan(padded.reshape(.{ blocks, block }), 1);
    const sums = local.slice(1, .single(block - 1));
    const offset = scan(sums, 0).sub(sums).reshape(.{ blocks, 1 });
    const prefix = local.add(offset).reshape(.{blocks * block}).slice(0, .{ .end = count });
    const wanted = ramp(capacity).add(i(1));
    const totals = prefix.slice(0, .single(count - 1));
    var index = i(-1).broad(wanted.shape());
    var distance: i64 = 1;
    while (distance < count) : (distance *= 2) {}
    distance = @divFloor(distance, 2);
    while (distance > 0) : (distance = @divFloor(distance, 2)) {
        const step = index.add(i(@intCast(distance)));
        const at = prefix.withTags(.{.row}).gather(.{ .row = step.clamp(i(0), i(@intCast(count - 1))).withTags(.{.target}) }, .{})
            .reshape(wanted.shape());
        const better = step.cmp(.LT, i(@intCast(count))).logical(.AND, at.cmp(.LT, wanted));
        index = better.select(step, index);
    }
    return .{ .index = index.add(i(1)).minimum(i(@intCast(count - 1))), .valid = wanted.cmp(.LE, totals) };
}

pub fn compact(rows: Tensor, valid: Tensor, capacity: i64) Tensor {
    const selected = compactIndices(valid.reshape(.{rows.dim(0)}), capacity);
    const result = rowsAt(rows, selected.index);
    return selected.valid.broadcast(result.shape().withDtype(.bool), &.{0}).select(result, f(0));
}
