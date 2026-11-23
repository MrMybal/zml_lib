const std = @import("std");
const zml = @import("zml/core");
const m = @import("post_math.zig");
const p = @import("sift_scale.zig");
const Tensor = zml.Tensor;
const f = m.scalar;
const i = p.i;

pub const descriptor_size = 128;
pub const feature_width = 3 + descriptor_size;
pub const oriented_width = 6;
pub const Settings = struct { pixel_scale: f32 };

const orientation_bins = 36;
const sample_side = 20;
const samples = sample_side * sample_side;
const cells = 4;
const bins = 8;
const tau: f32 = 2 * std.math.pi;

fn square(x: Tensor) Tensor {
    return x.mul(x);
}

fn gradients(gauss: Tensor) Tensor {
    const h = gauss.dim(1);
    const w = gauss.dim(2);
    const col = f(0).broad(gauss.shape().setDim(2, 1));
    const row = f(0).broad(gauss.shape().setDim(1, 1));
    const dx = gauss.slice(2, .{ .start = 2 }).sub(gauss.slice(2, .{ .end = w - 2 }));
    const dy = gauss.slice(1, .{ .start = 2 }).sub(gauss.slice(1, .{ .end = h - 2 }));
    const gx = Tensor.concatenate(&.{ col, dx, col }, 2);
    const gy = Tensor.concatenate(&.{ row, dy, row }, 1);
    return Tensor.stack(&.{ gx, gy }, 3, .vector).reshape(.{ p.levels * h * w, 2 });
}

fn read(field: Tensor, x: Tensor, y: Tensor, layer: Tensor, w: i64, h: i64) Tensor {
    const index = p.planeIndex(x.clamp(i(0), i(@intCast(w - 1))), y.clamp(i(0), i(@intCast(h - 1))), layer.clamp(i(0), i(p.levels - 1)).broad(x.shape()), w, h);
    return field.withTags(.{ .pixel, .vector }).gather(.{ .pixel = index.withTags(.{ .feature, .sample }) }, .{})
        .reshape(.{ x.dim(0), x.dim(1), 2 });
}

fn inside(x: Tensor, y: Tensor, w: i64, h: i64) Tensor {
    return x.cmp(.GE, f(1)).logical(.AND, x.cmp(.LE, f(@floatFromInt(w - 2))))
        .logical(.AND, y.cmp(.GE, f(1))).logical(.AND, y.cmp(.LE, f(@floatFromInt(h - 2))));
}

fn circular(position: Tensor, comptime period: f32) Tensor {
    return position.sub(position.scale(@as(f32, 1 / period)).floor().scale(period));
}

fn rotate(hist: Tensor, comptime step: i64) Tensor {
    const offset = @mod(step, orientation_bins);
    return Tensor.concatenate(&.{ hist.slice(1, .{ .start = offset }), hist.slice(1, .{ .end = offset }) }, 1);
}

fn binAt(hist: Tensor, index: Tensor) Tensor {
    const positions = Tensor.iota(hist.shape().withDtype(.i32), 1);
    return positions.cmp(.EQ, index.broad(positions.shape())).select(hist, f(0)).sum(1);
}

fn angleAt(hist: Tensor, before: Tensor, after: Tensor, peak: Tensor) Tensor {
    const lo = binAt(before, peak);
    const hi = binAt(after, peak);
    const center = binAt(hist, peak);
    const denom = center.scale(@as(f32, 2)).sub(lo).sub(hi).maximum(f(1e-12));
    const delta = hi.sub(lo).scale(@as(f32, 0.5)).div(denom).clamp(f(-0.5), f(0.5));
    return peak.convert(.f32).add(delta).scale(@as(f32, tau / orientation_bins));
}

pub fn orient(space: Tensor, keypoints: Tensor) struct { Tensor, Tensor } {
    const field = gradients(space);
    const h = space.dim(1);
    const w = space.dim(2);
    const n = keypoints.dim(0);
    const radius = 17;
    const side = 2 * radius + 1;
    const positions = p.ramp(side * side);
    const patch: zml.Shape = .init(.{ n, side * side }, .f32);
    const dx = positions.remainder(i(side)).sub(i(radius)).convert(.f32).broadcast(patch, &.{1});
    const dy = positions.div(i(side)).sub(i(radius)).convert(.f32).broadcast(patch, &.{1});
    const x = m.component(keypoints, 0).round().broad(patch).add(dx);
    const y = m.component(keypoints, 1).round().broad(patch).add(dy);
    const layer = m.component(keypoints, 2).convert(.i32);
    const sigma = m.component(keypoints, 3).scale(@as(f32, 1.5)).maximum(f(1e-3));
    const window = sigma.scale(@as(f32, 3)).round().broad(patch);
    const pixel_ok = inside(x, y, w, h).logical(.AND, dx.abs().cmp(.LE, window)).logical(.AND, dy.abs().cmp(.LE, window));
    const sample = read(field, x.convert(.i32), y.convert(.i32), layer, w, h);
    const gx = sample.slice(2, .single(0));
    const gy = sample.slice(2, .single(1));
    const weight = square(dx).add(square(dy)).scale(@as(f32, -0.5)).div(square(sigma)).exp();
    const magnitude = pixel_ok.select(square(gx).add(square(gy)).sqrt().mul(weight), f(0));
    const bucket = m.atan2(gy, gx).scale(@as(f32, orientation_bins / tau)).round().convert(.i32)
        .add(i(orientation_bins)).remainder(i(orientation_bins));
    const hist_shape: zml.Shape = .init(.{ n, side * side, orientation_bins }, .i32);
    const matches = Tensor.iota(hist_shape, 2).cmp(.EQ, bucket.broadcast(hist_shape, &.{ 0, 1 }));
    const hist = matches.select(magnitude.broadcast(hist_shape.withDtype(.f32), &.{ 0, 1 }), f(0)).sum(1).squeeze(1);
    const smooth = rotate(hist, -2).add(rotate(hist, 2)).add(rotate(hist, -1).add(rotate(hist, 1)).scale(@as(f32, 4)))
        .add(hist.scale(@as(f32, 6))).scale(@as(f32, 1.0 / 16.0));
    const before = rotate(smooth, -1);
    const after = rotate(smooth, 1);
    const peak = smooth.cmp(.GT, before).logical(.AND, smooth.cmp(.GT, after)).select(smooth, f(0));
    const first = peak.argMax(1);
    const bins_index = Tensor.iota(peak.shape().withDtype(.i32), 1);
    const other = bins_index.cmp(.NE, first.indices.broad(bins_index.shape())).select(peak, f(0)).argMax(1);
    const valid = m.component(keypoints, 5).cmp(.GT, f(0)).logical(.AND, first.values.cmp(.GT, f(0)));
    const secondary = valid.logical(.AND, other.values.cmp(.GE, first.values.scale(@as(f32, 0.8))));
    const location = keypoints.slice(1, .{ .end = 4 });
    const main = m.join(&.{ location, angleAt(smooth, before, after, first.indices), valid.convert(.f32) });
    const alternate = m.join(&.{ location, angleAt(smooth, before, after, other.indices), secondary.convert(.f32) });
    const oriented = Tensor.stack(&.{ main, alternate }, 1, .orientation).reshape(.{ 2 * n, oriented_width });
    return .{ field, p.compact(oriented, m.component(oriented, 5).cmp(.GT, f(0)), n) };
}

fn readLinear(field: Tensor, x: Tensor, y: Tensor, layer: Tensor, w: i64, h: i64) Tensor {
    const base_x = x.floor();
    const base_y = y.floor();
    const ix = base_x.convert(.i32);
    const iy = base_y.convert(.i32);
    const a = read(field, ix, iy, layer, w, h);
    const b = read(field, ix.add(i(1)), iy, layer, w, h);
    const c = read(field, ix, iy.add(i(1)), layer, w, h);
    const d = read(field, ix.add(i(1)), iy.add(i(1)), layer, w, h);
    const u = x.sub(base_x).broadcast(a.shape(), &.{ 0, 1 });
    const v = y.sub(base_y).broadcast(a.shape(), &.{ 0, 1 });
    const top = a.add(b.sub(a).mul(u));
    const bottom = c.add(d.sub(c).mul(u));
    return top.add(bottom.sub(top).mul(v));
}

fn sampleCoord(index: usize) f64 {
    return (@as(f64, @floatFromInt(index)) + 0.5) * (5.0 / @as(f64, sample_side)) - 2.5;
}

fn weights() [cells * cells * samples]f32 {
    @setEvalBranchQuota(100000);
    var result: [cells * cells * samples]f32 = undefined;
    for (&result, 0..) |*out, index| {
        const cell = index / samples;
        const sample = index % samples;
        const x = sampleCoord(sample % sample_side);
        const y = sampleCoord(sample / sample_side);
        const cx = @as(f64, @floatFromInt(cell % cells)) - 1.5;
        const cy = @as(f64, @floatFromInt(cell / cells)) - 1.5;
        const spatial = @max(0, 1 - @abs(x - cx)) * @max(0, 1 - @abs(y - cy));
        out.* = @floatCast(spatial * @exp(-(x * x + y * y) / 8));
    }
    return result;
}

pub fn describe(settings: Settings, field: Tensor, keypoints: Tensor, dimensions: Dimensions) Tensor {
    const w = dimensions.width;
    const h = dimensions.height;
    const n = keypoints.dim(0);
    const grid = p.ramp(samples);
    const spacing: f32 = 5.0 / @as(f32, sample_side);
    const origin: f32 = 2.5 - 0.5 * spacing;
    const sx = grid.remainder(i(sample_side)).convert(.f32).scale(spacing).subConstant(origin);
    const sy = grid.div(i(sample_side)).convert(.f32).scale(spacing).subConstant(origin);
    const screen_x = m.component(keypoints, 0);
    const screen_y = m.component(keypoints, 1);
    const angle = m.component(keypoints, 4);
    const cos = angle.cos();
    const sin = angle.sin();
    const scale = m.component(keypoints, 3).scale(@as(f32, 3));
    const patch: zml.Shape = .init(.{ n, samples }, .f32);
    const x_local = sx.broadcast(patch, &.{1}).mul(scale);
    const y_local = sy.broadcast(patch, &.{1}).mul(scale);
    const x = x_local.mul(cos).sub(y_local.mul(sin)).add(screen_x);
    const y = x_local.mul(sin).add(y_local.mul(cos)).add(screen_y);
    const grad = readLinear(field, x, y, m.component(keypoints, 2).convert(.i32), w, h);
    const dx = grad.slice(2, .single(0));
    const dy = grad.slice(2, .single(1));
    const norm = inside(x, y, w, h).select(square(dx).add(square(dy)).sqrt(), f(0));
    const orientation = circular(m.atan2(dy, dx).sub(angle).scale(@as(f32, bins / tau)), bins);

    const shape: zml.Shape = .init(.{ n, samples, bins }, .f32);
    const bin = Tensor.iota(shape.withDtype(.i32), 2).convert(.f32);
    const delta = orientation.broadcast(shape, &.{ 0, 1 }).sub(bin).abs();
    const distance = delta.minimum(f(bins).sub(delta));
    const vote = f(1).sub(distance).maximum(f(0)).mul(norm.broadcast(shape, &.{ 0, 1 }));
    const spatial = comptime weights();
    const bin_weights = Tensor.constantTensor(.init(.{ cells * cells, samples }, .f32), std.mem.asBytes(&spatial));
    const histogram = bin_weights.dotGeneral(vote, &.{.{ 1, 1 }}, &.{}).transpose(.{ 1, 0, 2 }).reshape(.{ n, descriptor_size });
    const first_norm = m.norm(histogram);
    const clipped = histogram.div(first_norm.maximum(f(1e-12)).broad(histogram.shape())).minimum(f(0.2));
    const normalized = m.normalize(clipped, 1e-12);
    const valid = m.component(keypoints, 5).cmp(.GT, f(0)).logical(.AND, first_norm.cmp(.GT, f(1e-12)));
    const rows = m.join(&.{ screen_x.scale(settings.pixel_scale), screen_y.scale(settings.pixel_scale), valid.convert(.f32), normalized });
    return valid.broad(rows.shape().withDtype(.bool)).select(rows, f(0));
}

pub const Dimensions = struct { height: i64, width: i64 };

pub const match_width = 5;

pub fn match(left: Tensor, right: Tensor) Tensor {
    const a = left.slice(1, .{ .start = 3 });
    const b = right.slice(1, .{ .start = 3 });
    const product = a.dotGeneral(b, &.{.{ 1, 1 }}, &.{});
    const right_ok = m.component(right, 2).reshape(.{right.dim(0)}).cmp(.GT, f(0));
    // Square L2 on unit-normalized descriptors. Use a stable two-neighbor
    // selection and square the original 0.7 test rather than taking square roots.
    const cost = f(2).sub(product.scale(@as(f32, 2))).maximum(f(0));
    const score = right_ok.broadcast(cost.shape().withDtype(.bool), &.{1}).select(cost.negate(), f(-8));
    const first = score.argMax(1);
    const rank = Tensor.iota(score.shape().withDtype(.i32), 1);
    const second = rank.cmp(.EQ, first.indices.broad(rank.shape())).select(f(-8), score).argMax(1);
    const passed = first.values.cmp(.GT, second.values.scale(@as(f32, 0.49)))
        .logical(.AND, second.values.cmp(.GT, f(-8))).logical(.AND, m.component(left, 2).cmp(.GT, f(0)));
    // Address coordinates in the original row stride. Passing a column
    // subview to a row gather loses that stride in the CUDA lowering.
    const indices_shape: zml.Shape = .init(.{ left.dim(0), 2 }, .i32);
    const addresses = p.ramp(2).broadcast(indices_shape, &.{1}).add(first.indices.mul(i(feature_width)));
    const nearest = right.reshape(.{right.dim(0) * feature_width}).withTags(.{.value})
        .gather(.{ .value = addresses.withTags(.{ .match, .coordinate }) }, .{}).reshape(indices_shape.withDtype(.f32));
    return m.join(&.{ left.slice(1, .{ .end = 2 }), nearest, passed.convert(.f32) });
}
