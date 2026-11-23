const std = @import("std");
const zml = @import("zml/core");
const m = @import("post_math.zig");
const p = @import("sift_scale.zig");
const Tensor = zml.Tensor;
const f = m.scalar;
const i = p.i;

pub const Search = struct { capacity: i64 };
// x, y, integer Gaussian layer, localized sigma, contrast, valid
pub const point_width = 6;
const contrast: f32 = 0.04 / @as(f32, p.layers);

fn maxNeighbors(x: Tensor, comptime axis: usize) Tensor {
    const outside = f(-1).broad(x.shape().setDim(axis, 1));
    const before = Tensor.concatenate(&.{ outside, x.slice(axis, .{ .end = x.dim(axis) - 1 }) }, axis);
    const after = Tensor.concatenate(&.{ x.slice(axis, .{ .start = 1 }), outside }, axis);
    return x.maximum(before).maximum(after);
}

fn extremaWindow(x: Tensor) Tensor {
    return maxNeighbors(maxNeighbors(maxNeighbors(x, 2), 1), 0);
}

fn inside(x: Tensor, y: Tensor, h: i64, w: i64) Tensor {
    return x.cmp(.GE, i(p.border)).logical(.AND, x.cmp(.LT, i(@intCast(w - p.border))))
        .logical(.AND, y.cmp(.GE, i(p.border))).logical(.AND, y.cmp(.LT, i(@intCast(h - p.border))));
}

fn candidates(dog: Tensor, capacity: i64) Tensor {
    const h = dog.dim(1);
    const w = dog.dim(2);
    const dims: zml.Shape = .init(.{ p.layers, h, w }, .i32);
    const center = dog.slice(0, .{ .start = 1, .end = 1 + p.layers });
    const positive = center.cmp(.GE, extremaWindow(dog).slice(0, .{ .start = 1, .end = 1 + p.layers }));
    const negative = center.negate().cmp(.GE, extremaWindow(dog.negate()).slice(0, .{ .start = 1, .end = 1 + p.layers }));
    const response = center.abs();
    const border_ok = inside(.iota(dims, 2), .iota(dims, 1), h, w);
    const valid = positive.logical(.OR, negative).logical(.AND, border_ok).logical(.AND, response.cmp(.GT, f(contrast / 2)));

    const found = p.compactIndices(valid.transpose(.{ 1, 2, 0 }).reshape(.{p.layers * h * w}), capacity);
    const index = found.index.reshape(.{ capacity, 1 });
    const pixel = index.div(i(p.layers));
    return m.join(&.{
        pixel.remainder(i(@intCast(w))).convert(.f32),
        pixel.div(i(@intCast(w))).convert(.f32),
        index.remainder(i(p.layers)).add(i(1)).convert(.f32),
        found.valid.reshape(.{ capacity, 1 }).convert(.f32),
    });
}

fn square(x: Tensor) Tensor {
    return x.mul(x);
}

const Fit = struct { dx: Tensor, dy: Tensor, ds: Tensor, response: Tensor, accepted: Tensor };

const Samples = enum {
    center,
    left,
    right,
    up,
    down,
    before,
    after,
    xy_pp,
    xy_nn,
    xy_pn,
    xy_np,
    xs_pp,
    xs_nn,
    xs_pn,
    xs_np,
    ys_pp,
    ys_nn,
    ys_pn,
    ys_np,
};

fn neighborhood(dog: Tensor, x: Tensor, y: Tensor, layer: Tensor) Tensor {
    const offsets = [_][3]i32{
        .{ 0, 0, 0 },  .{ -1, 0, 0 },  .{ 1, 0, 0 },   .{ 0, -1, 0 }, .{ 0, 1, 0 },  .{ 0, 0, -1 },  .{ 0, 0, 1 },
        .{ 1, 1, 0 },  .{ -1, -1, 0 }, .{ 1, -1, 0 },  .{ -1, 1, 0 }, .{ 1, 0, 1 },  .{ -1, 0, -1 }, .{ 1, 0, -1 },
        .{ -1, 0, 1 }, .{ 0, 1, 1 },   .{ 0, -1, -1 }, .{ 0, 1, -1 }, .{ 0, -1, 1 },
    };
    const w: i32 = @intCast(dog.dim(2));
    const h: i32 = @intCast(dog.dim(1));
    var address: [offsets.len]i32 = undefined;
    for (&address, offsets) |*dst, offset| dst.* = offset[0] + w * (offset[1] + h * offset[2]);
    const shifts = Tensor.constantTensor(.init(.{offsets.len}, .i32), std.mem.asBytes(&address));
    const shape: zml.Shape = .init(.{ x.dim(0), offsets.len }, .i32);
    const center = p.planeIndex(x, y, layer, w, h).broad(shape);
    const flat = dog.reshape(.{dog.shape().count()}).withTags(.{.pixel});
    const index = center.add(shifts.broadcast(shape, &.{1})).clamp(i(0), i(@intCast(flat.dim(0) - 1)));
    return flat.gather(.{ .pixel = index.withTags(.{ .feature, .offset }) }, .{}).reshape(shape.withDtype(.f32));
}

fn value(points: Tensor, at: Samples) Tensor {
    return m.component(points, @backingInt(at));
}

fn local(dog: Tensor, x: Tensor, y: Tensor, layer: Tensor) Fit {
    const points = neighborhood(dog, x, y, layer);
    const v = value(points, .center);
    const left = value(points, .left);
    const right = value(points, .right);
    const up = value(points, .up);
    const down = value(points, .down);
    const before = value(points, .before);
    const after = value(points, .after);
    const gx = right.sub(left).scale(@as(f32, 0.5));
    const gy = down.sub(up).scale(@as(f32, 0.5));
    const gs = after.sub(before).scale(@as(f32, 0.5));
    const twice = v.scale(@as(f32, 2));
    const xx = left.add(right).sub(twice);
    const yy = up.add(down).sub(twice);
    const ss = before.add(after).sub(twice);
    const xy = value(points, .xy_pp).add(value(points, .xy_nn)).sub(value(points, .xy_pn)).sub(value(points, .xy_np)).scale(@as(f32, 0.25));
    const xs = value(points, .xs_pp).add(value(points, .xs_nn)).sub(value(points, .xs_pn)).sub(value(points, .xs_np)).scale(@as(f32, 0.25));
    const ys = value(points, .ys_pp).add(value(points, .ys_nn)).sub(value(points, .ys_pn)).sub(value(points, .ys_np)).scale(@as(f32, 0.25));
    const a = yy.mul(ss).sub(square(ys));
    const b = xs.mul(ys).sub(xy.mul(ss));
    const c = xy.mul(ys).sub(xs.mul(yy));
    const d = xx.mul(ss).sub(square(xs));
    const e = xy.mul(xs).sub(xx.mul(ys));
    const area = xx.mul(yy).sub(square(xy));
    const det = xx.mul(a).add(xy.mul(b)).add(xs.mul(c));
    const enough = det.abs().cmp(.GT, f(1e-13));
    const gain = f(-1).div(enough.select(det, f(1)));
    const dx = gx.mul(a).add(gy.mul(b)).add(gs.mul(c)).mul(gain);
    const dy = gx.mul(b).add(gy.mul(d)).add(gs.mul(e)).mul(gain);
    const ds = gx.mul(c).add(gy.mul(e)).add(gs.mul(area)).mul(gain);
    const response = v.add(gx.mul(dx).add(gy.mul(dy)).add(gs.mul(ds)).scale(@as(f32, 0.5))).abs();
    const edge_ok = square(xx.add(yy)).scale(@as(f32, 10)).cmp(.LT, area.scale(@as(f32, 121)));
    const accepted = enough.logical(.AND, edge_ok).logical(.AND, response.cmp(.GE, f(contrast)));
    return .{ .dx = dx, .dy = dy, .ds = ds, .response = response, .accepted = accepted };
}

fn maxOffset(fit: Fit) Tensor {
    return fit.dx.abs().maximum(fit.dy.abs()).maximum(fit.ds.abs());
}

const Localize = struct {
    pub const State = struct { step: Tensor, dog: Tensor, x: Tensor, y: Tensor, layer: Tensor, valid: Tensor };

    pub fn cond(_: @This(), state: State) Tensor {
        return state.step.cmp(.LT, i(4));
    }

    pub fn body(_: @This(), state: State) State {
        const fit = local(state.dog, state.x, state.y, state.layer);
        const shift = maxOffset(fit).cmp(.GE, f(0.5)).logical(.AND, state.valid);
        const x = state.x.add(shift.select(fit.dx.clamp(f(-4), f(4)).round().convert(.i32), i(0)));
        const y = state.y.add(shift.select(fit.dy.clamp(f(-4), f(4)).round().convert(.i32), i(0)));
        const layer = state.layer.add(shift.select(fit.ds.clamp(f(-4), f(4)).round().convert(.i32), i(0)));
        const valid = state.valid.logical(.AND, inside(x, y, state.dog.dim(1), state.dog.dim(2)))
            .logical(.AND, layer.cmp(.GE, i(1))).logical(.AND, layer.cmp(.LE, i(p.layers)));
        return .{ .step = state.step.add(i(1)), .dog = state.dog, .x = x, .y = y, .layer = layer, .valid = valid };
    }
};

pub fn forward(search: Search, space: Tensor) Tensor {
    const dog = space.slice(0, .{ .start = 1 }).sub(space.slice(0, .{ .end = p.levels - 1 }));
    const coarse = candidates(dog, search.capacity);
    const key = zml.ops.while_(Localize, .{}, .{
        .step = i(0),
        .dog = dog,
        .x = m.component(coarse, 0).convert(.i32),
        .y = m.component(coarse, 1).convert(.i32),
        .layer = m.component(coarse, 2).convert(.i32),
        .valid = m.component(coarse, 3).cmp(.GT, f(0)),
    });
    const fit = local(dog, key.x, key.y, key.layer);
    const valid = key.valid.logical(.AND, fit.accepted).logical(.AND, maxOffset(fit).cmp(.LT, f(0.5)));
    const scale = key.layer.convert(.f32);
    const sigma = scale.add(fit.ds).scale(0.69314718056 / @as(f32, p.layers)).exp().scale(@as(f32, p.initial_sigma));
    const rows = m.join(&.{ key.x.convert(.f32).add(fit.dx), key.y.convert(.f32).add(fit.dy), scale, sigma, fit.response, valid.convert(.f32) });
    return valid.broad(rows.shape().withDtype(.bool)).select(rows, f(0));
}
