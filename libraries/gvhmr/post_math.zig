const std = @import("std");
const zml = @import("zml/core");
const stablehlo = @import("mlir/dialects/stablehlo");
const Tensor = zml.Tensor;

pub fn scalar(value: f32) Tensor {
    return .scalar(value, .f32);
}

pub fn component(x: Tensor, i: i64) Tensor {
    return x.slice(1, .{ .start = i, .end = i + 1 });
}

pub fn join(xs: []const Tensor) Tensor {
    return Tensor.concatenate(xs, 1);
}

pub fn column(m: Tensor, i: i64) Tensor {
    return m.slice(2, .single(i));
}

pub fn product(a: Tensor, b: Tensor) Tensor {
    return a.dotGeneral(b, &.{.{ 2, 1 }}, &.{.{ 0, 0 }});
}

pub fn apply(m: Tensor, x: Tensor) Tensor {
    return m.dotGeneral(x, &.{.{ 2, 1 }}, &.{.{ 0, 0 }});
}

pub fn dot(a: Tensor, b: Tensor) Tensor {
    return a.mul(b).sum(1);
}

pub fn norm(x: Tensor) Tensor {
    return dot(x, x).sqrt();
}

pub fn normalize(x: Tensor, floor: f32) Tensor {
    return x.div(norm(x).maximum(scalar(floor)).broad(x.shape()));
}

pub fn cross(a: Tensor, b: Tensor) Tensor {
    const ax = component(a, 0);
    const ay = component(a, 1);
    const az = component(a, 2);
    const bx = component(b, 0);
    const by = component(b, 1);
    const bz = component(b, 2);
    return join(&.{ ay.mul(bz).sub(az.mul(by)), az.mul(bx).sub(ax.mul(bz)), ax.mul(by).sub(ay.mul(bx)) });
}

pub fn atan2(y: Tensor, x: Tensor) Tensor {
    const ctx = zml.Compiler.current();
    const op = stablehlo.atan2(ctx.mlir_ctx, y.value(), x.value(), ctx.location).appendTo(ctx.currentScope().block);
    return .fromMlirValue(op.result(0));
}

fn halfSineOverAngle(angle: Tensor, half: Tensor) Tensor {
    const small = angle.abs().cmp(.LT, scalar(1e-6));
    const taylor = angle.mul(angle).scale(@as(f32, -1.0 / 48.0)).addConstant(@as(f32, 0.5));
    return small.select(taylor, half.sin().div(small.select(scalar(1), angle)));
}

pub fn fromAxisAngle(axis_angle: Tensor) Tensor {
    const angle = norm(axis_angle);
    const half = angle.scale(@as(f32, 0.5));
    const xyz = axis_angle.mul(halfSineOverAngle(angle, half).broad(axis_angle.shape()));
    return fromQuaternion(join(&.{ half.cos(), xyz }));
}

pub fn fromQuaternion(q: Tensor) Tensor {
    return quaternionMatrix(q, scalar(2).div(dot(q, q)));
}

pub fn toAxisAngle(m: Tensor) Tensor {
    const q = matrixQuaternion(m);
    const xyz = q.slice(1, .{ .start = 1 });
    const half = atan2(norm(xyz), component(q, 0));
    return xyz.div(halfSineOverAngle(half.scale(@as(f32, 2)), half).broad(xyz.shape()));
}

fn quaternionMatrix(q: Tensor, s: Tensor) Tensor {
    const w = component(q, 0);
    const x = component(q, 1);
    const y = component(q, 2);
    const z = component(q, 3);
    const xx = s.mul(x).mul(x);
    const yy = s.mul(y).mul(y);
    const zz = s.mul(z).mul(z);
    const xy = s.mul(x).mul(y);
    const xz = s.mul(x).mul(z);
    const yz = s.mul(y).mul(z);
    const wx = s.mul(w).mul(x);
    const wy = s.mul(w).mul(y);
    const wz = s.mul(w).mul(z);
    return join(&.{
        yy.add(zz).negate().addConstant(@as(f32, 1)), xy.sub(wz),                                   xz.add(wy),
        xy.add(wz),                                   xx.add(zz).negate().addConstant(@as(f32, 1)), yz.sub(wx),
        xz.sub(wy),                                   yz.add(wx),                                   xx.add(yy).negate().addConstant(@as(f32, 1)),
    }).reshape(.{ q.dim(0), 3, 3 });
}

pub fn matrixQuaternion(m: Tensor) Tensor {
    const c0 = column(m, 0);
    const c1 = column(m, 1);
    const c2 = column(m, 2);
    const a = component(c0, 0);
    const b = component(c1, 1);
    const c = component(c2, 2);
    const d = component(c1, 2).sub(component(c2, 1));
    const e = component(c2, 0).sub(component(c0, 2));
    const f = component(c0, 1).sub(component(c1, 0));
    const g = component(c0, 1).add(component(c1, 0));
    const h = component(c0, 2).add(component(c2, 0));
    const i = component(c1, 2).add(component(c2, 1));
    const score = [_]Tensor{
        a.add(b).add(c).addConstant(@as(f32, 1)), a.sub(b).sub(c).addConstant(@as(f32, 1)),
        b.sub(a).sub(c).addConstant(@as(f32, 1)), c.sub(a).sub(b).addConstant(@as(f32, 1)),
    };
    const candidates = [_]Tensor{
        join(&.{ score[0], d, e, f }), join(&.{ d, score[1], g, h }),
        join(&.{ e, g, score[2], i }), join(&.{ f, h, i, score[3] }),
    };
    var best = score[0];
    var q = candidates[0];
    for (score[1..], candidates[1..]) |s, option| {
        q = s.cmp(.GT, best).broad(q.shape().withDtype(.bool)).select(option, q);
        best = best.maximum(s);
    }
    q = q.div(best.maximum(scalar(0.01)).sqrt().scale(@as(f32, 2)).broad(q.shape()));
    return component(q, 0).cmp(.LT, scalar(0)).broad(q.shape().withDtype(.bool)).select(q.negate(), q);
}

fn quaternionProduct(a: Tensor, b: Tensor) Tensor {
    const u = a.slice(1, .{ .start = 1 });
    const v = b.slice(1, .{ .start = 1 });
    const w = component(a, 0);
    const z = component(b, 0);
    const xyz = v.mul(w.broad(v.shape())).add(u.mul(z.broad(u.shape()))).add(cross(u, v));
    return join(&.{ w.mul(z).sub(dot(u, v)), xyz });
}

pub fn turnTowards(rotation: Tensor, before: Tensor, after: Tensor, weight: f32) Tensor {
    const aim = cross(before, after);
    const w = dot(before, before).mul(dot(after, after)).sqrt().add(dot(before, after));
    const opposite = norm(aim).cmp(.EQ, scalar(0)).logical(.AND, w.abs().cmp(.LE, scalar(1e-4)));
    const y_axis = Tensor.constantTensor(.init(.{ 1, 3 }, .f32), @as(*const [12]u8, @ptrCast(&[_]f32{ 0, 1, 0 })));
    const axis = opposite.broad(aim.shape().withDtype(.bool)).select(y_axis.broad(aim.shape()), aim);
    const delta = normalize(join(&.{ w, axis }), 1e-8);
    const base = normalize(matrixQuaternion(rotation), 1e-8);
    const target = normalize(quaternionProduct(delta, base), 1e-8);
    const inverse_base = join(&.{ component(base, 0), base.slice(1, .{ .start = 1 }).negate() });
    const relative = normalize(quaternionProduct(target, inverse_base), 1e-8);
    const real = component(relative, 0).clamp(scalar(-1), scalar(1));
    const xyz = relative.slice(1, .{ .start = 1 });
    const sine = real.mul(real).negate().addConstant(@as(f32, 1)).maximum(scalar(0)).sqrt();
    const angle = atan2(sine, real).maximum(scalar(1e-9));
    const weighted = angle.scale(weight);
    const gain = weighted.sin().div(angle.sin());
    const correction = join(&.{ weighted.cos(), xyz.mul(gain.broad(xyz.shape())) });
    return quaternionMatrix(quaternionProduct(correction, base), scalar(2));
}

pub fn inverseNormalized(m: Tensor) Tensor {
    const a = column(m, 0);
    const b = column(m, 1);
    const c = column(m, 2);
    const cofactor = cross(b, c);
    const adjugate = Tensor.stack(&.{ cofactor, cross(c, a), cross(a, b) }, 1, .row);
    const det = dot(a, cofactor);
    const inverse = adjugate.div(det.broadcast(adjugate.shape(), &.{ 0, 2 }));
    return normalizeColumns(inverse);
}

pub fn normalizeColumns(m: Tensor) Tensor {
    return m.div(m.mul(m).sum(1).sqrt().addConstant(@as(f32, 1e-9)).broad(m.shape()));
}

pub fn prefixSum(x: Tensor) Tensor {
    var out = x;
    var step: i64 = 1;
    while (step < x.dim(0)) : (step *= 2) out = out.add(shift(out, step, 0));
    return out;
}

pub fn shift(x: Tensor, distance: i64, fill: f32) Tensor {
    const head = scalar(fill).broad(x.shape().set(0, distance));
    return Tensor.concatenate(&.{ head, x.slice(0, .{ .end = x.dim(0) - distance }) }, 0);
}

fn gaussian(comptime sigma: comptime_int) [8 * sigma + 1]f32 {
    const radius = 4 * sigma;
    var exact: [2 * radius + 1]f64 = undefined;
    var total: f64 = 0;
    for (&exact, 0..) |*p, i| {
        const f: f64 = @floatFromInt(i);
        const delta = f - radius;
        p.* = @exp(-(delta * delta) / (2 * sigma * sigma));
        total += p.*;
    }
    var out: [exact.len]f32 = undefined;
    for (&out, exact) |*p, coefficient| p.* = @floatCast(coefficient / total);
    return out;
}

pub fn smooth(x: Tensor, comptime sigma: comptime_int) Tensor {
    const radius = 4 * sigma;
    const weights = comptime gaussian(sigma);
    const n = x.dim(0);
    const width = x.dim(1);
    const left = x.slice(0, .{ .end = 1 }).broad(.init(.{ radius, width }, .f32));
    const right = x.slice(0, .{ .start = n - 1 }).broad(left.shape());
    const padded = Tensor.concatenate(&.{ left, x, right }, 0).transpose(.{ 1, 0 }).reshape(.{ width, 1, n + 2 * radius });
    const kernel = Tensor.constantTensor(.init(.{ 1, 1, weights.len }, .f32), std.mem.asBytes(&weights));
    return padded.conv1d(kernel, .{}).reshape(.{ width, n }).transpose(.{ 1, 0 });
}
