const zml = @import("zml/core");
const m = @import("post_math.zig");
const Tensor = zml.Tensor;
const scalar = m.scalar;

pub const pose_width = 12;
pub const output_width = 18;

fn gather(rows: Tensor, indices: Tensor) Tensor {
    return rows.withTags(.{ .key, .field }).gather(.{ .key = indices.withTags(.{.frame}) }, .{});
}

fn slerp(a: Tensor, b: Tensor, phase: Tensor) Tensor {
    const product = m.dot(a, b);
    const target = product.cmp(.LT, scalar(0)).broad(b.shape().withDtype(.bool)).select(b.negate(), b);
    const dot = product.abs().clamp(scalar(0), scalar(1));
    const near = dot.cmp(.GT, scalar(0.9995));
    const sine = scalar(1).sub(dot).mul(dot.addConstant(@as(f32, 1))).sqrt();
    const angle = m.atan2(sine, dot);
    const safe = sine.maximum(scalar(1e-10));
    const fraction = phase.mul(angle);
    const left = angle.sub(fraction).sin().div(safe).broad(a.shape());
    const right = fraction.sin().div(safe).broad(a.shape());
    const linear = a.add(target.sub(a).mul(phase.broad(a.shape())));
    return m.normalize(near.broad(a.shape().withDtype(.bool)).select(linear, a.mul(left).add(target.mul(right))), 1e-12);
}

pub fn forward(deltas: Tensor, plan: Tensor) Tensor {
    const count = deltas.dim(0);
    const frames = plan.dim(0);
    const row = Tensor.iota(.init(.{ 3, 3 }, .i32), 0);
    const eye = row.cmp(.EQ, Tensor.iota(row.shape(), 1)).convert(.f32).reshape(.{ 1, 3, 3 });
    var rotation = deltas.slice(1, .{ .end = 9 }).reshape(.{ count, 3, 3 });
    var translation = deltas.slice(1, .{ .start = 9 });
    // T_new = T_delta * T_previous in GVHMR's world-to-camera convention.
    var step: i64 = 1;
    while (step < count) : (step *= 2) {
        const earlier = Tensor.concatenate(&.{ eye.broad(eye.shape().set(0, step)), rotation.slice(0, .{ .end = count - step }) }, 0);
        translation = translation.add(m.apply(rotation, m.shift(translation, step, 0)));
        rotation = m.product(rotation, earlier);
    }

    const a = plan.slice(1, .single(0)).convert(.i32);
    const b = plan.slice(1, .single(1)).convert(.i32);
    const t = m.component(plan, 2);
    const q = m.normalize(m.matrixQuaternion(rotation), 1e-12);
    const interpolated = m.fromQuaternion(slerp(gather(q, a), gather(q, b), t));
    const sampled = gather(rotation.reshape(.{ count, 9 }), a).reshape(.{ frames, 3, 3 });
    const sampled_mask = a.cmp(.EQ, b).reshape(.{ frames, 1, 1 }).broad(sampled.shape().withDtype(.bool));
    const poses = sampled_mask.select(sampled, interpolated);
    const start = gather(translation, a);
    const end = gather(translation, b);
    const positions = start.add(end.sub(start).mul(t.broad(start.shape())));
    const relative = m.product(poses.slice(0, .{ .start = 1 }), poses.slice(0, .{ .end = frames - 1 }).transpose(.{ 0, 2, 1 }));
    const angular = relative.slice(1, .{ .end = 2 }).reshape(.{ frames - 1, 6 });
    const velocities = Tensor.concatenate(&.{ angular, angular.slice(0, .{ .start = frames - 2 }) }, 0);
    return m.join(&.{ poses.reshape(.{ frames, 9 }), positions, velocities });
}
