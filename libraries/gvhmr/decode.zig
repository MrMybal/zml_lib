const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const m = @import("post_math.zig");
const Tensor = zml.Tensor;

pub const frames_max = 4096;
const body_joints = 21;
const motion_width = 151;
const camera_offset = motion_width;
const crop_offset = camera_offset + 3;
const intrinsics_offset = crop_offset + 3;
const angular_offset = intrinsics_offset + 3;
pub const input_width = angular_offset + 6;
pub const output_width = body_joints * 3 + 10 + 6 * 3;

const schema = [_]rt.layers.Parameter{
    .{ .name = "mean", .shape = .init(.{motion_width}, .f32) },
    .{ .name = "std", .shape = .init(.{motion_width}, .f32) },
};

fn rotation6d(value: Tensor) Tensor {
    const first = m.normalize(value.slice(1, .{ .end = 3 }), 1e-12);
    const second = value.slice(1, .{ .start = 3 });
    const orthogonal = m.normalize(second.sub(first.mul(m.dot(first, second).broad(first.shape()))), 1e-12);
    return Tensor.stack(&.{ first, orthogonal, m.cross(first, orthogonal) }, 1, .row);
}

fn identity() Tensor {
    const row = Tensor.iota(.init(.{ 3, 3 }, .i32), 0);
    const col = Tensor.iota(row.shape(), 1);
    return row.cmp(.EQ, col).convert(.f32).reshape(.{ 1, 3, 3 });
}

fn nearIdentity(rotation: Tensor) Tensor {
    return m.norm(m.toAxisAngle(rotation)).cmp(.LT, m.scalar(1e-5));
}

fn snapIdentity(rotation: Tensor, unit: Tensor, small: Tensor) Tensor {
    const n = rotation.dim(0);
    return small.reshape(.{ n, 1, 1 }).broad(rotation.shape().withDtype(.bool)).select(unit.broad(rotation.shape()), rotation);
}

fn horizontal(x: Tensor) Tensor {
    const zero = m.scalar(0).broad(.init(.{ x.dim(0), 1 }, .f32));
    return m.normalize(m.join(&.{ m.component(x, 0), zero, m.component(x, 2) }), 1e-12);
}

fn cameraToInitial(camera: Tensor, gravity: Tensor, angular: Tensor, unit: Tensor) Tensor {
    const frames = angular.dim(0);
    const raw_delta = rotation6d(angular);
    const small = nearIdentity(raw_delta);
    const delta = snapIdentity(raw_delta, unit, small);
    const view = m.product(gravity, camera.transpose(.{ 0, 2, 1 }));
    const direction = horizontal(m.column(view, 2));
    const next = horizontal(m.column(m.product(view, delta.transpose(.{ 0, 2, 1 })), 2));
    const cosine = m.dot(direction, next).clamp(m.scalar(-1), m.scalar(1));
    // Use the cross product for the sine so small camera rotations do not
    // lose precision when the normalized dot product rounds near one.
    const cross = m.cross(next, direction);
    const angle = m.atan2(m.norm(cross), cosine);
    const axis = m.normalize(cross, 1e-12);
    const yaw_step = axis.mul(angle.broad(axis.shape()));
    const yaw = m.smooth(small.broad(axis.shape().withDtype(.bool)).select(m.scalar(0), yaw_step), 3);
    const relative = m.fromAxisAngle(yaw).transpose(.{ 0, 2, 1 });

    // Compose the camera increments in chronological order, anchored at frame zero.
    var prefix = Tensor.concatenate(&.{ unit, relative.slice(0, .{ .start = 1 }) }, 0);
    var step: i64 = 1;
    while (step < frames) : (step *= 2) {
        const earlier = Tensor.concatenate(&.{ unit.broad(unit.shape().set(0, step)), prefix.slice(0, .{ .end = frames - step }) }, 0);
        prefix = m.product(earlier, prefix);
    }
    return snapIdentity(prefix, unit, nearIdentity(prefix));
}

fn cameraTranslation(pred: Tensor, bbox: Tensor, intrinsics: Tensor) Tensor {
    const scale = m.component(pred, 0).mul(m.component(bbox, 2)).addConstant(@as(f32, 1e-9));
    const dx = m.component(bbox, 0).sub(m.component(intrinsics, 1)).scale(@as(f32, 2)).div(scale);
    const dy = m.component(bbox, 1).sub(m.component(intrinsics, 2)).scale(@as(f32, 2)).div(scale);
    const depth = m.component(intrinsics, 0).scale(@as(f32, 2)).div(scale);
    return m.join(&.{ m.component(pred, 1).add(dx), m.component(pred, 2).add(dy), depth });
}

fn gravityToWorld(x: Tensor) Tensor {
    // Apply the reference half turn around Z, including the FP32 sine of pi.
    const sine = @sin(@as(f32, std.math.pi));
    const a = x.slice(1, .{ .end = 1 });
    const b = x.slice(1, .{ .start = 1, .end = 2 });
    return Tensor.concatenate(&.{ a.negate().sub(b.scale(sine)), b.negate().add(a.scale(sine)), x.slice(1, .{ .start = 2 }) }, 1);
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const frames = input.dim(0);
    const normalized = input.slice(1, .{ .end = motion_width });
    const motion = normalized.mul(parameters[1].broadcast(normalized.shape(), &.{1}))
        .add(parameters[0].broadcast(normalized.shape(), &.{1}));
    const pose = motion.slice(1, .{ .end = 126 }).reshape(.{ frames * body_joints, 6 });
    const betas = motion.slice(1, .{ .start = 126, .end = 136 });
    const camera_root = m.toAxisAngle(rotation6d(motion.slice(1, .{ .start = 136, .end = 142 })));
    const gravity_root = m.toAxisAngle(rotation6d(motion.slice(1, .{ .start = 142, .end = 148 })));
    const velocity = motion.slice(1, .{ .start = 148 });
    const angular = input.slice(1, .{ .start = angular_offset });
    const grav_rotation = m.fromAxisAngle(gravity_root);
    const scene_rotation = if (frames == 1) grav_rotation else m.product(cameraToInitial(m.fromAxisAngle(camera_root), grav_rotation, angular, identity()), grav_rotation);
    const aligned = m.fromAxisAngle(m.toAxisAngle(scene_rotation));
    const displacements = m.apply(aligned, velocity);
    const translation = if (frames == 1) m.scalar(0).broad(velocity.shape()) else m.prefixSum(m.shift(displacements, 1, 0));
    const world_root = m.toAxisAngle(gravityToWorld(aligned));
    const world_translation = gravityToWorld(translation);
    const incam_translation = cameraTranslation(
        input.slice(1, .{ .start = camera_offset, .end = crop_offset }),
        input.slice(1, .{ .start = crop_offset, .end = intrinsics_offset }),
        input.slice(1, .{ .start = intrinsics_offset, .end = angular_offset }),
    );
    return m.join(&.{
        m.toAxisAngle(rotation6d(pose)).reshape(.{ frames, body_joints * 3 }), betas,
        camera_root,                                                           gravity_root,
        velocity,                                                              incam_translation,
        world_root,                                                            world_translation,
    });
}

pub const Model = struct {
    graph: rt.TensorModel,

    pub fn load(session: *const rt.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn predict(self: *Model, frames: i64, values: []const f32, output: []f32) !u32 {
        if (frames < 1 or frames > frames_max) return error.InvalidFrameCount;
        return self.graph.infer(forward, .init(.{ frames, input_width }, .f32), .init(.{ frames, output_width }, .f32), values, output);
    }
};
