const zml = @import("zml/core");
const Tensor = zml.Tensor;

pub const joints = 17;
const landmarks = joints * 3;
pub const box_start = landmarks;
pub const camera_start = box_start + 3;
const angular_start = camera_start + 3;
pub const input_width = angular_start + 6;
pub const output_width = landmarks + 3 + 6;

pub fn forward(values: Tensor, stats: Tensor) Tensor {
    const frames = values.dim(0);
    const keypoints = values.slice(1, .{ .end = landmarks }).reshape(.{ frames, joints, 3 });
    const box = values.slice(1, .{ .start = box_start, .end = camera_start });
    const center = box.slice(1, .{ .end = 2 });
    const size = box.slice(1, .{ .start = 2 });
    const xy = keypoints.slice(2, .{ .end = 2 });
    const origin = center.broadcast(xy.shape(), &.{ 0, 2 });
    const side = size.broadcast(xy.shape(), &.{ 0, 2 });
    const radius = side.scale(@as(f32, 0.5));
    const outside = xy.cmp(.LT, origin.sub(radius)).logical(.OR, xy.cmp(.GT, origin.add(radius)));
    const invisible = outside.slice(2, .{ .end = 1 }).logical(.OR, outside.slice(2, .{ .start = 1 }));
    const visible = invisible.cmp(.EQ, .scalar(false, .bool)).convert(.f32);
    const confidence = keypoints.slice(2, .{ .start = 2 }).mul(visible);
    const relative = xy.sub(origin).scale(@as(f32, 2)).div(side);
    const obs = Tensor.concatenate(&.{ relative, confidence }, 2).reshape(.{ frames, landmarks });

    const focal = values.slice(1, .{ .start = camera_start, .end = camera_start + 1 });
    const principal = values.slice(1, .{ .start = camera_start + 1, .end = angular_start });
    const offset = Tensor.concatenate(&.{ center.sub(principal), size }, 1);
    const cliffcam = offset.div(focal.broad(offset.shape()));

    const velocity = values.slice(1, .{ .start = angular_start });
    const mean = stats.slice(0, .{ .end = 6 }).broadcast(velocity.shape(), &.{1});
    const deviation = stats.slice(0, .{ .start = 6 }).broadcast(velocity.shape(), &.{1});
    return Tensor.concatenate(&.{ obs, cliffcam, velocity.sub(mean).div(deviation) }, 1);
}
