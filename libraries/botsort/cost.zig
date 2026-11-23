const zml = @import("zml/core");
const Tensor = zml.Tensor;

pub const track_fields = 4;
pub const detection_fields = 5;

fn f(value: f32) Tensor {
    return .scalar(value, .f32);
}

fn column(values: Tensor, index: i64) Tensor {
    return values.slice(1, .single(index));
}

fn area(boxes: Tensor) Tensor {
    return column(boxes, 2).sub(column(boxes, 0)).mul(column(boxes, 3).sub(column(boxes, 1))).optimizationBarrier();
}

pub fn forward(tracks: Tensor, detections: Tensor, fuse: Tensor) Tensor {
    const shape: zml.Shape = .init(.{ tracks.dim(0), detections.dim(0) }, .f32);
    const left = column(tracks, 0).broadcast(shape, &.{0}).maximum(column(detections, 0).broadcast(shape, &.{1}));
    const top = column(tracks, 1).broadcast(shape, &.{0}).maximum(column(detections, 1).broadcast(shape, &.{1}));
    const right = column(tracks, 2).broadcast(shape, &.{0}).minimum(column(detections, 2).broadcast(shape, &.{1}));
    const bottom = column(tracks, 3).broadcast(shape, &.{0}).minimum(column(detections, 3).broadcast(shape, &.{1}));
    const intersection = right.sub(left).maximum(f(0)).mul(bottom.sub(top).maximum(f(0))).optimizationBarrier();
    const union_area = area(detections).broadcast(shape, &.{1}).add(area(tracks).broadcast(shape, &.{0})).sub(intersection);
    const distance = f(1).sub(intersection.div(union_area.addConstant(@as(f32, 1e-7)))).optimizationBarrier();

    // NumPy's score fusion uses the rounded distance to recover similarity.
    const similarity = f(1).sub(distance);
    const weighted = similarity.mul(column(detections, 4).broadcast(shape, &.{1})).optimizationBarrier();
    const fuse_score = fuse.reshape(.{}).cmp(.NE, .scalar(0, .i32)).broad(shape.withDtype(.bool));
    return fuse_score.select(f(1).sub(weighted), distance);
}
