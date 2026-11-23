const Tensor = @import("layers.zig").Tensor;

pub fn reflectEnd(input: Tensor, comptime axis: u3, amount: i64) Tensor {
    if (amount == 0) return input;
    const end = input.dim(axis) - 1;
    if (end == 0) return input.broadcast(input.shape().setDim(axis, amount + 1), &.{ 0, 1, 2, 3 });
    return Tensor.concatenate(&.{ input, input.slice(axis, .{ .start = end - amount, .end = end }).reverse(.{axis}) }, axis);
}

fn sample(input: Tensor, comptime axis: u3, indices: Tensor, weight: Tensor) Tensor {
    const valid = indices.maximum(.scalar(@as(i32, 0), .i32))
        .minimum(.scalar(@as(i32, @intCast(input.dim(axis) - 1)), .i32));
    const tagged = input.withTags(.{ .batch, .channel, .row, .col });
    const result = if (axis == 2)
        tagged.gather(.{ .row = valid }, .{})
    else
        tagged.gather(.{ .col = valid }, .{});
    const aligned = result.reshape(input.shape().setDim(axis, indices.dim(0)));
    return aligned.mul(weight.broadcast(aligned.shape(), &.{axis}));
}

fn resizeLinear(input: Tensor, comptime axis: u3, size: i64) Tensor {
    if (input.dim(axis) == size) return input;
    const scale: f32 = if (size == 1) 0 else @as(f32, @floatFromInt(input.dim(axis) - 1)) / @as(f32, @floatFromInt(size - 1));
    const coordinate = Tensor.iota(.init(.{size}, .i32), 0).convert(.f32).scale(scale);
    const base = coordinate.floor();
    const fraction = coordinate.sub(base);
    const index = base.convert(.i32);
    return sample(input, axis, index, fraction.scale(@as(f32, -1)).addConstant(@as(f32, 1)))
        .add(sample(input, axis, index.addConstant(@as(i32, 1)), fraction));
}

pub fn bilinearAlignCorners(input: Tensor, height: i64, width: i64) Tensor {
    return resizeLinear(resizeLinear(input, 2, height), 3, width);
}

const cubic_coefficient: f32 = -0.75;

fn innerCubic(x: Tensor) Tensor {
    return x.scale(cubic_coefficient + 2).subConstant(cubic_coefficient + 3)
        .mul(x).mul(x).addConstant(@as(f32, 1));
}

fn outerCubic(x: Tensor) Tensor {
    return x.scale(cubic_coefficient).subConstant(5 * cubic_coefficient)
        .mul(x).addConstant(8 * cubic_coefficient).mul(x).subConstant(4 * cubic_coefficient);
}

fn resizeCubic(input: Tensor, comptime axis: u3, size: i64, offset: f64) Tensor {
    const scale: f32 = @floatCast(@as(f64, @floatFromInt(input.dim(axis))) / (@as(f64, @floatFromInt(size)) + offset));
    const coordinate = Tensor.iota(.init(.{size}, .i32), 0).convert(.f32)
        .addConstant(@as(f32, 0.5)).scale(scale).subConstant(@as(f32, 0.5));
    const base = coordinate.floor();
    const x = coordinate.sub(base);
    const index = base.convert(.i32);
    const inverse = x.scale(@as(f32, -1)).addConstant(@as(f32, 1));
    return sample(input, axis, index.subConstant(@as(i32, 1)), outerCubic(x.addConstant(@as(f32, 1))))
        .add(sample(input, axis, index, innerCubic(x)))
        .add(sample(input, axis, index.addConstant(@as(i32, 1)), innerCubic(inverse)))
        .add(sample(input, axis, index.addConstant(@as(i32, 2)), outerCubic(inverse.addConstant(@as(f32, 1)))));
}

pub fn bicubicHalfPixel(input: Tensor, height: i64, width: i64, offset: f64) Tensor {
    return resizeCubic(resizeCubic(input, 3, width, offset), 2, height, offset);
}

fn antialiasLast(input: Tensor, size: i64) Tensor {
    const source = input.dim(3);
    if (source == size) return input;
    const scale = @as(f32, @floatFromInt(source)) / @as(f32, @floatFromInt(size));
    const Shape = @TypeOf(input.shape());
    const shape = Shape.init(.{ size, source }, .f32);
    const target = Tensor.iota(shape.withDtype(.i32), 0).convert(.f32).addConstant(@as(f32, 0.5)).scale(scale);
    const src = Tensor.iota(shape.withDtype(.i32), 1).convert(.f32).addConstant(@as(f32, 0.5));
    const x = src.sub(target).abs().scale(1 / @max(1, scale));
    const inner = x.scale(@as(f32, 1.5)).subConstant(@as(f32, 2.5)).mul(x).mul(x).addConstant(@as(f32, 1));
    const outer = x.scale(@as(f32, -0.5)).addConstant(@as(f32, 2.5)).mul(x).subConstant(@as(f32, 4)).mul(x).addConstant(@as(f32, 2));
    const zero = Tensor.scalar(@as(f32, 0), .f32);
    const weight = Tensor.select(x.cmp(.LT, .scalar(@as(f32, 1), .f32)), inner, Tensor.select(x.cmp(.LT, .scalar(@as(f32, 2), .f32)), outer, zero.broad(shape)));
    return input.dotGeneral(weight.div(weight.sum(1).broad(shape)), &.{.{ 3, 1 }}, &.{});
}

pub fn bicubicAntialias(input: Tensor, height: i64, width: i64) Tensor {
    const horizontal = antialiasLast(input, width);
    return antialiasLast(horizontal.transpose(.{ 0, 1, 3, 2 }), height).transpose(.{ 0, 1, 3, 2 });
}
