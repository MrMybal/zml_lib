const std = @import("std");
const Tensor = @import("zml_lib/runtime").layers.Tensor;

pub const joints = 17;
pub const height = 64;
pub const width = 48;
const radius = 5;

const gaussian = blk: {
    var coefficients: [radius + 1]f64 = undefined;
    var sum: f64 = 0;
    for (0..radius + 1) |i| {
        // OpenCV's automatic sigma for an 11-sample Gaussian is 2.
        coefficients[i] = @exp(-@as(f64, @floatFromInt(i * i)) / 8);
        sum += coefficients[i] * (if (i == 0) @as(f64, 1) else 2);
    }
    var result: [radius + 1]f32 = undefined;
    for (&result, coefficients) |*value, coefficient| value.* = @floatCast(coefficient / sum);
    break :blk result;
};

fn reflected(input: Tensor, index: Tensor, comptime axis: u3) Tensor {
    // REFLECT_101 mirrors interior samples, excluding the nearest edge sample.
    const end: i32 = @intCast(input.dim(axis) - 1);
    const positive = index.abs();
    const reflected_index = positive.minimum(positive.scale(@as(i32, -1)).addConstant(2 * end))
        .withTags(.{.sample});
    const grid = input.withTags(.{ .plane, .row, .column });
    const result = if (axis == 1) grid.gather(.{ .row = reflected_index }, .{}) else grid.gather(.{ .column = reflected_index }, .{});
    return result.reshape(input.shape());
}

fn blurAxis(input: Tensor, comptime axis: u3) Tensor {
    const index = Tensor.iota(.init(.{input.dim(axis)}, .i32), 0);
    var result = input.scale(gaussian[0]);
    for (1..radius + 1) |distance| {
        const delta: i32 = @intCast(distance);
        const left = reflected(input, index.subConstant(delta), axis);
        const right = reflected(input, index.addConstant(delta), axis);
        result = result.add(left.add(right).scale(gaussian[distance]));
    }
    return result;
}

fn sample(table: Tensor, plane: Tensor, x: Tensor, y: Tensor, off_x: i32, off_y: i32) Tensor {
    const zero = Tensor.scalar(@as(i32, 0), .i32);
    // The finite differences use edge replication, after Gaussian filtering.
    const col = x.addConstant(off_x).clamp(zero, .scalar(@as(i32, width - 1), .i32));
    const row = y.addConstant(off_y).clamp(zero, .scalar(@as(i32, height - 1), .i32));
    const address = plane.add(row.scale(@as(i32, width))).add(col).withTags(.{.joint});
    return table.gather(.{ .pixel = address }, .{}).reshape(x.shape().withDtype(.f32));
}

fn project(coordinate: Tensor, center: Tensor, size: Tensor, samples: i64) Tensor {
    const shape = coordinate.shape();
    const scale = size.divByConst(@as(f32, 200)).scale(@as(f32, 200));
    const pixel_scale = scale.divByConst(@as(f32, @floatFromInt(samples - 1))).broadcast(shape, &.{0});
    return coordinate.mul(pixel_scale).add(center.broadcast(shape, &.{0}))
        .sub(scale.scale(@as(f32, 0.5)).broadcast(shape, &.{0}));
}

pub fn decode(heatmaps: Tensor, boxes: Tensor) Tensor {
    const frames = heatmaps.dim(0);
    const planes = frames * joints;
    const grid = heatmaps.reshape(.{ planes, height, width });
    const peak = grid.reshape(.{ planes, height * width }).argMax(1);
    const score = peak.values.reshape(.{planes});
    const index = peak.indices.reshape(.{planes});
    const x = index.remainderConst(@as(i32, width));
    const y = index.divByConst(@as(i32, width));
    const plane = Tensor.iota(index.shape(), 0).scale(@as(i32, height * width));
    const log = blurAxis(blurAxis(grid, 2), 1).clamp(.scalar(@as(f32, 0.001), .f32), .scalar(@as(f32, 50), .f32)).log()
        .flatten().withTags(.{.pixel});
    const center = sample(log, plane, x, y, 0, 0);
    const left = sample(log, plane, x, y, -1, 0);
    const right = sample(log, plane, x, y, 1, 0);
    const up = sample(log, plane, x, y, 0, -1);
    const down = sample(log, plane, x, y, 0, 1);
    const lower_right = sample(log, plane, x, y, 1, 1);
    const upper_left = sample(log, plane, x, y, -1, -1);

    const dx = right.sub(left).scale(@as(f32, 0.5)).convert(.f64);
    const dy = down.sub(up).scale(@as(f32, 0.5)).convert(.f64);
    const eps: f64 = std.math.floatEps(f32);
    const xx = right.sub(center.scale(@as(f32, 2))).add(left).convert(.f64).addConstant(eps);
    const yy = down.sub(center.scale(@as(f32, 2))).add(up).convert(.f64).addConstant(eps);
    const xy = lower_right.sub(right).sub(down).add(center).add(center).sub(left).sub(up).add(upper_left)
        .scale(@as(f32, 0.5)).convert(.f64);
    const determinant = xx.mul(yy).sub(xy.mul(xy));
    const nonsingular = determinant.cmp(.NE, .scalar(@as(f64, 0), .f64));
    const divisor = nonsingular.select(determinant, .scalar(@as(f64, 1), .f64));
    const refine_x = nonsingular.select(yy.mul(dx).sub(xy.mul(dy)).div(divisor), .scalar(@as(f64, 0), .f64));
    const refine_y = nonsingular.select(xx.mul(dy).sub(xy.mul(dx)).div(divisor), .scalar(@as(f64, 0), .f64));
    const visible = score.cmp(.GT, .scalar(@as(f32, 0), .f32));
    const absent = Tensor.scalar(@as(f32, -1), .f32);
    const ux = visible.select(x.convert(.f64).sub(refine_x).convert(.f32), absent).reshape(.{ frames, joints });
    const uy = visible.select(y.convert(.f64).sub(refine_y).convert(.f32), absent).reshape(.{ frames, joints });
    const size = boxes.slice(1, .single(2));
    return Tensor.stack(&.{
        project(ux, boxes.slice(1, .single(0)), size.scale(@as(f32, 0.75)), width),
        project(uy, boxes.slice(1, .single(1)), size, height),
        score.reshape(.{ frames, joints }),
    }, 2, .field);
}
