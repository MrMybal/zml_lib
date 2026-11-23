const std = @import("std");
const rt = @import("zml_lib/runtime");
const scale = @import("pose_scale.zig");
const Tensor = rt.layers.Tensor;
const Shape = @import("zml/core").Shape;
const crop_height = rt.pose_vit.height;
const crop_width = rt.pose_vit.width;
const square_span: f64 = crop_height - 1;
const first_column = (crop_height - crop_width) / 2;

fn int(value: i32) Tensor {
    return .scalar(value, .i32);
}

fn double(value: f64) Tensor {
    return .scalar(value, .f64);
}

// Preserve the separate rounding of the FP64 products that feed additions in
// the reference affine solve and grid construction.
fn roundedProduct(left: Tensor, right: Tensor) Tensor {
    return left.mul(right).optimizationBarrier();
}

// Solve the affine correspondence with partial pivoting. The two right-hand
// sides share the same three source points, all rounded to FP32 before solving.
fn inverseAffine(boxes: Tensor) [2][3]Tensor {
    const cx = boxes.slice(1, .single(0)).scale(@as(f32, 0.5));
    const cy = boxes.slice(1, .single(1)).scale(@as(f32, 0.5));
    const radius = boxes.slice(1, .single(2)).scale(@as(f32, 0.25));
    const x0 = cx.sub(radius).convert(.f64);
    const x1 = cx.add(radius).convert(.f64);
    const y0 = cy.sub(radius).convert(.f64);
    const zero = double(0).broad(x0.shape());
    const one = double(1).broad(x0.shape());
    const half = double(square_span / 2.0).broad(x0.shape());
    var rows = [3][5]Tensor{
        .{ x0, y0, one, zero, zero },
        .{ x1, y0, one, double(square_span).broad(x0.shape()), zero },
        .{ cx.convert(.f64), cy.convert(.f64), one, half, half },
    };
    var valid = one.cmp(.GT, zero);
    for (0..3) |col| {
        var pivot = rows[col];
        var selected = int(@intCast(col)).broad(x0.shape().withDtype(.i32));
        for (col + 1..3) |i| {
            const swap = rows[i][col].abs().cmp(.GT, pivot[col].abs());
            selected = swap.select(int(@intCast(i)), selected);
            for (col..5) |j| pivot[j] = swap.select(rows[i][j], pivot[j]);
        }
        for (col + 1..3) |i| {
            const swap = selected.cmp(.EQ, int(@intCast(i)));
            for (col..5) |j| rows[i][j] = swap.select(rows[col][j], rows[i][j]);
        }
        rows[col] = pivot;
        const nonsingular = pivot[col].abs().cmp(.GE, double(100 * std.math.floatEps(f64)));
        valid = valid.logical(.AND, nonsingular);
        rows[col][col] = nonsingular.select(pivot[col], one);
        const reciprocal = double(-1).div(rows[col][col]);
        for (col + 1..3) |i| {
            const factor = rows[i][col].mul(reciprocal);
            for (col + 1..5) |j| rows[i][j] = rows[i][j].add(roundedProduct(factor, pivot[j]));
        }
    }

    var solution: [2][3]Tensor = undefined;
    for (&solution, 3..) |*coefficients, rhs| {
        for (0..3) |i| {
            const row = 2 - i;
            var value = rows[row][rhs];
            for (row + 1..3) |j| value = value.sub(roundedProduct(rows[row][j], coefficients[j]));
            coefficients[row] = value.div(rows[row][row]);
        }
    }
    const a, const b = solution;
    const determinant = roundedProduct(a[0], b[1]).sub(roundedProduct(a[1], b[0]));
    valid = valid.logical(.AND, determinant.cmp(.NE, zero));
    const reciprocal = valid.select(one.div(valid.select(determinant, one)), zero);
    const u = b[1].mul(reciprocal);
    const v = a[1].scale(@as(f64, -1)).mul(reciprocal);
    const s = b[0].scale(@as(f64, -1)).mul(reciprocal);
    const t = a[0].mul(reciprocal);
    return .{
        .{ u, v, roundedProduct(u.negate(), a[2]).sub(roundedProduct(v, b[2])) },
        .{ s, t, roundedProduct(s.negate(), a[2]).sub(roundedProduct(t, b[2])) },
    };
}

fn reflected(image: Tensor, offset: i32, comptime axis: u3) Tensor {
    const length = image.dim(axis);
    if (length == 1) return image;
    const period: i32 = @intCast(2 * (length - 1));
    const position = Tensor.iota(.init(.{length}, .i32), 0).addConstant(offset).abs().remainderConst(period);
    const index = position.minimum(int(period).sub(position)).withTags(.{.sample});
    const labels = image.withTags(.{ .frame, .row, .column, .channel });
    const result = if (axis == 1) labels.gather(.{ .row = index }, .{}) else labels.gather(.{ .column = index }, .{});
    return result.reshape(image.shape());
}

fn filter(image: Tensor, weights: [3]Tensor, comptime axis: u3) Tensor {
    const shape = image.shape();
    return image.mul(weights[0].broadcast(shape, &.{0}))
        .add(reflected(image, -1, axis).add(reflected(image, 1, axis)).mul(weights[1].broadcast(shape, &.{0})))
        .add(reflected(image, -2, axis).add(reflected(image, 2, axis)).mul(weights[2].broadcast(shape, &.{0})));
}

fn antialias(image: Tensor, boxes: Tensor) Tensor {
    const factor = boxes.slice(1, .single(2)).divByConst(@as(f32, 1024));
    const active = factor.cmp(.GT, .scalar(@as(f32, 1.1), .f32));
    const sigma = factor.subConstant(@as(f32, 1)).scale(@as(f32, 0.5)).convert(.f64).maximum(double(0.05));
    const variance = sigma.mul(sigma);
    const outer = double(-2).div(variance).exp();
    const inner = double(-0.5).div(variance).exp();
    const gain = double(256).div(outer.add(inner).scale(@as(f64, 2)).addConstant(@as(f64, 1)));
    // Quantize the two symmetric taps with error diffusion, and force the
    // integer kernel sum to 256. Round only after the second filter pass.
    const exact_outer = outer.mul(gain);
    const q_outer = exact_outer.round();
    const q_inner = inner.mul(gain).add(exact_outer.sub(q_outer)).round();
    const outer_i = active.select(q_outer.convert(.i32), int(0));
    const inner_i = active.select(q_inner.convert(.i32), int(0));
    const weights = [3]Tensor{ int(256).sub(outer_i.add(inner_i).scale(@as(i32, 2))), inner_i, outer_i };
    return filter(filter(image, weights, 2), weights, 1)
        .addConstant(@as(i32, 32768)).shiftRightArithmetic(int(16));
}

fn coordinate(coefficients: [3]Tensor, shape: Shape, row: Tensor, column: Tensor) Tensor {
    const a, const b, const c = coefficients;
    const x = a.broadcast(shape, &.{0}).mul(column).scale(@as(f64, 1024)).round().convert(.i32);
    const y = roundedProduct(b.broadcast(shape, &.{0}), row).add(c.broadcast(shape, &.{0}))
        .scale(@as(f64, 1024)).round().convert(.i32);
    return x.add(y).addConstant(@as(i32, 16)).shiftRightArithmetic(int(5));
}

fn fetch(image: Tensor, frame_offset: Tensor, x: Tensor, y: Tensor) Tensor {
    const height: i32 = @intCast(image.dim(1));
    const width: i32 = @intCast(image.dim(2));
    const inside = x.cmp(.GE, int(0)).logical(.AND, x.cmp(.LT, int(width)))
        .logical(.AND, y.cmp(.GE, int(0))).logical(.AND, y.cmp(.LT, int(height)));
    const sx = x.clamp(int(0), int(width - 1));
    const sy = y.clamp(int(0), int(height - 1));
    const index = frame_offset.add(sy.scale(width)).add(sx).withTags(.{ .frame, .row, .column });
    const pixels = image.reshape(.{ image.dim(0) * height * width, 3 }).withTags(.{ .pixel, .channel });
    const values = pixels.gather(.{ .pixel = index }, .{}).reshape(.{ x.dim(0), crop_height, crop_width, 3 });
    return inside.broadcast(values.shape().withDtype(.bool), &.{ 0, 1, 2 }).select(values, int(0));
}

fn process(rgb: Tensor, boxes: Tensor, comptime blur: bool) Tensor {
    const image = if (blur) antialias(rgb.convert(.i32), boxes) else rgb.convert(.i32);
    const matrix = inverseAffine(boxes);
    const shape: Shape = .init(.{ image.dim(0), crop_height, crop_width }, .f64);
    const row = Tensor.iota(shape.withDtype(.i32), 1).convert(.f64);
    const col = Tensor.iota(shape.withDtype(.i32), 2).addConstant(@as(i32, first_column)).convert(.f64);
    const fx = coordinate(matrix[0], shape, row, col);
    const fy = coordinate(matrix[1], shape, row, col);
    const x = fx.shiftRightArithmetic(int(5));
    const y = fy.shiftRightArithmetic(int(5));
    const offset = Tensor.iota(x.shape(), 0).scale(@as(i32, @intCast(image.dim(1) * image.dim(2))));
    const x1 = x.addConstant(@as(i32, 1));
    const y1 = y.addConstant(@as(i32, 1));
    const sample_shape: Shape = .init(.{ image.dim(0), crop_height, crop_width, 3 }, .i32);
    const tx = fx.sub(x.shiftLeft(int(5))).broadcast(sample_shape, &.{ 0, 1, 2 });
    const ty = fy.sub(y.shiftLeft(int(5))).broadcast(sample_shape, &.{ 0, 1, 2 });
    const rx = int(32).sub(tx);
    const ry = int(32).sub(ty);
    const accum = fetch(image, offset, x, y).mul(rx.mul(ry))
        .add(fetch(image, offset, x1, y).mul(tx.mul(ry)))
        .add(fetch(image, offset, x, y1).mul(rx.mul(ty)))
        .add(fetch(image, offset, x1, y1).mul(tx.mul(ty)));
    const samples = accum.addConstant(@as(i32, 512)).shiftRightArithmetic(int(10))
        .convert(.f32).divByConst(@as(f32, 255));
    const mean = Tensor.stack(&.{ .scalar(@as(f32, 0.485), .f32), .scalar(@as(f32, 0.456), .f32), .scalar(@as(f32, 0.406), .f32) }, 0, .channel);
    const deviation = Tensor.stack(&.{ .scalar(@as(f32, 0.229), .f32), .scalar(@as(f32, 0.224), .f32), .scalar(@as(f32, 0.225), .f32) }, 0, .channel);
    return samples.sub(mean.broadcast(samples.shape(), &.{3})).div(deviation.broadcast(samples.shape(), &.{3})).transpose(.{ 0, 3, 1, 2 });
}

pub const Mode = enum { crop, half, half_check };

pub fn Pipeline(comptime mode: Mode, comptime blur: bool) type {
    return struct {
        const Output = if (mode == .half_check) struct { Tensor, Tensor } else Tensor;

        pub fn forward(rgb: Tensor, boxes: Tensor) Output {
            const images = if (mode == .crop) rgb else scale.forward(rgb);
            const crops = process(images, boxes, blur);
            return if (mode == .half_check) .{ crops, images } else crops;
        }
    };
}
