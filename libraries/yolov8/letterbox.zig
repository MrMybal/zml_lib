const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;

const limit = 640;
const stride = 32;
const unity = 1 << 11;

pub const Layout = struct {
    height: i64,
    width: i64,
    canvas_height: i64,
    canvas_width: i64,
};

fn nearestEven(value: f64) i64 {
    const lower: i64 = @intFromFloat(@floor(value));
    const remainder = value - @as(f64, @floatFromInt(lower));
    return lower + @intFromBool(remainder > 0.5 or (remainder == 0.5 and @mod(lower, 2) != 0));
}

pub const SourceTransform = struct {
    gain: f32,
    pad_x: f32,
    pad_y: f32,
    width: f32,
    height: f32,
};

pub fn sourceTransform(canvas: Layout, height: i64, width: i64) SourceTransform {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const canvas_h: f64 = @floatFromInt(canvas.canvas_height);
    const canvas_w: f64 = @floatFromInt(canvas.canvas_width);
    // Match scale_boxes with no ratio_pad. It derives the gain and margins
    // from the two shapes, using the unrounded scaled extent for the margin.
    const gain = @min(canvas_h / h, canvas_w / w);
    return .{
        .gain = @floatCast(gain),
        .pad_x = @floatFromInt(nearestEven((canvas_w - w * gain) * 0.5 - 0.1)),
        .pad_y = @floatFromInt(nearestEven((canvas_h - h * gain) * 0.5 - 0.1)),
        .width = @floatCast(w),
        .height = @floatCast(h),
    };
}

pub fn layout(height: i64, width: i64) !Layout {
    if (height <= 0 or width <= 0 or height > 8192 or width > 8192) return error.InvalidLetterboxDimensions;
    const scale = limit / @as(f64, @floatFromInt(@max(height, width)));
    const h = @max(1, nearestEven(@as(f64, @floatFromInt(height)) * scale));
    const w = @max(1, nearestEven(@as(f64, @floatFromInt(width)) * scale));
    return .{
        .height = h,
        .width = w,
        .canvas_height = @divFloor(h + stride - 1, stride) * stride,
        .canvas_width = @divFloor(w + stride - 1, stride) * stride,
    };
}

fn integer(value: i32) Tensor {
    return .scalar(value, .i32);
}

fn sumPairs(input: Tensor, comptime axis: u3) Tensor {
    const size = input.dim(axis);
    const first = input.slice(axis, .{ .start = 0, .end = size - 1, .step = 2 });
    const second = input.slice(axis, .{ .start = 1, .end = size, .step = 2 });
    return first.add(second);
}

fn sample(input: Tensor, comptime axis: u3, indices: Tensor, weights: Tensor) Tensor {
    const tagged = input.withTags(.{ .image, .row, .column, .channel });
    const index = indices.withTags(.{.sample});
    const values = if (axis == 1) tagged.gather(.{ .row = index }, .{}) else tagged.gather(.{ .column = index }, .{});
    const shape = input.shape().setDim(axis, indices.dim(0));
    return values.reshape(shape).mul(weights.broadcast(shape, &.{axis}));
}

fn linearAxis(input: Tensor, comptime axis: u3, size: i64) Tensor {
    const extent = input.dim(axis);
    if (size == extent) {
        if (axis == 2) return input.scale(@as(i32, unity >> 4));
        return input.shiftRightArithmetic(integer(5)).addConstant(@as(i32, 2)).shiftRightArithmetic(integer(2));
    }
    const count: usize = @intCast(size);
    var indices: [2][limit]i32 = undefined;
    var weights: [2][limit]i32 = undefined;
    const inverse = 1.0 / (@as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(extent)));
    for (0..count) |i| {
        const pos: f32 = @floatCast((@as(f64, @floatFromInt(i)) + 0.5) * inverse - 0.5);
        const base = @floor(pos);
        const first: i64 = @intFromFloat(base);
        const fraction = if (axis == 2 and (first < 0 or first >= extent - 1)) 0 else pos - base;
        indices[0][i] = @intCast(std.math.clamp(first, 0, extent - 1));
        indices[1][i] = @intCast(std.math.clamp(first + 1, 0, extent - 1));
        weights[0][i] = @intCast(nearestEven((1 - fraction) * unity));
        weights[1][i] = @intCast(nearestEven(fraction * unity));
    }
    const vector: zml.Shape = .init(.{size}, .i32);
    const first = sample(input, axis, Tensor.constantTensor(vector, std.mem.sliceAsBytes(indices[0][0..count])), Tensor.constantTensor(vector, std.mem.sliceAsBytes(weights[0][0..count])));
    const second = sample(input, axis, Tensor.constantTensor(vector, std.mem.sliceAsBytes(indices[1][0..count])), Tensor.constantTensor(vector, std.mem.sliceAsBytes(weights[1][0..count])));
    // OpenCV's uint8 INTER_LINEAR narrows each weighted vertical term before rounding.
    if (axis == 2) return first.add(second).shiftRightArithmetic(integer(4));
    return first.shiftRightArithmetic(integer(16)).add(second.shiftRightArithmetic(integer(16)))
        .addConstant(@as(i32, 2)).shiftRightArithmetic(integer(2));
}

fn resize(image: Tensor, destination: Layout) Tensor {
    const height = image.dim(1);
    const width = image.dim(2);
    if (height == destination.height and width == destination.width) return image.convert(.i32);
    const value = image.convert(.i32);
    if (height == 2 * destination.height and width == 2 * destination.width) {
        // OpenCV selects its 2-by-2 area path for this exact downscale.
        return sumPairs(sumPairs(value, 1), 2).addConstant(@as(i32, 2)).shiftRightArithmetic(integer(2));
    }
    return linearAxis(linearAxis(value, 2, destination.width), 1, destination.height);
}

pub fn forward(bgr: Tensor) Tensor {
    const destination = layout(bgr.dim(1), bgr.dim(2)) catch unreachable;
    const vertical = destination.canvas_height - destination.height;
    const horizontal = destination.canvas_width - destination.width;
    const top = @divFloor(vertical, 2);
    const left = @divFloor(horizontal, 2);
    const pixels = resize(bgr, destination).withTags(.{ .image, .row, .column, .channel })
        .pad(@as(i32, 114), .{
        .row = Tensor.Pad{ .low = top, .high = vertical - top },
        .column = Tensor.Pad{ .low = left, .high = horizontal - left },
    });
    return pixels.transpose(.{ 0, 3, 1, 2 }).reverse(.{1}).convert(.f32).divByConst(@as(f32, 255));
}
