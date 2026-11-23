const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;

const pixel_step = 1 << 17;
const phase_unit = 1 << 30;
const max_taps = 13; // floor(W / 2) gives an input-to-output ratio at most 3.

pub fn halfSize(input: i64) i64 {
    return @max(1, @divFloor(input, 2));
}

const Window = struct {
    start: i64,
    weights: [max_taps]i64,
};

fn roundSigned(value: i64, divisor: i64) i64 {
    const half = @divTrunc(divisor, 2);
    return @divTrunc(value + (if (value < 0) -half else half), divisor);
}

fn cubic(distance: i64, unity: i64) i64 {
    if (distance >= 2 * phase_unit) return 0;
    // Mitchell–Netravali B=0, C=0.6 in libswscale's fixed-point coordinates.
    const parameter_unit = 1 << 24;
    const tension = 10_066_329;
    const square = distance * distance >> 30;
    const cube = square * distance >> 30;
    const numerator = if (distance < phase_unit)
        (12 * parameter_unit - 6 * tension) * cube + (-18 * parameter_unit + 6 * tension) * square + 6 * parameter_unit * phase_unit
    else
        -6 * tension * cube + 30 * tension * square - 48 * tension * distance + 24 * tension * phase_unit;
    return @divTrunc(numerator, @divExact(1 << 54, unity));
}

fn tapCount(windows: []Window, initial: usize, unity: i64) usize {
    const cutoff: u64 = @intCast(@divTrunc(unity, 500));
    var longest: usize = 1;
    var i = windows.len;
    while (i != 0) {
        i -= 1;
        const window = &windows[i];
        var tail: u64 = 0;
        for (0..initial) |_| {
            tail += @abs(window.weights[0]);
            if (tail > cutoff or (i + 1 < windows.len and window.start >= windows[i + 1].start)) break;
            for (0..initial - 1) |tap| window.weights[tap] = window.weights[tap + 1];
            window.weights[initial - 1] = 0;
            window.start += 1;
        }
        tail = 0;
        var keep = initial;
        while (keep > 1) : (keep -= 1) {
            tail += @abs(window.weights[keep - 1]);
            if (tail > cutoff) break;
        }
        longest = @max(longest, keep);
    }
    // Padding the table to four taps retains the same tail used by the x64 scaler.
    return @min(initial, std.mem.alignForward(usize, longest, 4));
}

fn axis(input: Tensor, comptime dimension: u3, output: i64, precision: u5) Tensor {
    const extent = input.dim(dimension);
    if (extent == output) return input.scale(@as(i32, 1) << precision);
    const count: usize = @intCast(output);
    const initial: usize = @intCast(@max(1, @min(extent - 2, 1 + @divTrunc(4 * extent + output - 1, output))));
    const increment = @divTrunc((extent << 16) + @divTrunc(output, 2), output);
    const unity = @as(i64, 1) << @as(u6, @intCast(54 - std.math.log2_int(u64, @intCast(@divTrunc(extent, output)))));
    const allocator = zml.Compiler.current().arena.allocator();
    const windows = allocator.alloc(Window, count) catch @panic("Pose scale table allocation failed");
    for (windows, 0..) |*window, i| {
        const center = increment - @divExact(pixel_step, 2) + 2 * @as(i64, @intCast(i)) * increment;
        const start = @divTrunc(center - (@as(i64, @intCast(initial)) - 2) * @divExact(pixel_step, 2), pixel_step);
        window.* = .{ .start = start, .weights = @splat(0) };
        for (window.weights[0..initial], 0..) |*weight, tap| {
            const sample = start + @as(i64, @intCast(tap));
            const delta: i64 = @intCast(@abs(sample * pixel_step - center));
            weight.* = cubic(@divTrunc(delta * 8192 * output, extent), unity);
        }
    }
    const taps = tapCount(windows, initial, unity);
    const last_start = extent - @as(i64, @intCast(taps));
    const starts = allocator.alloc(i32, count) catch @panic("Pose scale table allocation failed");
    var first_offset: i64 = 0;
    var last_offset: i64 = 0;
    for (windows, starts, 0..) |window, *start, i| {
        start.* = @intCast(std.math.clamp(window.start, 0, last_start));
        const offset = start.* - 2 * @as(i64, @intCast(i));
        first_offset = @min(first_offset, offset);
        last_offset = @max(last_offset, offset + @as(i64, @intCast(taps)) - 1);
    }
    const offsets: usize = @intCast(last_offset - first_offset + 1);
    const coefficients = allocator.alloc(i32, count * offsets) catch @panic("Pose scale table allocation failed");
    @memset(coefficients, 0);
    for (windows, 0..) |window, i| {
        const start: i64 = starts[i];
        const row: usize = @intCast(start - 2 * @as(i64, @intCast(i)) - first_offset);
        var folded: [max_taps]i64 = @splat(0);
        for (window.weights[0..taps], 0..) |weight, tap| {
            const source = std.math.clamp(window.start + @as(i64, @intCast(tap)), 0, extent - 1);
            folded[@intCast(source - start)] += weight;
        }
        var sum: i64 = 0;
        for (folded[0..taps]) |weight| sum += weight;
        const divisor = @max(1, roundSigned(sum, @as(i64, 1) << precision));
        var error_term: i64 = 0;
        for (folded[0..taps], 0..) |weight, tap| {
            const value = weight + error_term;
            const rounded = roundSigned(value, divisor);
            coefficients[(row + tap) * count + i] = @intCast(rounded);
            error_term = value - rounded * divisor;
        }
    }
    // Express each tap as a static stride-two slice. Its position-dependent
    // weight accounts for the fractional phase and for the clamped borders.
    const padding: Tensor.Pad = .{ .low = -first_offset, .high = @max(0, 2 * (output - 1) + last_offset + 1 - extent) };
    const tagged = input.withTags(.{ .frame, .row, .column, .channel });
    const padded = if (dimension == 1) tagged.pad(@as(i32, 0), .{ .row = padding }) else tagged.pad(@as(i32, 0), .{ .column = padding });
    const result_shape = input.shape().setDim(dimension, output);
    var result = integer(0).broad(result_shape);
    for (0..offsets) |tap| {
        const start: i64 = @intCast(tap);
        const samples = padded.slice(dimension, .{ .start = start, .end = start + 2 * (output - 1) + 1, .step = 2 });
        const weight = Tensor.constantTensor(.init(.{output}, .i32), std.mem.sliceAsBytes(coefficients[tap * count ..][0..count]));
        result = result.add(samples.reshape(result_shape).mul(weight.broadcast(result_shape, &.{dimension})));
    }
    return result;
}

fn integer(value: i32) Tensor {
    return .scalar(value, .i32);
}

fn matrix(input: Tensor, weights: [3]i32, offset: i32, shift: i32) Tensor {
    return input.slice(3, .single(0)).scale(weights[0])
        .add(input.slice(3, .single(1)).scale(weights[1]))
        .add(input.slice(3, .single(2)).scale(weights[2]))
        .addConstant(offset).shiftRightArithmetic(integer(shift));
}

pub fn forward(rgb: Tensor) Tensor {
    const n = rgb.dim(0);
    const height = rgb.dim(1);
    const width = rgb.dim(2);
    const h = halfSize(height);
    const w = halfSize(width);
    if (height == h and width == w) return rgb;
    const image = rgb.convert(.i32);
    const paired = @mod(width, 2) == 0;
    // At exact horizontal halving, the RGB input path forms one chroma
    // sample per pair without rounding that pair to RGB8 first.
    const colors = if (paired) blk: {
        const pairs = image.reshape(.{ n, height, w, 2, 3 });
        break :blk pairs.slice(3, .single(0)).add(pairs.slice(3, .single(1))).reshape(.{ n, height, w, 3 });
    } else image;
    const cb = matrix(colors, .{ -4865, -9528, 14392 }, if (paired) 0x4001 * 512 else 0x4001 * 256, if (paired) 10 else 9);
    const cr = matrix(colors, .{ 14392, -12061, -2332 }, if (paired) 0x4001 * 512 else 0x4001 * 256, if (paired) 10 else 9);
    const y = matrix(image, .{ 8414, 16519, 3208 }, 0x801 * 256, 9);
    const chroma = Tensor.stack(&.{ cb, cr }, 3, .channel).reshape(.{ n, height, colors.dim(2), 2 });
    const luma = axis(y.reshape(.{ n, height, width, 1 }), 2, w, 14);
    const uv = axis(chroma, 2, w, 14);
    const horizontal = Tensor.concatenate(&.{ luma, uv }, 3).shiftRightArithmetic(integer(13)).minimum(integer(32767));
    const vertical = axis(horizontal, 1, h, 12);
    const y_term = vertical.slice(3, .single(0)).addConstant(@as(i32, 512)).shiftRightArithmetic(integer(10))
        .subConstant(@as(i32, 8192)).scale(@as(i32, 9539)).addConstant(@as(i32, 1 << 21));
    const u_term = vertical.slice(3, .single(1)).addConstant(@as(i32, 512 - (128 << 19))).shiftRightArithmetic(integer(10));
    const v_term = vertical.slice(3, .single(2)).addConstant(@as(i32, 512 - (128 << 19))).shiftRightArithmetic(integer(10));
    const red = y_term.add(v_term.scale(@as(i32, 13075)));
    const green = y_term.add(v_term.scale(@as(i32, -6660))).add(u_term.scale(@as(i32, -3209)));
    const blue = y_term.add(u_term.scale(@as(i32, 16525)));
    return Tensor.stack(&.{ red, green, blue }, 3, .channel).reshape(.{ n, h, w, 3 })
        .shiftRightArithmetic(integer(22)).clamp(integer(0), integer(255)).convert(.u8);
}
