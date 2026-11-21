const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const Tensor = l.Tensor;
const schema = @import("weights.zig").schema;

const same = [4]i64{ 1, 1, 1, 1 };
const valid = [4]i64{ 0, 0, 0, 0 };
const steps = [_]u32{ 8, 16, 32 };
const anchor_sizes = [3][2]f32{ .{ 16, 32 }, .{ 64, 128 }, .{ 256, 512 } };
pub const raw_fields = 16;
pub const face_fields = 15;

fn convNorm(w: l.Weights, prefix: []const u8, x: Tensor, stride: i64, padding: [4]i64, act: bool) Tensor {
    var key: [64]u8 = undefined;
    const conv = l.conv(w, l.name(&key, "{s}.0", .{prefix}), x, stride, padding, false);
    const normalized = l.batchNorm(w, l.name(&key, "{s}.1", .{prefix}), conv);
    return if (act) normalized.relu() else normalized;
}

fn bottleneck(w: l.Weights, x: Tensor, stage: usize, index: usize) Tensor {
    var block_buffer: [48]u8 = undefined;
    var key: [64]u8 = undefined;
    const block = l.name(&block_buffer, "body.layer{d}.{d}", .{ stage + 1, index });
    const stride: i64 = if (stage > 0 and index == 0) 2 else 1;
    var y = l.conv(w, l.name(&key, "{s}.conv1", .{block}), x, 1, valid, false);
    y = l.batchNorm(w, l.name(&key, "{s}.bn1", .{block}), y).relu();
    y = l.conv(w, l.name(&key, "{s}.conv2", .{block}), y, stride, same, false);
    y = l.batchNorm(w, l.name(&key, "{s}.bn2", .{block}), y).relu();
    y = l.conv(w, l.name(&key, "{s}.conv3", .{block}), y, 1, valid, false);
    y = l.batchNorm(w, l.name(&key, "{s}.bn3", .{block}), y);
    const skip = if (index == 0)
        convNorm(w, l.name(&key, "{s}.downsample", .{block}), x, stride, valid, false)
    else
        x;
    return y.add(skip).relu();
}

fn nearestTo(x: Tensor, target: Tensor) Tensor {
    const n, const c, const h, const w = x.dims()[0..4].*;
    const to_h = target.dim(2);
    const to_w = target.dim(3);
    if (to_h == 2 * h and to_w == 2 * w) return l.nearest2x(x);
    const rows = Tensor.iota(.init(.{ .row = to_h }, .i32), 0)
        .scale(@as(i32, @intCast(h))).divByConst(@as(i32, @intCast(to_h)));
    const cols = Tensor.iota(.init(.{ .col = to_w }, .i32), 0)
        .scale(@as(i32, @intCast(w))).divByConst(@as(i32, @intCast(to_w)));
    return x.reshape(.{ .batch = n, .channel = c, .y = h, .x = w })
        .gather(.{ .y = rows }, .{}).gather(.{ .x = cols }, .{}).reshape(.{ n, c, to_h, to_w });
}

fn ssh(w: l.Weights, level: usize, x: Tensor) Tensor {
    var key: [48]u8 = undefined;
    const a = convNorm(w, l.name(&key, "ssh{d}.conv3X3", .{level}), x, 1, same, false);
    const b = convNorm(w, l.name(&key, "ssh{d}.conv5X5_1", .{level}), x, 1, same, true);
    const c = convNorm(w, l.name(&key, "ssh{d}.conv5X5_2", .{level}), b, 1, same, false);
    const d = convNorm(w, l.name(&key, "ssh{d}.conv7X7_2", .{level}), b, 1, same, true);
    const e = convNorm(w, l.name(&key, "ssh{d}.conv7x7_3", .{level}), d, 1, same, false);
    return Tensor.concatenate(&.{ a, c, e }, 1).relu();
}

fn head(w: l.Weights, x: Tensor, prefix: []const u8, level: usize, fields: i64) Tensor {
    var key: [48]u8 = undefined;
    return l.conv(w, l.name(&key, "{s}.{d}.conv1x1", .{ prefix, level }), x, 1, valid, true)
        .transpose(.{ 0, 2, 3, 1 }).reshape(.{ 2 * x.dim(2) * x.dim(3), fields });
}

fn poolAxis(x: Tensor, axis: u3) Tensor {
    const output_size = @divFloor(x.dim(axis) - 1, 2);
    const end = output_size * 2 - 1;
    const a = x.slice(axis, .{ .end = end, .step = 2 });
    const b = x.slice(axis, .{ .start = 1, .end = end + 1, .step = 2 });
    const c = x.slice(axis, .{ .start = 2, .end = end + 2, .step = 2 });
    return a.maximum(b).maximum(c);
}

fn stemMaxPool(x: Tensor) Tensor {
    // Separable 3x3 max-pooling with stride 2. Slices avoid IREE's reduce-window lowering.
    const padded = x.withTags(.{ .n, .c, .h, .w }).pad(-std.math.inf(f32), .{
        .h = Tensor.Pad{ .low = 1, .high = 1 },
        .w = Tensor.Pad{ .low = 1, .high = 1 },
    });
    const result = poolAxis(poolAxis(padded, 3), 2);
    return result.reshape(.{ result.dim(0), result.dim(1), result.dim(2), result.dim(3) });
}

fn forward(parameters: []const Tensor, image: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    var y = l.conv(w, "body.conv1", image, 2, @splat(3), false);
    y = l.batchNorm(w, "body.bn1", y).relu();
    y = stemMaxPool(y);
    var stages: [3]Tensor = undefined;
    for ([_]usize{ 3, 4, 6, 3 }, 0..) |count, stage| {
        for (0..count) |index| y = bottleneck(w, y, stage, index);
        if (stage != 0) stages[stage - 1] = y;
    }
    const high = convNorm(w, "fpn.output3", stages[2], 1, valid, true);
    const middle = convNorm(w, "fpn.output2", stages[1], 1, valid, true);
    const low = convNorm(w, "fpn.output1", stages[0], 1, valid, true);
    const fused_middle = convNorm(w, "fpn.merge2", middle.add(nearestTo(high, middle)), 1, same, true);
    const fused_low = convNorm(w, "fpn.merge1", low.add(nearestTo(fused_middle, low)), 1, same, true);
    var levels: [3]Tensor = undefined;
    for ([_]Tensor{ fused_low, fused_middle, high }, 0..) |level, i| {
        const feature = ssh(w, i + 1, level);
        levels[i] = Tensor.concatenate(&.{
            head(w, feature, "BboxHead", i, 4),      head(w, feature, "ClassHead", i, 2).softmax(1),
            head(w, feature, "LandmarkHead", i, 10),
        }, 1);
    }
    return Tensor.concatenate(&levels, 0);
}

fn cells(length: u32, step: u32) u32 {
    return (length - 1) / step + 1;
}

pub fn anchors(width: u32, height: u32) !usize {
    if (width == 0 or height == 0 or width > 8192 or height > 8192) return error.InvalidDimensions;
    var total: usize = 0;
    for (steps) |step| total += 2 * @as(usize, cells(width, step)) * cells(height, step);
    return total;
}

const Face = [face_fields]f32;

fn higherScoreFirst(_: void, a: Face, b: Face) bool {
    return a[4] > b[4];
}

fn overlaps(a: Face, b: Face, limit: f32) bool {
    const iw = @max(0, @min(a[2], b[2]) - @max(a[0], b[0]) + 1);
    const ih = @max(0, @min(a[3], b[3]) - @max(a[1], b[1]) + 1);
    const intersection = iw * ih;
    const a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    const b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return intersection / (a_area + b_area - intersection) > limit;
}

pub fn decode(allocator: std.mem.Allocator, width: u32, height: u32, raw: []const f32, output: []f32) !usize {
    const count = try anchors(width, height);
    if (raw.len != count * raw_fields) return error.InvalidBufferSize;
    var faces: std.ArrayList(Face) = .empty;
    defer faces.deinit(allocator);
    const width_f: f32 = @floatFromInt(width);
    const height_f: f32 = @floatFromInt(height);
    var index: usize = 0;
    for (steps, anchor_sizes) |step, sizes| {
        const step_f: f32 = @floatFromInt(step);
        for (0..cells(height, step)) |row| {
            const cy = (@as(f32, @floatFromInt(row)) + 0.5) * step_f / height_f;
            for (0..cells(width, step)) |col| {
                const cx = (@as(f32, @floatFromInt(col)) + 0.5) * step_f / width_f;
                for (sizes) |size| {
                    const pred = raw[index..][0..raw_fields];
                    index += raw_fields;
                    const score = pred[5];
                    if (score <= 0.8 or !std.math.isFinite(score)) continue;
                    const sw = size / width_f;
                    const sh = size / height_f;
                    const center_x = cx + pred[0] * 0.1 * sw;
                    const center_y = cy + pred[1] * 0.1 * sh;
                    const bw = sw * @exp(pred[2] * 0.2);
                    const bh = sh * @exp(pred[3] * 0.2);
                    const left = center_x - bw / 2;
                    const top = center_y - bh / 2;
                    var face: Face = undefined;
                    face[0..5].* = .{ left * width_f, top * height_f, (left + bw) * width_f, (top + bh) * height_f, score };
                    for (0..5) |landmark| {
                        face[5 + 2 * landmark] = (cx + pred[6 + 2 * landmark] * 0.1 * sw) * width_f;
                        face[6 + 2 * landmark] = (cy + pred[7 + 2 * landmark] * 0.1 * sh) * height_f;
                    }
                    for (face) |value| {
                        if (!std.math.isFinite(value)) return error.NonFiniteDetection;
                    }
                    try faces.append(allocator, face);
                }
            }
        }
    }
    std.mem.sort(Face, faces.items, {}, higherScoreFirst);
    var kept: usize = 0;
    for (faces.items) |face| {
        var suppressed = false;
        for (faces.items[0..kept]) |previous| {
            if (overlaps(previous, face, 0.4)) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        faces.items[kept] = face;
        kept += 1;
    }
    if (output.len < kept * face_fields) return error.InvalidBufferSize;
    for (faces.items[0..kept], 0..) |face, i| output[i * face_fields ..][0..face_fields].* = face;
    return kept;
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn infer(self: *Model, width: u32, height: u32, input: []const f32, output: []f32) !u32 {
        const count = try anchors(width, height);
        return self.graph.infer(forward, .init(.{ 1, 3, @as(i64, height), @as(i64, width) }, .f32), .init(.{ @as(i64, @intCast(count)), raw_fields }, .f32), input, output);
    }
};
