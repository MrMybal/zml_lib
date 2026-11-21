const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const schemas = @import("weights.zig");

pub const Profile = enum { face, hand, person };
const same = [4]i64{ 1, 1, 1, 1 };
const valid = [4]i64{ 0, 0, 0, 0 };
pub const mask_channels = 32;

fn schema(profile: Profile) []const l.Parameter {
    return switch (profile) {
        .face => &schemas.face,
        .hand => &schemas.hand,
        .person => &schemas.person,
    };
}

fn cbs(w: l.Weights, prefix: []const u8, x: Tensor, stride: i64, padding: [4]i64) Tensor {
    var key: [80]u8 = undefined;
    const y = l.conv(w, l.name(&key, "{s}.conv", .{prefix}), x, stride, padding, false);
    return l.batchNormEpsilon(w, l.name(&key, "{s}.bn", .{prefix}), y, 1e-3).silu();
}

fn layerConv(w: l.Weights, index: usize, x: Tensor, stride: i64) Tensor {
    var key: [32]u8 = undefined;
    return cbs(w, l.name(&key, "model.{d}", .{index}), x, stride, same);
}

fn c2f(w: l.Weights, index: usize, x: Tensor, repeat: usize, residual: bool) Tensor {
    var key: [64]u8 = undefined;
    const split = cbs(w, l.name(&key, "model.{d}.cv1", .{index}), x, 1, valid);
    const half = @divExact(split.dim(1), 2);
    var features: [6]Tensor = undefined;
    features[0] = split.slice(1, .{ .end = half });
    features[1] = split.slice(1, .{ .start = half });
    for (0..repeat) |i| {
        const last = features[i + 1];
        const hidden = cbs(w, l.name(&key, "model.{d}.m.{d}.cv1", .{ index, i }), last, 1, same);
        const out = cbs(w, l.name(&key, "model.{d}.m.{d}.cv2", .{ index, i }), hidden, 1, same);
        features[i + 2] = if (residual) out.add(last) else out;
    }
    return cbs(w, l.name(&key, "model.{d}.cv2", .{index}), Tensor.concatenate(features[0 .. repeat + 2], 1), 1, valid);
}

fn maxPoolAxis(x: Tensor, axis: u3) Tensor {
    const size = x.dim(axis) - 4;
    var result = x.slice(axis, .{ .end = size });
    for (1..5) |offset| {
        const start: i64 = @intCast(offset);
        result = result.maximum(x.slice(axis, .{ .start = start, .end = start + size }));
    }
    return result;
}

fn maxPool(x: Tensor) Tensor {
    const y = x.withTags(.{ .n, .c, .h, .w }).pad(-std.math.inf(f32), .{
        .h = Tensor.Pad{ .low = 2, .high = 2 },
        .w = Tensor.Pad{ .low = 2, .high = 2 },
    });
    return maxPoolAxis(maxPoolAxis(y, 3), 2).reshape(x.shape());
}

fn sppf(w: l.Weights, x: Tensor) Tensor {
    const a = cbs(w, "model.9.cv1", x, 1, valid);
    const b = maxPool(a);
    const c = maxPool(b);
    const d = maxPool(c);
    return cbs(w, "model.9.cv2", Tensor.concatenate(&.{ a, b, c, d }, 1), 1, valid);
}

fn head(w: l.Weights, x: Tensor, branch: usize, index: usize) Tensor {
    var key: [56]u8 = undefined;
    const a = cbs(w, l.name(&key, "model.22.cv{d}.{d}.0", .{ branch, index }), x, 1, same);
    const b = cbs(w, l.name(&key, "model.22.cv{d}.{d}.1", .{ branch, index }), a, 1, same);
    return l.conv(w, l.name(&key, "model.22.cv{d}.{d}.2", .{ branch, index }), b, 1, valid, true);
}

fn decoded(w: l.Weights, regression: Tensor, confidence: Tensor, stride: i64) Tensor {
    const h = regression.dim(2);
    const width = regression.dim(3);
    const grid = h * width;
    const distribution = regression.reshape(.{ 4, 16, grid }).softmax(1);
    const bins = w.get("model.22.dfl.conv", ".weight").reshape(.{16}).broadcast(distribution.shape(), &.{1});
    const d = distribution.mul(bins).sum(1).reshape(.{ 4, h, width });
    const plane: @TypeOf(d.shape()) = .init(.{ h, width }, .f32);
    const x = Tensor.iota(plane, 1).convert(.f32).addConstant(@as(f32, 0.5));
    const y = Tensor.iota(plane, 0).convert(.f32).addConstant(@as(f32, 0.5));
    const x0 = x.sub(d.slice(0, .single(0)));
    const y0 = y.sub(d.slice(0, .single(1)));
    const x1 = x.add(d.slice(0, .single(2)));
    const y1 = y.add(d.slice(0, .single(3)));
    const scale: f32 = @floatFromInt(stride);
    return Tensor.stack(&.{
        x0.add(x1).scale(scale * 0.5),               y0.add(y1).scale(scale * 0.5),
        x1.sub(x0).scale(scale),                     y1.sub(y0).scale(scale),
        confidence.sigmoid().reshape(.{ h, width }),
    }, 0, .field).reshape(.{ 5, grid });
}

fn forward(comptime profile: Profile, parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = schema(profile), .tensors = parameters };
    const repeats: usize = if (profile == .hand) 1 else 2;
    const x0 = layerConv(w, 0, input, 2);
    const x1 = layerConv(w, 1, x0, 2);
    const x2 = c2f(w, 2, x1, repeats, true);
    const x3 = layerConv(w, 3, x2, 2);
    const small = c2f(w, 4, x3, 2 * repeats, true);
    const x5 = layerConv(w, 5, small, 2);
    const medium = c2f(w, 6, x5, 2 * repeats, true);
    const x7 = layerConv(w, 7, medium, 2);
    const x8 = c2f(w, 8, x7, repeats, true);
    const large = sppf(w, x8);
    const top = c2f(w, 12, Tensor.concatenate(&.{ l.nearest2x(large), medium }, 1), repeats, false);
    const p8 = c2f(w, 15, Tensor.concatenate(&.{ l.nearest2x(top), small }, 1), repeats, false);
    const p16 = c2f(w, 18, Tensor.concatenate(&.{ layerConv(w, 16, p8, 2), top }, 1), repeats, false);
    const p32 = c2f(w, 21, Tensor.concatenate(&.{ layerConv(w, 19, p16, 2), large }, 1), repeats, false);
    var predictions: [3]Tensor = undefined;
    for ([_]Tensor{ p8, p16, p32 }, [_]i64{ 8, 16, 32 }, 0..) |features, stride, i| {
        const boxes = decoded(w, head(w, features, 2, i), head(w, features, 3, i), stride);
        predictions[i] = if (profile == .person)
            Tensor.concatenate(&.{ boxes, head(w, features, 4, i).reshape(.{ mask_channels, features.dim(2) * features.dim(3) }) }, 0)
        else
            boxes;
    }
    const detections = Tensor.concatenate(&predictions, 1).flatten();
    if (profile != .person) return detections;
    var masks = cbs(w, "model.22.proto.cv1", p8, 1, same);
    masks = t.deconv2x(w, "model.22.proto.upsample", masks);
    masks = cbs(w, "model.22.proto.cv2", masks, 1, same);
    masks = cbs(w, "model.22.proto.cv3", masks, 1, valid);
    return Tensor.concatenate(&.{ detections, masks.flatten() }, 0);
}

fn detectFace(p: []const Tensor, x: Tensor) Tensor {
    return forward(.face, p, x);
}
fn detectHand(p: []const Tensor, x: Tensor) Tensor {
    return forward(.hand, p, x);
}
fn segmentPerson(p: []const Tensor, x: Tensor) Tensor {
    return forward(.person, p, x);
}

pub fn outputCount(profile: Profile, width: u32, height: u32) !usize {
    if (width == 0 or height == 0 or width % 32 != 0 or height % 32 != 0 or width > 4096 or height > 4096)
        return error.InvalidDimensions;
    const cells = @as(usize, width / 32) * (height / 32) * 21;
    return if (profile == .person) cells * (5 + mask_channels) + @as(usize, width / 4) * (height / 4) * mask_channels else cells * 5;
}

pub const Model = struct {
    graph: runtime.TensorModel,
    profile: Profile,

    pub fn load(session: *const runtime.Session, profile: Profile, path: []const u8) !Model {
        return .{ .graph = try .load(session, schema(profile), path), .profile = profile };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn infer(self: *Model, width: u32, height: u32, input: []const f32, output: []f32) !u32 {
        const input_shape = .{ 1, 3, @as(i64, height), @as(i64, width) };
        const result_shape = .{@as(i64, @intCast(try outputCount(self.profile, width, height)))};
        return switch (self.profile) {
            .face => self.graph.infer(detectFace, .init(input_shape, .f32), .init(result_shape, .f32), input, output),
            .hand => self.graph.infer(detectHand, .init(input_shape, .f32), .init(result_shape, .f32), input, output),
            .person => self.graph.infer(segmentPerson, .init(input_shape, .f32), .init(result_shape, .f32), input, output),
        };
    }
};
