const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const Tensor = l.Tensor;
pub const scale = 4;
const features = 64;
const stages = 33;

const schema = blk: {
    @setEvalBranchQuota(40_000);
    var specs: [stages * 3 + 2]l.Parameter = undefined;
    for (0..stages) |i| {
        const conv_name = std.fmt.comptimePrint("body.{d}", .{2 * i});
        specs[3 * i ..][0..3].* = .{
            .{ .name = conv_name ++ ".weight", .shape = .init(.{ features, if (i == 0) 3 else features, 3, 3 }, .f32) },
            .{ .name = conv_name ++ ".bias", .shape = .init(.{features}, .f32) },
            .{ .name = std.fmt.comptimePrint("body.{d}.weight", .{2 * i + 1}), .shape = .init(.{features}, .f32) },
        };
    }
    specs[stages * 3 ..][0..2].* = .{
        .{ .name = "body.66.weight", .shape = .init(.{ 3 * scale * scale, features, 3, 3 }, .f32) },
        .{ .name = "body.66.bias", .shape = .init(.{3 * scale * scale}, .f32) },
    };
    break :blk specs;
};

fn forward(parameters: []const Tensor, image: Tensor) Tensor {
    const weights: l.Weights = .{ .schema = &schema, .tensors = parameters };
    var key: [32]u8 = undefined;
    var x = image;
    for (0..stages) |i| {
        x = l.conv(weights, l.name(&key, "body.{d}", .{2 * i}), x, 1, @splat(1), true);
        const slopes = weights.get(l.name(&key, "body.{d}", .{2 * i + 1}), ".weight");
        const negative = x.minimum(.scalar(@as(f32, 0), .f32));
        x = x.relu().add(negative.mul(slopes.broadcast(x.shape(), &.{1})));
    }
    x = l.conv(weights, "body.66", x, 1, @splat(1), true);
    const h = image.dim(2);
    const w = image.dim(3);
    const residual = x.reshape(.{ 1, 3, scale, scale, h, w }).transpose(.{ 0, 1, 4, 2, 5, 3 })
        .reshape(.{ 1, 3, h * scale, w * scale });
    return residual.add(l.nearest2x(l.nearest2x(image)));
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn infer(self: *Model, h: u32, w: u32, input: []const f32, output: []f32) !u32 {
        if (h == 0 or w == 0) return error.InvalidDimensions;
        return self.graph.infer(forward, .init(.{ 1, 3, @as(i64, h), @as(i64, w) }, .f32), .init(.{ 1, 3, @as(i64, h) * scale, @as(i64, w) * scale }, .f32), input, output);
    }
};
