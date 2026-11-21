const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const Tensor = l.Tensor;

pub const latent_channels = 16;
pub const factor = 8;
const width = 64;
const residuals = [_]usize{ 3, 4, 5, 8, 9, 10, 13, 14, 15, 18 };
const upprojections = [_]usize{ 7, 12, 17 };

fn weight(name: []const u8, out: i64, in: i64) l.Parameter {
    return .{ .name = name, .shape = .init(.{ out, in, 3, 3 }, .f32) };
}

fn pair(comptime name: []const u8, out: i64, in: i64) [2]l.Parameter {
    return .{ weight(name ++ ".weight", out, in), .{ .name = name ++ ".bias", .shape = .init(.{out}, .f32) } };
}

const schema = blk: {
    @setEvalBranchQuota(30_000);
    var specs: [67]l.Parameter = undefined;
    specs[0..4].* = pair("1", width, latent_channels) ++ pair("19", 3, width);
    for (upprojections, 0..) |i, p| specs[4 + p] = weight(std.fmt.comptimePrint("{d}.weight", .{i}), width, width);
    for (residuals, 0..) |i, p| {
        for (0..3) |j| {
            const name = std.fmt.comptimePrint("{d}.conv.{d}", .{ i, 2 * j });
            specs[7 + 6 * p + 2 * j ..][0..2].* = pair(name, width, width);
        }
    }
    break :blk specs;
};

fn block(w: l.Weights, index: usize, x: Tensor) Tensor {
    var key: [32]u8 = undefined;
    const a = l.conv(w, l.name(&key, "{d}.conv.0", .{index}), x, 1, @splat(1), true).relu();
    const b = l.conv(w, l.name(&key, "{d}.conv.2", .{index}), a, 1, @splat(1), true).relu();
    return x.add(l.conv(w, l.name(&key, "{d}.conv.4", .{index}), b, 1, @splat(1), true)).relu();
}

fn forward(parameters: []const Tensor, latent: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    var x = l.conv(w, "1", latent.divByConst(@as(f32, 3)).tanh().scale(@as(f32, 3)), 1, @splat(1), true).relu();
    var key: [32]u8 = undefined;
    for (upprojections, 0..) |index, stage| {
        for (residuals[stage * 3 ..][0..3]) |i| x = block(w, i, x);
        x = l.conv(w, l.name(&key, "{d}", .{index}), l.nearest2x(x), 1, @splat(1), false);
    }
    return l.conv(w, "19", block(w, residuals[9], x), 1, @splat(1), true);
}

pub const Decoder = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Decoder {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Decoder) void {
        self.graph.deinit();
    }

    pub fn decode(self: *Decoder, height: u32, image_width: u32, latent: []const f32, rgb: []f32) !u32 {
        if (height == 0 or image_width == 0) return error.InvalidDimensions;
        const h: i64 = height;
        const w: i64 = image_width;
        return self.graph.infer(forward, .init(.{ 1, latent_channels, h, w }, .f32), .init(.{ 1, 3, h * factor, w * factor }, .f32), latent, rgb);
    }
};
