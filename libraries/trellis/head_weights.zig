const std = @import("std");
const Parameter = @import("zml_lib/runtime").layers.Parameter;

pub const Kind = enum { mesh, gaussian, radiance };
pub const Profile = struct {
    width: i64,
    heads: i64,
    blocks: usize,
    groups: i64,
    resolution: i64,
};
pub const official: Profile = .{ .width = 768, .heads = 12, .blocks = 12, .groups = 32, .resolution = 64 };
pub const fixture: Profile = .{ .width = 32, .heads = 4, .blocks = 2, .groups = 4, .resolution = 8 };
pub const gaussians = 32;
pub const rank = 16;
pub const coefficients = 8;

pub fn channels(kind: Kind) i64 {
    return switch (kind) {
        .mesh => 101,
        .gaussian => gaussians * 14,
        .radiance => rank * (3 * coefficients + 4),
    };
}

fn tensor(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn pair(comptime key: []const u8, width: i64, dims: anytype) [2]Parameter {
    return .{ tensor(key ++ ".weight", dims), tensor(key ++ ".bias", .{width}) };
}

fn upsample(comptime key: []const u8, source: i64, dest: i64) [10]Parameter {
    return pair(key ++ ".act_layers.0", source, .{source}) ++
        pair(key ++ ".out_layers.0.conv", dest, .{ dest, 3, 3, 3, source }) ++
        pair(key ++ ".out_layers.1", dest, .{dest}) ++
        pair(key ++ ".out_layers.3.conv", dest, .{ dest, 3, 3, 3, dest }) ++
        pair(key ++ ".skip_connection.conv", dest, .{ dest, 1, 1, 1, source });
}

pub fn schema(comptime p: Profile, comptime kind: Kind) [
    p.blocks * 8 + 4 + (switch (kind) {
        .gaussian => @as(usize, 1),
        .mesh => @as(usize, 20),
        .radiance => @as(usize, 0),
    })
]Parameter {
    @setEvalBranchQuota(100_000);
    const w = p.width;
    const in_out = pair("input_layer", w, .{ w, 8 }) ++
        pair("out_layer", channels(kind), .{ channels(kind), if (kind == .mesh) w / 8 else w });
    const extra = switch (kind) {
        .gaussian => [1]Parameter{tensor("offset_perturbation", .{ gaussians, 3 })},
        .mesh => upsample("upsample.0", w, w / 4) ++ upsample("upsample.1", w / 4, w / 8),
        .radiance => [0]Parameter{},
    };
    const prefix = in_out ++ extra;
    var result: [prefix.len + p.blocks * 8]Parameter = undefined;
    result[0..prefix.len].* = prefix;
    for (0..p.blocks) |i| {
        const base = std.fmt.comptimePrint("blocks.{d}.", .{i});
        result[prefix.len + i * 8 ..][0..8].* =
            pair(base ++ "attn.to_qkv", 3 * w, .{ 3 * w, w }) ++ pair(base ++ "attn.to_out", w, .{ w, w }) ++
            pair(base ++ "mlp.mlp.0", 4 * w, .{ 4 * w, w }) ++ pair(base ++ "mlp.mlp.2", w, .{ w, 4 * w });
    }
    return result;
}
