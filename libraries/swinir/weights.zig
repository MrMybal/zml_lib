const std = @import("std");
const l = @import("zml_lib/runtime").layers;
pub const blocks_per_stage = 6;
pub const stages = 6;
pub const channels = 180;
pub const heads = 6;

fn param(name: []const u8, shape: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(shape, .f32) };
}

fn pair(comptime name: []const u8, weight_shape: anytype, channels_out: i64) [2]l.Parameter {
    return .{ param(name ++ ".weight", weight_shape), param(name ++ ".bias", .{channels_out}) };
}

pub fn schemaFor(comptime window: i64) [490]l.Parameter {
    @setEvalBranchQuota(140_000);
    var specs: [490]l.Parameter = undefined;
    specs[0..10].* = pair("conv_first", .{ channels, 3, 3, 3 }, channels) ++
        pair("patch_embed.norm", .{channels}, channels) ++ pair("norm", .{channels}, channels) ++
        pair("conv_after_body", .{ channels, channels, 3, 3 }, channels) ++
        pair("conv_last", .{ 3, channels, 3, 3 }, 3);
    const step = 2 + 13 * blocks_per_stage;
    for (0..stages) |i| {
        const stage = std.fmt.comptimePrint("layers.{d}", .{i});
        const base = 10 + i * step;
        specs[base..][0..2].* = pair(stage ++ ".conv", .{ channels, channels, 3, 3 }, channels);
        for (0..blocks_per_stage) |j| {
            const b = stage ++ std.fmt.comptimePrint(".residual_group.blocks.{d}", .{j});
            specs[base + 2 + j * 13 ..][0..13].* =
                pair(b ++ ".norm1", .{channels}, channels) ++ pair(b ++ ".norm2", .{channels}, channels) ++
                [_]l.Parameter{param(b ++ ".attn.relative_position_bias_table", .{ (2 * window - 1) * (2 * window - 1), heads })} ++
                pair(b ++ ".attn.qkv", .{ 3 * channels, channels }, 3 * channels) ++
                pair(b ++ ".attn.proj", .{ channels, channels }, channels) ++
                pair(b ++ ".mlp.fc1", .{ 2 * channels, channels }, 2 * channels) ++
                pair(b ++ ".mlp.fc2", .{ channels, 2 * channels }, channels);
        }
    }
    return specs;
}
