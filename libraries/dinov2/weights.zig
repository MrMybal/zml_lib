const std = @import("std");
const Parameter = @import("zml_lib/runtime").layers.Parameter;

pub const Architecture = struct {
    channels: i64,
    layers: usize,
    heads: i64,
    outputs: [4]usize,
};

pub const large: Architecture = .{ .channels = 1024, .layers = 24, .heads = 16, .outputs = .{ 4, 11, 17, 23 } };
pub const small: Architecture = .{ .channels = 384, .layers = 12, .heads = 6, .outputs = .{ 2, 5, 8, 11 } };
pub const base_grid = 37;
pub const patch_size = 14;

fn parameter(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

pub fn schema(comptime profile: Architecture) [7 + profile.layers * 14]Parameter {
    @setEvalBranchQuota(100_000);
    const channels = profile.channels;
    var result: [7 + profile.layers * 14]Parameter = undefined;
    result[0..7].* = .{
        parameter("pretrained.cls_token", .{ 1, 1, channels }),
        parameter("pretrained.mask_token", .{ 1, channels }),
        parameter("pretrained.norm.weight", .{channels}),
        parameter("pretrained.norm.bias", .{channels}),
        parameter("pretrained.patch_embed.proj.weight", .{ channels, 3, patch_size, patch_size }),
        parameter("pretrained.patch_embed.proj.bias", .{channels}),
        parameter("pretrained.pos_embed", .{ 1, base_grid * base_grid + 1, channels }),
    };
    for (0..profile.layers) |i| {
        const prefix = std.fmt.comptimePrint("pretrained.blocks.{d}", .{i});
        result[7 + i * 14 ..][0..14].* = .{
            parameter(prefix ++ ".attn.proj.weight", .{ channels, channels }),
            parameter(prefix ++ ".attn.proj.bias", .{channels}),
            parameter(prefix ++ ".attn.qkv.weight", .{ 3 * channels, channels }),
            parameter(prefix ++ ".attn.qkv.bias", .{3 * channels}),
            parameter(prefix ++ ".ls1.gamma", .{channels}),
            parameter(prefix ++ ".ls2.gamma", .{channels}),
            parameter(prefix ++ ".mlp.fc1.weight", .{ 4 * channels, channels }),
            parameter(prefix ++ ".mlp.fc1.bias", .{4 * channels}),
            parameter(prefix ++ ".mlp.fc2.weight", .{ channels, 4 * channels }),
            parameter(prefix ++ ".mlp.fc2.bias", .{channels}),
            parameter(prefix ++ ".norm1.weight", .{channels}),
            parameter(prefix ++ ".norm1.bias", .{channels}),
            parameter(prefix ++ ".norm2.weight", .{channels}),
            parameter(prefix ++ ".norm2.bias", .{channels}),
        };
    }
    return result;
}
