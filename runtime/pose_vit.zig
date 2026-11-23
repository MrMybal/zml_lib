const std = @import("std");
const l = @import("layers.zig");
const t = @import("transformer.zig");
const Tensor = l.Tensor;

pub const height = 256;
pub const width = 192;
pub const patch = 16;
pub const grid_height = height / patch;
pub const grid_width = width / patch;
pub const patch_count = grid_height * grid_width;
pub const channels = 1280;
const depth = 32;
const heads = 16;
const hidden = channels * 4;

pub fn parameter(comptime name: []const u8, dims: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

pub const schema = blk: {
    @setEvalBranchQuota(100_000);
    var blocks: [depth * 12]l.Parameter = undefined;
    for (0..depth) |idx| {
        const base = std.fmt.comptimePrint("backbone.blocks.{d}.", .{idx});
        blocks[idx * 12 ..][0..12].* = .{
            parameter(base ++ "norm1.weight", .{channels}),
            parameter(base ++ "norm1.bias", .{channels}),
            parameter(base ++ "attn.qkv.weight", .{ channels * 3, channels }),
            parameter(base ++ "attn.qkv.bias", .{channels * 3}),
            parameter(base ++ "attn.proj.weight", .{ channels, channels }),
            parameter(base ++ "attn.proj.bias", .{channels}),
            parameter(base ++ "norm2.weight", .{channels}),
            parameter(base ++ "norm2.bias", .{channels}),
            parameter(base ++ "mlp.fc1.weight", .{ hidden, channels }),
            parameter(base ++ "mlp.fc1.bias", .{hidden}),
            parameter(base ++ "mlp.fc2.weight", .{ channels, hidden }),
            parameter(base ++ "mlp.fc2.bias", .{channels}),
        };
    }
    break :blk [_]l.Parameter{
        parameter("backbone.pos_embed", .{ 1, patch_count + 1, channels }),
        parameter("backbone.patch_embed.proj.weight", .{ channels, 3, patch, patch }),
        parameter("backbone.patch_embed.proj.bias", .{channels}),
        parameter("backbone.last_norm.weight", .{channels}),
        parameter("backbone.last_norm.bias", .{channels}),
    } ++ blocks;
};

fn block(w: l.Weights, image: Tensor, index: usize) Tensor {
    var buffer: [80]u8 = undefined;
    const norm = t.norm(w, l.name(&buffer, "backbone.blocks.{d}.norm1", .{index}), image, 1e-6);
    const qkv = t.linear(w, l.name(&buffer, "backbone.blocks.{d}.attn.qkv", .{index}), norm, true)
        .reshape(.{ image.dim(0), patch_count, 3, channels });
    const attention = t.attention(qkv.slice(2, .single(0)), qkv.slice(2, .single(1)), qkv.slice(2, .single(2)), heads);
    const spatial = image.add(t.linear(w, l.name(&buffer, "backbone.blocks.{d}.attn.proj", .{index}), attention, true));
    const normalized = t.norm(w, l.name(&buffer, "backbone.blocks.{d}.norm2", .{index}), spatial, 1e-6);
    const activated = t.gelu(t.linear(w, l.name(&buffer, "backbone.blocks.{d}.mlp.fc1", .{index}), normalized, true));
    return spatial.add(t.linear(w, l.name(&buffer, "backbone.blocks.{d}.mlp.fc2", .{index}), activated, true));
}

pub fn encode(w: l.Weights, image: Tensor) Tensor {
    // The source Conv2d pads by two pixels on each side and uses stride 16.
    const patches = l.conv(w, "backbone.patch_embed.proj", image, patch, @splat(2), true);
    var x = patches.transpose(.{ 0, 2, 3, 1 }).reshape(.{ image.dim(0), patch_count, channels });
    const positions = w.get("backbone.pos_embed", "");
    x = x.add(positions.slice(1, .{ .start = 1 }).broad(x.shape()))
        .add(positions.slice(1, .{ .end = 1 }).broad(x.shape()));
    for (0..depth) |index| x = block(w, x, index);
    return t.norm(w, "backbone.last_norm", x, 1e-6);
}
