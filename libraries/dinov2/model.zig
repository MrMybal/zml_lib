const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const weights = @import("weights.zig");

pub const large = weights.large;
pub const small = weights.small;
pub const schemaFor = weights.schema;
pub const schema = schemaFor(large);
pub const channels = large.channels;
pub const patch_size = weights.patch_size;
pub const feature_blocks = large.outputs;

pub const Image = struct {
    height: i64,
    width: i64,
    pixels: usize,
    feature_count: usize,

    pub fn init(height: i64, width: i64) !Image {
        if (height < patch_size or width < patch_size or
            @mod(height, patch_size) != 0 or @mod(width, patch_size) != 0 or
            height > std.math.maxInt(i32) or width > std.math.maxInt(i32))
            return error.InvalidImageDimensions;
        const pixels = try std.math.mul(usize, @intCast(height), @intCast(width));
        _ = try std.math.mul(usize, pixels, 3 * @sizeOf(f32));
        const tokens = pixels / (patch_size * patch_size) + 1;
        const feature_count = try std.math.mul(usize, tokens, channels * feature_blocks.len);
        _ = try std.math.mul(usize, try std.math.add(usize, feature_count, pixels), @sizeOf(f32));
        return .{ .height = height, .width = width, .pixels = pixels, .feature_count = feature_count };
    }
};

fn positions(w: l.Weights, height: i64, width: i64) Tensor {
    const position = w.get("pretrained.pos_embed", "");
    const dim = position.dim(2);
    const grid = weights.base_grid;
    if (height == grid and width == grid) return position;
    const spatial = position.slice(1, .{ .start = 1 }).reshape(.{ 1, grid, grid, dim })
        .transpose(.{ 0, 3, 1, 2 });
    const scaled = runtime.spatial.bicubicHalfPixel(spatial, height, width, 0.1)
        .transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, height * width, dim });
    return Tensor.concatenate(&.{ position.slice(1, .{ .end = 1 }), scaled }, 1);
}

fn block(w: l.Weights, index: usize, x: Tensor, heads: i64) Tensor {
    var base: [32]u8 = undefined;
    var key: [64]u8 = undefined;
    const prefix = l.name(&base, "pretrained.blocks.{d}", .{index});
    const norm1 = t.norm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 1e-6);
    const qkv = t.linear(w, l.name(&key, "{s}.attn.qkv", .{prefix}), norm1, true)
        .reshape(.{ x.dim(0), x.dim(1), 3, x.dim(2) });
    const attended = t.attention(qkv.slice(2, .single(0)), qkv.slice(2, .single(1)), qkv.slice(2, .single(2)), heads);
    const projected = t.linear(w, l.name(&key, "{s}.attn.proj", .{prefix}), attended, true);
    const ls1 = w.get(prefix, ".ls1.gamma").broadcast(x.shape(), &.{2});
    const y = x.add(projected.mul(ls1));
    const norm2 = t.norm(w, l.name(&key, "{s}.norm2", .{prefix}), y, 1e-6);
    const hidden = t.gelu(t.linear(w, l.name(&key, "{s}.mlp.fc1", .{prefix}), norm2, true));
    const mlp = t.linear(w, l.name(&key, "{s}.mlp.fc2", .{prefix}), hidden, true);
    const ls2 = w.get(prefix, ".ls2.gamma").broadcast(y.shape(), &.{2});
    return y.add(mlp.mul(ls2));
}

pub fn forward(parameters: []const Tensor, image: Tensor) [feature_blocks.len]Tensor {
    return encode(large, parameters, image);
}

pub fn encode(comptime profile: weights.Architecture, parameters: []const Tensor, image: Tensor) [4]Tensor {
    const params = comptime schemaFor(profile);
    const w: l.Weights = .{ .schema = &params, .tensors = parameters };
    const patches = l.conv(w, "pretrained.patch_embed.proj", image, patch_size, @splat(0), true);
    const height = patches.dim(2);
    const width = patches.dim(3);
    const n = patches.dim(0);
    const tokens = patches.transpose(.{ 0, 2, 3, 1 }).reshape(.{ n, height * width, profile.channels });
    const cls = w.get("pretrained.cls_token", "").broadcast(.init(.{ n, 1, profile.channels }, .f32), &.{ 0, 1, 2 });
    const combined = Tensor.concatenate(&.{ cls, tokens }, 1);
    var x = combined.add(positions(w, height, width).broadcast(combined.shape(), &.{ 0, 1, 2 }));
    var features: [4]Tensor = undefined;
    var next: usize = 0;
    for (0..profile.layers) |i| {
        x = block(w, i, x, profile.heads);
        if (i == profile.outputs[next]) {
            features[next] = t.norm(w, "pretrained.norm", x, 1e-6);
            next += 1;
        }
    }
    return features;
}
