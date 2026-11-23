const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const weights = @import("weights.zig");

pub const official = weights.official;
pub const fixture = weights.fixture;
pub const Profile = weights.Profile;
pub const encoderSchema = weights.encoderSchema;
pub const fieldSchema = weights.fieldSchema;
pub const radius: f32 = 0.87;
pub const field_channels = 4;
pub const query_batch: usize = 8192;
const dino = "image_tokenizer.model.";

fn dinoBlock(w: l.Weights, index: usize, x: Tensor, heads: i64) Tensor {
    var base: [64]u8 = undefined;
    var key: [112]u8 = undefined;
    const prefix = l.name(&base, dino ++ "encoder.layer.{d}.", .{index});
    const n = t.norm(w, l.name(&key, "{s}layernorm_before", .{prefix}), x, 1e-12);
    const q = t.linear(w, l.name(&key, "{s}attention.attention.query", .{prefix}), n, true);
    const k = t.linear(w, l.name(&key, "{s}attention.attention.key", .{prefix}), n, true);
    const v = t.linear(w, l.name(&key, "{s}attention.attention.value", .{prefix}), n, true);
    const activation = t.attention(q, k, v, heads);
    const y = x.add(t.linear(w, l.name(&key, "{s}attention.output.dense", .{prefix}), activation, true));
    const normalized = t.norm(w, l.name(&key, "{s}layernorm_after", .{prefix}), y, 1e-12);
    const hidden = t.gelu(t.linear(w, l.name(&key, "{s}intermediate.dense", .{prefix}), normalized, true));
    return y.add(t.linear(w, l.name(&key, "{s}output.dense", .{prefix}), hidden, true));
}

fn imageTokens(comptime p: Profile, w: l.Weights, image: Tensor) Tensor {
    const patches = l.conv(w, dino ++ "embeddings.patch_embeddings.projection", image, p.patch, @splat(0), true);
    const grid = patches.dim(2);
    const tokens = patches.transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, grid * grid, p.dino_width });
    const table = w.get(dino ++ "embeddings.position_embeddings", "");
    const spatial = table.slice(1, .{ .start = 1 }).reshape(.{ 1, p.position_grid, p.position_grid, p.dino_width })
        .transpose(.{ 0, 3, 1, 2 });
    const resized = runtime.spatial.bicubicHalfPixel(spatial, grid, grid, 0.1)
        .transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, grid * grid, p.dino_width });
    const cls = w.get(dino ++ "embeddings.cls_token", "").add(table.slice(1, .{ .end = 1 }));
    var x = Tensor.concatenate(&.{ cls, tokens.add(resized) }, 1);
    for (0..p.dino_layers) |i| x = dinoBlock(w, i, x, p.dino_heads);
    return t.norm(w, dino ++ "layernorm", x, 1e-12);
}

fn attention(w: l.Weights, prefix: []const u8, x: Tensor, context: Tensor, heads: i64) Tensor {
    var key: [96]u8 = undefined;
    const q = t.linear(w, l.name(&key, "{s}.to_q", .{prefix}), x, false);
    const k = t.linear(w, l.name(&key, "{s}.to_k", .{prefix}), context, false);
    const v = t.linear(w, l.name(&key, "{s}.to_v", .{prefix}), context, false);
    return t.linear(w, l.name(&key, "{s}.to_out.0", .{prefix}), t.attention(q, k, v, heads), true);
}

fn planeBlock(w: l.Weights, index: usize, x: Tensor, context: Tensor, heads: i64) Tensor {
    var base: [64]u8 = undefined;
    var key: [96]u8 = undefined;
    const prefix = l.name(&base, "backbone.transformer_blocks.{d}", .{index});
    const n1 = t.norm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 1e-5);
    const self = x.add(attention(w, l.name(&key, "{s}.attn1", .{prefix}), n1, n1, heads));
    const n2 = t.norm(w, l.name(&key, "{s}.norm2", .{prefix}), self, 1e-5);
    const cross = self.add(attention(w, l.name(&key, "{s}.attn2", .{prefix}), n2, context, heads));
    const n3 = t.norm(w, l.name(&key, "{s}.norm3", .{prefix}), cross, 1e-5);
    const projected = t.linear(w, l.name(&key, "{s}.ff.net.0.proj", .{prefix}), n3, true);
    const mid = @divExact(projected.dim(2), 2);
    const gated = projected.slice(2, .{ .end = mid }).mul(t.gelu(projected.slice(2, .{ .start = mid })));
    return cross.add(t.linear(w, l.name(&key, "{s}.ff.net.2", .{prefix}), gated, true));
}

pub fn Encoder(comptime p: Profile) type {
    return struct {
        graph: runtime.TensorModel,
        const Self = @This();
        const schema = encoderSchema(p);
        pub const input_count = 3 * p.image * p.image;

        pub fn load(session: *const runtime.Session, path: []const u8) !Self {
            return .{ .graph = try .load(session, &schema, path) };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        pub fn encode(self: *Self, normalized_rgb: []const f32, planes: []f32) !u32 {
            return self.graph.infer(forward, .init(.{ 1, 3, p.image, p.image }, .f32), .init(p.planeShape(), .f32), normalized_rgb, planes);
        }

        fn forward(parameters: []const Tensor, image: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
            const context = imageTokens(p, w, image);
            const tokens = 3 * p.plane * p.plane;
            const embeddings = w.get("tokenizer.embeddings", "").transpose(.{ 1, 0, 2, 3 })
                .reshape(.{ 1, p.width, 1, tokens });
            const normalized = l.groupNorm(w, "backbone.norm", embeddings, p.groups, 1e-6)
                .reshape(.{ 1, p.width, tokens }).transpose(.{ 0, 2, 1 });
            var x = t.linear(w, "backbone.proj_in", normalized, true);
            for (0..p.layers) |i| x = planeBlock(w, i, x, context, p.heads);
            const planes = t.linear(w, "backbone.proj_out", x, true).transpose(.{ 0, 2, 1 })
                .add(embeddings.reshape(.{ 1, p.width, tokens }))
                .reshape(.{ p.width, 3, p.plane, p.plane }).transpose(.{ 1, 0, 2, 3 });
            return t.deconv2x(w, "post_processor.upsample", planes);
        }
    };
}

fn samplePlane(comptime p: Profile, plane: []const f32, uv: [2]f32, features: []f32) void {
    const size: i32 = @intCast(2 * p.plane);
    const plane_area: usize = @intCast(size * size);
    const extent: f32 = @floatFromInt(size);
    const units = extent / (2 * radius);
    const center = (extent - 1) * 0.5;
    const x = std.math.clamp(uv[0] * units + center, -1, extent);
    const y = std.math.clamp(uv[1] * units + center, -1, extent);
    const x0: i32 = @intFromFloat(@floor(x));
    const y0: i32 = @intFromFloat(@floor(y));
    const dx = x - @floor(x);
    const dy = y - @floor(y);
    const weight_x = [2]f32{ 1 - dx, dx };
    const weight_y = [2]f32{ 1 - dy, dy };
    @memset(features, 0);
    // Bilinear grid_sample with align_corners=false and zero padding.
    for (weight_y, 0..) |wy, row| {
        const iy = y0 + @as(i32, @intCast(row));
        if (iy < 0 or iy >= size or wy == 0) continue;
        for (weight_x, 0..) |wx, column| {
            const ix = x0 + @as(i32, @intCast(column));
            if (ix < 0 or ix >= size or wx == 0) continue;
            const pixel: usize = @intCast(iy * size + ix);
            for (features, 0..) |*value, channel| {
                value.* += plane[channel * plane_area + pixel] * wx * wy;
            }
        }
    }
}

pub fn Field(comptime p: Profile) type {
    return struct {
        graph: runtime.TensorModel,
        sampled: []f32,
        const Self = @This();
        const schema = fieldSchema(p);
        const features: usize = @intCast(p.features);
        const features_per_point = 3 * features;
        const plane_size = p.planeCount() / 3;

        pub fn load(session: *const runtime.Session, path: []const u8) !Self {
            const sampled = try session.allocator.alloc(f32, query_batch * features_per_point);
            errdefer session.allocator.free(sampled);
            return .{ .graph = try .load(session, &schema, path), .sampled = sampled };
        }

        pub fn deinit(self: *Self) void {
            self.graph.session.allocator.free(self.sampled);
            self.graph.deinit();
        }

        pub fn query(self: *Self, planes: []const f32, coordinates: []const f32, output: []f32) !u32 {
            if (planes.len != p.planeCount() or coordinates.len % 3 != 0) return error.InvalidBufferSize;
            const count = coordinates.len / 3;
            if (count == 0 or count > query_batch or output.len != count * field_channels) return error.InvalidBufferSize;
            const input = self.sampled[0 .. count * features_per_point];
            for (0..count) |i| {
                const x, const y, const z = coordinates[3 * i ..][0..3].*;
                if (!std.math.isFinite(x) or !std.math.isFinite(y) or !std.math.isFinite(z)) return error.InvalidCoordinate;
                const row = input[i * features_per_point ..][0..features_per_point];
                samplePlane(p, planes[0..plane_size], .{ x, y }, row[0..features]);
                samplePlane(p, planes[plane_size..][0..plane_size], .{ x, z }, row[features..][0..features]);
                samplePlane(p, planes[2 * plane_size ..], .{ y, z }, row[2 * features ..]);
            }
            const rows: i64 = @intCast(count);
            return self.graph.infer(forward, .init(.{ rows, features_per_point }, .f32), .init(.{ rows, field_channels }, .f32), input, output);
        }

        fn forward(parameters: []const Tensor, input: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
            var x = input;
            var key: [32]u8 = undefined;
            for (0..p.decoder_layers) |i| {
                x = t.linear(w, l.name(&key, "decoder.layers.{d}", .{2 * i}), x, true).silu();
            }
            const y = t.linear(w, l.name(&key, "decoder.layers.{d}", .{2 * p.decoder_layers}), x, true);
            const density = y.slice(1, .{ .end = 1 }).subConstant(@as(f32, 1)).exp();
            return Tensor.concatenate(&.{ density, y.slice(1, .{ .start = 1 }).sigmoid() }, 1);
        }
    };
}
