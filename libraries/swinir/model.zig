const std = @import("std");
const runtime = @import("zml_lib/runtime");
const Shape = @import("zml/core").Shape;
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const spec = @import("weights.zig");

pub const scale = 1;
const c = spec.channels;
const norm_eps = 1e-5;

fn positionIndices(comptime win: usize) [win * win * win * win]i32 {
    @setEvalBranchQuota(30_000);
    const tokens = win * win;
    var index: [tokens * tokens]i32 = undefined;
    for (0..tokens) |q| {
        for (0..tokens) |k| {
            const dy: i32 = @intCast(q / win + win - 1);
            const dx: i32 = @intCast(q % win + win - 1);
            index[q * tokens + k] = (dy - @as(i32, @intCast(k / win))) * @as(i32, @intCast(2 * win - 1)) +
                dx - @as(i32, @intCast(k % win));
        }
    }
    return index;
}

fn region(comptime win: i64, coords: Tensor, length: i64) Tensor {
    const shift = @divTrunc(win, 2);
    const edge_window: i32 = @intCast(length - win);
    const edge_shift: i32 = @intCast(length - shift);
    return coords.cmp(.GE, .scalar(edge_window, .i32)).convert(.i32)
        .add(coords.cmp(.GE, .scalar(edge_shift, .i32)).convert(.i32));
}

fn windowMask(comptime win: i64, h: i64, w: i64) Tensor {
    const tokens = win * win;
    const map_shape = Shape.init(.{ 1, h, w, 1 }, .i32);
    const rows = region(win, Tensor.iota(map_shape, 1), h);
    const cols = region(win, Tensor.iota(map_shape, 2), w);
    const windows = @divExact(h * w, tokens);
    const labels = t.windowPartition(rows.scale(@as(i32, 3)).add(cols).convert(.f32), win).reshape(.{ windows, tokens });
    const pairs = Shape.init(.{ windows, tokens, tokens }, .f32);
    return labels.broadcast(pairs, &.{ 0, 1 }).cmp(.NE, labels.broadcast(pairs, &.{ 0, 2 }))
        .convert(.f32).scale(@as(f32, -100));
}

fn rotate(x: Tensor, comptime axis: u3, start: i64) Tensor {
    return Tensor.concatenate(&.{ x.slice(axis, .{ .start = start }), x.slice(axis, .{ .end = start }) }, axis);
}

fn attention(w: l.Weights, prefix: []const u8, x: Tensor, mask: ?Tensor, index: Tensor) Tensor {
    var key: [96]u8 = undefined;
    const qkv = t.linear(w, l.name(&key, "{s}.attn.qkv", .{prefix}), x, true)
        .reshape(.{ x.dim(0), x.dim(1), 3, c });
    const q = t.splitHeads(qkv.slice(2, .single(0)), spec.heads);
    const k = t.splitHeads(qkv.slice(2, .single(1)), spec.heads);
    const v = t.splitHeads(qkv.slice(2, .single(2)), spec.heads);
    const scores = t.scores(q, k);
    const table = w.get(l.name(&key, "{s}.attn", .{prefix}), ".relative_position_bias_table");
    const position = table.withTags(.{ .relative, .head }).gather(.{ .relative = index }, .{}).transpose(.{ 2, 0, 1 });
    var logits = scores.add(position.broadcast(scores.shape(), &.{ 1, 2, 3 }));
    if (mask) |value| logits = logits.add(value.broadcast(scores.shape(), &.{ 0, 2, 3 }));
    return t.linear(w, l.name(&key, "{s}.attn.proj", .{prefix}), t.attend(logits, v), true);
}

fn block(comptime win: i64, w: l.Weights, prefix: []const u8, x: Tensor, mask: ?Tensor, index: Tensor) Tensor {
    const shift = @divTrunc(win, 2);
    const tokens = win * win;
    var key: [96]u8 = undefined;
    const h = x.dim(1);
    const width = x.dim(2);
    const normalized = t.norm(w, l.name(&key, "{s}.norm1", .{prefix}), x, norm_eps);
    const shifted = if (mask != null) rotate(rotate(normalized, 1, shift), 2, shift) else normalized;
    const windows = t.windowPartition(shifted, win).reshape(.{ @divExact(h * width, tokens), tokens, c });
    const attended = attention(w, prefix, windows, mask, index).reshape(.{ windows.dim(0), win, win, c });
    const tiled = t.windowUnpartition(attended, h, width);
    const y = x.add(if (mask != null) rotate(rotate(tiled, 1, h - shift), 2, width - shift) else tiled);
    const normalized_y = t.norm(w, l.name(&key, "{s}.norm2", .{prefix}), y, norm_eps);
    const hidden = t.gelu(t.linear(w, l.name(&key, "{s}.mlp.fc1", .{prefix}), normalized_y, true));
    return y.add(t.linear(w, l.name(&key, "{s}.mlp.fc2", .{prefix}), hidden, true));
}

fn padding(comptime win: i64, dim: i64) i64 {
    return @mod(win - @mod(dim, win), win);
}

fn forward(comptime win: i64, comptime image_range: f32, comptime schema: []const l.Parameter, parameters: []const Tensor, image: Tensor) Tensor {
    const tokens = win * win;
    const position_indices = comptime positionIndices(@intCast(win));
    const w: l.Weights = .{ .schema = schema, .tensors = parameters };
    const source_h = image.dim(2);
    const source_w = image.dim(3);
    const input = runtime.spatial.reflectEnd(runtime.spatial.reflectEnd(image, 2, padding(win, source_h)), 3, padding(win, source_w));
    const mean = Tensor.constantTensor(.init(.{3}, .f32), std.mem.asBytes(&[_]f32{ 0.4488, 0.4371, 0.4040 }));
    const mean_image = mean.broadcast(input.shape(), &.{1});
    const normalized = input.sub(mean_image).scale(image_range);
    const shallow = l.conv(w, "conv_first", normalized, 1, @splat(1), true);
    var x = t.norm(w, "patch_embed.norm", shallow.transpose(.{ 0, 2, 3, 1 }), norm_eps);
    const mask = windowMask(win, input.dim(2), input.dim(3));
    const index = Tensor.constantTensor(.init(.{ tokens, tokens }, .i32), std.mem.asBytes(&position_indices));
    var key: [96]u8 = undefined;
    for (0..spec.stages) |i| {
        const residual = x;
        for (0..spec.blocks_per_stage) |j| {
            x = block(win, w, l.name(&key, "layers.{d}.residual_group.blocks.{d}", .{ i, j }), x, if (j % 2 == 1) mask else null, index);
        }
        const y = l.conv(w, l.name(&key, "layers.{d}.conv", .{i}), x.transpose(.{ 0, 3, 1, 2 }), 1, @splat(1), true);
        x = y.transpose(.{ 0, 2, 3, 1 }).add(residual);
    }
    const deep = t.norm(w, "norm", x, norm_eps).transpose(.{ 0, 3, 1, 2 });
    const residual = l.conv(w, "conv_after_body", deep, 1, @splat(1), true).add(shallow);
    return normalized.add(l.conv(w, "conv_last", residual, 1, @splat(1), true)).divByConst(image_range)
        .add(mean_image).slice(2, .{ .end = source_h }).slice(3, .{ .end = source_w });
}

pub const Model = ModelFor(7, 255);
pub const ColorDenoise = struct {
    pub const scale = 1;
    pub const Model = ModelFor(8, 1);
};

fn ModelFor(comptime win: i64, comptime image_range: f32) type {
    return struct {
        const Self = @This();
        const schema = spec.schemaFor(win);
        graph: runtime.TensorModel,

        pub fn load(session: *const runtime.Session, path: []const u8) !Self {
            return .{ .graph = try .load(session, &schema, path) };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        fn run(parameters: []const Tensor, image: Tensor) Tensor {
            return forward(win, image_range, &schema, parameters, image);
        }

        pub fn infer(self: *Self, h: u32, w: u32, input: []const f32, output: []f32) !u32 {
            if (h < win or w < win) return error.InvalidDimensions;
            const shape = Shape.init(.{ 1, 3, @as(i64, h), @as(i64, w) }, .f32);
            return self.graph.infer(run, shape, shape, input, output);
        }
    };
}
