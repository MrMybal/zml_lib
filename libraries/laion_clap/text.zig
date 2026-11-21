const runtime = @import("zml_lib/runtime");
const std = @import("std");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const vocabulary = 50_265;
const positions = 514;
const width = 768;
const epsilon: f32 = 1e-12;
const embeddings = "text_model.embeddings.";
pub const context_limit = positions - 2;
pub const embedding_size = 512;

fn parameter(name: []const u8, shape: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(shape, .f32) };
}

fn dense(comptime name: []const u8, input: i64, output: i64) [2]l.Parameter {
    return .{ parameter(name ++ ".weight", .{ output, input }), parameter(name ++ ".bias", .{output}) };
}

fn norm(comptime name: []const u8) [2]l.Parameter {
    return .{ parameter(name ++ ".weight", .{width}), parameter(name ++ ".bias", .{width}) };
}

const schema = blk: {
    @setEvalBranchQuota(75_000);
    const common = [_]l.Parameter{
        parameter(embeddings ++ "word_embeddings.weight", .{ vocabulary, width }),
        parameter(embeddings ++ "position_embeddings.weight", .{ positions, width }),
        parameter(embeddings ++ "token_type_embeddings.weight", .{ 1, width }),
    } ++ norm(embeddings ++ "LayerNorm") ++
        dense("text_model.pooler.dense", width, width) ++
        dense("text_projection.linear1", width, embedding_size) ++
        dense("text_projection.linear2", embedding_size, embedding_size);
    var entries: [common.len + 12 * 16]l.Parameter = undefined;
    entries[0..common.len].* = common;
    for (0..12) |i| {
        const base = std.fmt.comptimePrint("text_model.encoder.layer.{d}.", .{i});
        entries[common.len + i * 16 ..][0..16].* =
            dense(base ++ "attention.self.query", width, width) ++
            dense(base ++ "attention.self.key", width, width) ++
            dense(base ++ "attention.self.value", width, width) ++
            dense(base ++ "attention.output.dense", width, width) ++
            norm(base ++ "attention.output.LayerNorm") ++
            dense(base ++ "intermediate.dense", width, 3072) ++
            dense(base ++ "output.dense", 3072, width) ++
            norm(base ++ "output.LayerNorm");
    }
    break :blk entries;
};

fn encodeLayer(w: l.Weights, index: usize, input: Tensor, masked: Tensor) Tensor {
    var prefix_buffer: [44]u8 = undefined;
    const prefix = l.name(&prefix_buffer, "text_model.encoder.layer.{d}.", .{index});
    var key: [80]u8 = undefined;
    const q = t.splitHeads(t.linear(w, l.name(&key, "{s}attention.self.query", .{prefix}), input, true), 12);
    const k = t.splitHeads(t.linear(w, l.name(&key, "{s}attention.self.key", .{prefix}), input, true), 12);
    const v = t.splitHeads(t.linear(w, l.name(&key, "{s}attention.self.value", .{prefix}), input, true), 12);
    const scores = t.scores(q, k);
    const attend = t.attend(scores.add(masked.broadcast(scores.shape(), &.{3})), v);
    const projection = t.linear(w, l.name(&key, "{s}attention.output.dense", .{prefix}), attend, true);
    const x = t.norm(w, l.name(&key, "{s}attention.output.LayerNorm", .{prefix}), input.add(projection), epsilon);
    const intermediate = t.gelu(t.linear(w, l.name(&key, "{s}intermediate.dense", .{prefix}), x, true));
    const output = t.linear(w, l.name(&key, "{s}output.dense", .{prefix}), intermediate, true);
    return t.norm(w, l.name(&key, "{s}output.LayerNorm", .{prefix}), x.add(output), epsilon);
}

fn row(input: Tensor, offset: i64) Tensor {
    return input.slice(0, .{ .start = offset, .end = offset + 1 }).flatten();
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const len = input.dim(1);
    const ids = row(input, 0).convert(.i32);
    const position_ids = row(input, 1).convert(.i32);
    const mask_input = row(input, 2);
    const mask = mask_input.cmp(.EQ, Tensor.scalar(@as(f32, 0), .f32).broad(mask_input.shape()))
        .select(Tensor.scalar(-std.math.inf(f32), .f32).broad(mask_input.shape()), Tensor.scalar(@as(f32, 0), .f32).broad(mask_input.shape()));
    const words = w.get(embeddings ++ "word_embeddings", ".weight").withTags(.{ .index, .channel }).gather(.{ .index = ids }, .{});
    const position = w.get(embeddings ++ "position_embeddings", ".weight").withTags(.{ .index, .channel }).gather(.{ .index = position_ids }, .{});
    const types = w.get(embeddings ++ "token_type_embeddings", ".weight").broadcast(words.shape(), &.{ 0, 1 });
    const tokens = words.add(types).add(position).reshape(.{ 1, len, width });
    var x = t.norm(w, embeddings ++ "LayerNorm", tokens, epsilon);
    for (0..12) |i| x = encodeLayer(w, i, x, mask);
    const first = x.slice(1, .{ .end = 1 }).reshape(.{ 1, width });
    const pooled = t.linear(w, "text_model.pooler.dense", first, true).tanh();
    const hidden = t.linear(w, "text_projection.linear1", pooled, true);
    const projected = t.linear(w, "text_projection.linear2", hidden.maximum(Tensor.scalar(@as(f32, 0), .f32).broad(hidden.shape())), true);
    const magnitude = projected.mul(projected).sum(1).sqrt();
    return projected.div(magnitude.maximum(Tensor.scalar(epsilon, .f32).broad(magnitude.shape())).broad(projected.shape())).flatten();
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn encode(self: *Model, ids: []const u32, mask: []const u8, output: []f32) !u32 {
        const len = ids.len;
        if (len == 0 or len > context_limit or mask.len != len) return error.InvalidTokenCount;
        var encoded: [3 * context_limit]f32 = undefined;
        var position: u32 = 1;
        var active = false;
        for (ids, mask, 0..) |id, present, i| {
            if (id >= vocabulary or present > 1) return error.InvalidTokens;
            if (id != 1) position += 1;
            encoded[i] = @floatFromInt(id);
            encoded[len + i] = @floatFromInt(if (id == 1) 1 else position);
            encoded[2 * len + i] = @floatFromInt(present);
            active = active or present != 0;
        }
        if (!active) return error.InvalidAttentionMask;
        return self.graph.infer(
            forward,
            .init(.{ 3, @as(i64, @intCast(len)) }, .f32),
            .init(.{embedding_size}, .f32),
            encoded[0 .. 3 * len],
            output,
        );
    }
};
