const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = zml.Tensor;

pub const context = 77;
pub const vocabulary = 49_408;
pub const channels = 768;
const heads = 12;
const blocks = 12;
const mlp_width = 3072;

fn param(name: []const u8, dims: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn pair(comptime name: []const u8, dims: anytype, outputs: i64) [2]l.Parameter {
    return .{ param(name ++ ".weight", dims), param(name ++ ".bias", .{outputs}) };
}

const schema = blk: {
    @setEvalBranchQuota(80_000);
    var spec: [4 + blocks * 16]l.Parameter = undefined;
    spec[0..4].* = [_]l.Parameter{
        param("text_model.embeddings.token_embedding.weight", .{ vocabulary, channels }),
        param("text_model.embeddings.position_embedding.weight", .{ context, channels }),
    } ++ pair("text_model.final_layer_norm", .{channels}, channels);
    for (0..blocks) |i| {
        const base = std.fmt.comptimePrint("text_model.encoder.layers.{d}", .{i});
        spec[4 + i * 16 ..][0..16].* =
            pair(base ++ ".self_attn.q_proj", .{ channels, channels }, channels) ++
            pair(base ++ ".self_attn.k_proj", .{ channels, channels }, channels) ++
            pair(base ++ ".self_attn.v_proj", .{ channels, channels }, channels) ++
            pair(base ++ ".self_attn.out_proj", .{ channels, channels }, channels) ++
            pair(base ++ ".layer_norm1", .{channels}, channels) ++
            pair(base ++ ".layer_norm2", .{channels}, channels) ++
            pair(base ++ ".mlp.fc1", .{ mlp_width, channels }, mlp_width) ++
            pair(base ++ ".mlp.fc2", .{ channels, mlp_width }, channels);
    }
    break :blk spec;
};

fn block(w: l.Weights, x: Tensor, i: usize, mask: Tensor) Tensor {
    var base: [64]u8 = undefined;
    var key: [96]u8 = undefined;
    const prefix = l.name(&base, "text_model.encoder.layers.{d}", .{i});
    const norm = t.norm(w, l.name(&key, "{s}.layer_norm1", .{prefix}), x, 1e-5);
    const q = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.q_proj", .{prefix}), norm, true), heads);
    const k = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.k_proj", .{prefix}), norm, true), heads);
    const v = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.v_proj", .{prefix}), norm, true), heads);
    const scores = t.scores(q, k);
    const values = t.attend(scores.add(mask.broadcast(scores.shape(), &.{ 2, 3 })), v);
    const y = x.add(t.linear(w, l.name(&key, "{s}.self_attn.out_proj", .{prefix}), values, true));
    const mlp_input = t.norm(w, l.name(&key, "{s}.layer_norm2", .{prefix}), y, 1e-5);
    const hidden = t.linear(w, l.name(&key, "{s}.mlp.fc1", .{prefix}), mlp_input, true);
    const activated = hidden.mul(hidden.scale(@as(f32, 1.702)).sigmoid());
    return y.add(t.linear(w, l.name(&key, "{s}.mlp.fc2", .{prefix}), activated, true));
}

fn forward(parameters: []const Tensor, tokens: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const len = tokens.dim(0);
    const pair_shape = zml.Shape.init(.{ len, len }, .i32);
    const after = Tensor.iota(pair_shape, 1).cmp(.GT, Tensor.iota(pair_shape, 0));
    const shape = pair_shape.withDtype(.f32);
    const mask = after.select(Tensor.scalar(-std.math.inf(f32), .f32).broad(shape), Tensor.scalar(@as(f32, 0), .f32).broad(shape));
    const word = w.get("text_model.embeddings.token_embedding", ".weight").withTags(.{ .vocabulary, .embedding });
    const embedding = word.gather(.{ .vocabulary = tokens.convert(.i32) }, .{}).reshape(.{ 1, len, channels });
    const position = w.get("text_model.embeddings.position_embedding", ".weight").slice(0, .{ .end = len });
    var x = embedding.add(position.reshape(embedding.shape()));
    for (0..blocks) |i| x = block(w, x, i, mask);
    return t.norm(w, "text_model.final_layer_norm", x, 1e-5);
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn encode(self: *Model, tokens: []const u32, output: []f32) !u32 {
        if (tokens.len == 0 or tokens.len > context) return error.InvalidSequenceLength;
        var input: [context]f32 = undefined;
        for (tokens, input[0..tokens.len]) |id, *slot| {
            if (id >= vocabulary) return error.InvalidToken;
            slot.* = @floatFromInt(id);
        }
        const len: i64 = @intCast(tokens.len);
        return self.graph.infer(forward, .init(.{len}, .f32), .init(.{ 1, len, channels }, .f32), input[0..tokens.len], output);
    }
};
