const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = zml.Tensor;

pub const vocabulary = 151_936;
pub const Profile = struct {
    vocab: i64 = vocabulary,
    layers: usize,
    width: i64,
    mlp: i64,
    heads: i64,
    kv_heads: i64,
    head_width: i64,
    has_bias: bool,
    head_norm: bool,
    separate_output: bool,
    scale_attention_after_dot: bool = false,
    eos_ids: []const u32,

    fn tensorCount(p: Profile) usize {
        return 2 + @as(usize, @intFromBool(p.separate_output)) +
            p.layers * (9 + 3 * @as(usize, @intFromBool(p.has_bias)) + 2 * @as(usize, @intFromBool(p.head_norm)));
    }
};

pub const qwen2_05b: Profile = .{
    .layers = 24,
    .width = 896,
    .mlp = 4864,
    .heads = 14,
    .kv_heads = 2,
    .head_width = 64,
    .has_bias = true,
    .head_norm = false,
    .separate_output = false,
    .eos_ids = &.{151645},
};
pub const qwen3_17b: Profile = .{
    .layers = 28,
    .width = 2048,
    .mlp = 6144,
    .heads = 16,
    .kv_heads = 8,
    .head_width = 128,
    .has_bias = false,
    .head_norm = true,
    .separate_output = true,
    .eos_ids = &.{ 151645, 151643 },
};

fn parameter(name: []const u8, dims: anytype, dtype: zml.DataType) l.Parameter {
    return .{ .name = name, .shape = .init(dims, dtype) };
}

pub fn schema(comptime p: Profile, comptime dtype: zml.DataType) [p.tensorCount()]l.Parameter {
    @setEvalBranchQuota(100_000);
    var result: [p.tensorCount()]l.Parameter = undefined;
    result[0] = parameter("model.embed_tokens.weight", .{ p.vocab, p.width }, dtype);
    result[1] = parameter("model.norm.weight", .{p.width}, dtype);
    var offset: usize = 2;
    if (p.separate_output) {
        result[offset] = parameter("lm_head.weight", .{ p.vocab, p.width }, dtype);
        offset += 1;
    }
    for (0..p.layers) |i| {
        const base = std.fmt.comptimePrint("model.layers.{d}", .{i});
        const attn = base ++ ".self_attn";
        result[offset..][0..9].* = .{
            parameter(base ++ ".input_layernorm.weight", .{p.width}, dtype),
            parameter(base ++ ".post_attention_layernorm.weight", .{p.width}, dtype),
            parameter(attn ++ ".q_proj.weight", .{ p.heads * p.head_width, p.width }, dtype),
            parameter(attn ++ ".k_proj.weight", .{ p.kv_heads * p.head_width, p.width }, dtype),
            parameter(attn ++ ".v_proj.weight", .{ p.kv_heads * p.head_width, p.width }, dtype),
            parameter(attn ++ ".o_proj.weight", .{ p.width, p.heads * p.head_width }, dtype),
            parameter(base ++ ".mlp.gate_proj.weight", .{ p.mlp, p.width }, dtype),
            parameter(base ++ ".mlp.up_proj.weight", .{ p.mlp, p.width }, dtype),
            parameter(base ++ ".mlp.down_proj.weight", .{ p.width, p.mlp }, dtype),
        };
        offset += 9;
        if (p.has_bias) {
            result[offset..][0..3].* = .{
                parameter(attn ++ ".q_proj.bias", .{p.heads * p.head_width}, dtype),
                parameter(attn ++ ".k_proj.bias", .{p.kv_heads * p.head_width}, dtype),
                parameter(attn ++ ".v_proj.bias", .{p.kv_heads * p.head_width}, dtype),
            };
            offset += 3;
        }
        if (p.head_norm) {
            result[offset..][0..2].* = .{
                parameter(attn ++ ".q_norm.weight", .{p.head_width}, dtype),
                parameter(attn ++ ".k_norm.weight", .{p.head_width}, dtype),
            };
            offset += 2;
        }
    }
    return result;
}

pub fn rms(x: Tensor, gain: Tensor) Tensor {
    const axis = x.rank() - 1;
    const inv = x.mul(x).mean(axis).addConstant(@as(f32, 1e-6)).rsqrt();
    return x.mul(inv.broad(x.shape())).mul(gain.convert(.f32).broadcast(x.shape(), &.{axis}));
}

pub fn linear(w: l.Weights, name: []const u8, x: Tensor, has_bias: bool) Tensor {
    const bias = if (has_bias) w.get(name, ".bias").convert(.f32) else null;
    return t.linearParameters(x, w.get(name, ".weight").convert(.f32), bias);
}

pub fn tokenEmbeddings(table: Tensor, ids: Tensor) Tensor {
    return table.withTags(.{ .vocabulary, .embedding }).gather(.{ .vocabulary = ids }, .{})
        .convert(.f32).reshape(.{ 1, ids.dim(0), table.dim(1) });
}

pub fn rotate(x: Tensor, sin: Tensor, cos: Tensor) Tensor {
    const half = @divExact(x.dim(3), 2);
    const rotated = Tensor.concatenate(&.{
        x.slice(3, .{ .start = half }).scale(@as(f32, -1)), x.slice(3, .{ .end = half }),
    }, 3);
    return x.mul(cos.broadcast(x.shape(), &.{ 2, 3 })).add(rotated.mul(sin.broadcast(x.shape(), &.{ 2, 3 })));
}

pub fn scoresAfterDot(q: Tensor, k: Tensor) Tensor {
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(q.dim(3))));
    return q.dotGeneral(k, &.{.{ 3, 3 }}, &.{ .{ 0, 0 }, .{ 1, 1 } }).scale(scale);
}

fn expandKV(comptime p: Profile, x: Tensor) Tensor {
    const group = @divExact(p.heads, p.kv_heads);
    return x.broadcast(.init(.{ 1, p.kv_heads, group, x.dim(2), p.head_width }, .f32), &.{ 0, 1, 3, 4 })
        .reshape(.{ 1, p.heads, x.dim(2), p.head_width });
}

pub fn layersForward(comptime p: Profile, w: l.Weights, tokens: Tensor, mask: Tensor, sin: Tensor, cos: Tensor) Tensor {
    var x = tokens;
    for (0..p.layers) |i| {
        var base: [48]u8 = undefined;
        var key: [80]u8 = undefined;
        const prefix = l.name(&base, "model.layers.{d}", .{i});
        const y = rms(x, w.get(l.name(&key, "{s}.input_layernorm", .{prefix}), ".weight"));
        var q = t.splitHeads(linear(w, l.name(&key, "{s}.self_attn.q_proj", .{prefix}), y, p.has_bias), p.heads);
        var k = t.splitHeads(linear(w, l.name(&key, "{s}.self_attn.k_proj", .{prefix}), y, p.has_bias), p.kv_heads);
        const v = t.splitHeads(linear(w, l.name(&key, "{s}.self_attn.v_proj", .{prefix}), y, p.has_bias), p.kv_heads);
        if (p.head_norm) {
            q = rms(q, w.get(l.name(&key, "{s}.self_attn.q_norm", .{prefix}), ".weight"));
            k = rms(k, w.get(l.name(&key, "{s}.self_attn.k_norm", .{prefix}), ".weight"));
        }
        const rotated_q = rotate(q, sin, cos);
        const expanded_k = expandKV(p, rotate(k, sin, cos));
        const logits = if (p.scale_attention_after_dot)
            scoresAfterDot(rotated_q, expanded_k)
        else
            t.scores(rotated_q, expanded_k);
        const a = t.attend(logits.add(mask.broadcast(logits.shape(), &.{ 2, 3 })), expandKV(p, v));
        x = x.add(linear(w, l.name(&key, "{s}.self_attn.o_proj", .{prefix}), a, false));
        const z = rms(x, w.get(l.name(&key, "{s}.post_attention_layernorm", .{prefix}), ".weight"));
        const gate = linear(w, l.name(&key, "{s}.mlp.gate_proj", .{prefix}), z, false);
        const up = linear(w, l.name(&key, "{s}.mlp.up_proj", .{prefix}), z, false);
        const fused = gate.mul(gate.sigmoid()).mul(up);
        x = x.add(linear(w, l.name(&key, "{s}.mlp.down_proj", .{prefix}), fused, false));
    }
    return rms(x, w.get("model.norm", ".weight"));
}

pub fn Model(comptime p: Profile) type {
    return struct {
        graph: runtime.TensorModel,
        const Self = @This();
        const params = schema(p, .f32);

        pub fn load(session: *const runtime.Session, path: []const u8) !Self {
            return .{ .graph = try .load(session, &params, path) };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        pub fn logits(self: *Self, token_ids: []const u32, context: usize, output: []f32) !u32 {
            if (token_ids.len == 0 or token_ids.len > context or context > 256) return error.InvalidContext;
            var encoded: [257]f32 = @splat(0);
            for (token_ids, encoded[0..token_ids.len]) |id, *slot| {
                if (id >= p.vocab) return error.InvalidToken;
                slot.* = @floatFromInt(id);
            }
            encoded[context] = @floatFromInt(token_ids.len - 1);
            return self.graph.infer(forward, .init(.{@as(i64, @intCast(context + 1))}, .f32), .init(.{p.vocab}, .f32), encoded[0 .. context + 1], output);
        }

        fn forward(parameters: []const Tensor, input: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &params, .tensors = parameters };
            const len = input.dim(0) - 1;
            const embedding = w.get("model.embed_tokens", ".weight");
            const ids = input.slice(0, .{ .end = len }).convert(.i32);
            const tokens = tokenEmbeddings(embedding, ids);
            const shape = zml.Shape.init(.{ len, len }, .i32);
            const after = Tensor.iota(shape, 1).cmp(.GT, Tensor.iota(shape, 0));
            const mask_shape = shape.withDtype(.f32);
            const mask = after.select(Tensor.scalar(-std.math.inf(f32), .f32).broad(mask_shape), Tensor.scalar(@as(f32, 0), .f32).broad(mask_shape));
            const rope_shape = zml.Shape.init(.{ len, @divExact(p.head_width, 2) }, .i32);
            const phase = Tensor.iota(rope_shape, 1).convert(.f32).scale(-2 / @as(f32, @floatFromInt(p.head_width)));
            const frequency = Tensor.scalar(@as(f32, 1_000_000), .f32).broad(rope_shape.withDtype(.f32)).pow(phase);
            const angles = Tensor.iota(rope_shape, 0).convert(.f32).mul(frequency);
            const doubled = Tensor.concatenate(&.{ angles, angles }, 1);
            const activations = layersForward(p, w, tokens, mask, doubled.sin(), doubled.cos());
            const end = input.slice(0, .{ .start = len }).convert(.i32);
            const last = activations.withTags(.{ .batch, .token, .embedding }).gather(.{ .token = end }, .{}).reshape(.{ 1, p.width });
            return t.linearParameters(last, if (p.separate_output) w.get("lm_head", ".weight") else embedding, null).flatten();
        }
    };
}

pub fn run(comptime p: Profile, init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 6) return error.ExpectedBackendWeightsContextNewTokensAndPromptIds;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const context = try std.fmt.parseInt(usize, args[3], 10);
    const limit = try std.fmt.parseInt(usize, args[4], 10);
    if (context > 256 or args.len - 5 >= context or limit == 0) return error.InvalidContext;
    var tokens: [256]u32 = undefined;
    var len = args.len - 5;
    for (args[5..], tokens[0..len]) |text, *id| id.* = try std.fmt.parseInt(u32, text, 10);
    const scores = try allocator.alloc(f32, @intCast(p.vocab));
    defer allocator.free(scores);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try Model(p).load(&session, args[2]);
    defer model.deinit();
    var buffer: [512]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    for (0..@min(limit, context - len)) |_| {
        runtime.probe_io.placement(&session, try model.logits(tokens[0..len], context, scores));
        var best: usize = 0;
        for (scores[1..], 1..) |score, i| {
            if (score > scores[best]) best = i;
        }
        try stdout.interface.print("{d}\n", .{best});
        const token: u32 = @intCast(best);
        if (std.mem.indexOfScalar(u32, p.eos_ids, token) != null) break;
        tokens[len] = token;
        len += 1;
    }
    try stdout.interface.flush();
}
