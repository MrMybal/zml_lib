const std = @import("std");
const runtime = @import("zml_lib/runtime");
const zml = @import("zml/core");
const Shape = zml.Shape;
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;

pub const max_tokens = 512;
const buckets = 32;
const max_distance = 128;
const eps: f32 = 1e-6;

pub const Profile = enum {
    t5_xxl,
    umt5_xxl,
    t5_fixture,
    umt5_fixture,

    pub fn config(self: Profile) Config {
        return switch (self) {
            .t5_xxl => .{ .width = 4096, .ff = 10_240, .heads = 64, .key_width = 64, .blocks = 24, .vocab = 32_128 },
            .umt5_xxl => .{ .width = 4096, .ff = 10_240, .heads = 64, .key_width = 64, .blocks = 24, .vocab = 256_384, .each_bias = true },
            .t5_fixture => .{ .width = 32, .ff = 64, .heads = 4, .key_width = 8, .blocks = 2, .vocab = 80 },
            .umt5_fixture => .{ .width = 32, .ff = 64, .heads = 4, .key_width = 8, .blocks = 2, .vocab = 80, .each_bias = true },
        };
    }
};

const Config = struct {
    width: i64,
    ff: i64,
    heads: i64,
    key_width: i64,
    blocks: usize,
    vocab: u32,
    each_bias: bool = false,
};

fn param(name: []const u8, dims: anytype, dtype: zml.DataType) l.Parameter {
    return .{ .name = name, .shape = .init(dims, dtype) };
}

fn parameterCount(cfg: Config) usize {
    return 2 + cfg.blocks * 9 + if (cfg.each_bias) cfg.blocks else @as(usize, 1);
}

fn parameters(comptime cfg: Config, dtype: zml.DataType) [parameterCount(cfg)]l.Parameter {
    @setEvalBranchQuota(120_000);
    var schema: [parameterCount(cfg)]l.Parameter = undefined;
    schema[0] = param("shared.weight", .{ cfg.vocab, cfg.width }, dtype);
    schema[1] = param("encoder.final_layer_norm.weight", .{cfg.width}, dtype);
    const projected = cfg.heads * cfg.key_width;
    var offset: usize = 2;
    for (0..cfg.blocks) |i| {
        const base = std.fmt.comptimePrint("encoder.block.{d}.layer.", .{i});
        const attn = base ++ "0.SelfAttention.";
        schema[offset..][0..9].* = .{
            param(attn ++ "q.weight", .{ projected, cfg.width }, dtype),
            param(attn ++ "k.weight", .{ projected, cfg.width }, dtype),
            param(attn ++ "v.weight", .{ projected, cfg.width }, dtype),
            param(attn ++ "o.weight", .{ cfg.width, projected }, dtype),
            param(base ++ "0.layer_norm.weight", .{cfg.width}, dtype),
            param(base ++ "1.layer_norm.weight", .{cfg.width}, dtype),
            param(base ++ "1.DenseReluDense.wi_0.weight", .{ cfg.ff, cfg.width }, dtype),
            param(base ++ "1.DenseReluDense.wi_1.weight", .{ cfg.ff, cfg.width }, dtype),
            param(base ++ "1.DenseReluDense.wo.weight", .{ cfg.width, cfg.ff }, dtype),
        };
        offset += 9;
        if (cfg.each_bias or i == 0) {
            schema[offset] = param(attn ++ "relative_attention_bias.weight", .{ buckets, cfg.heads }, dtype);
            offset += 1;
        }
    }
    return schema;
}

fn relativeBuckets(length: i64) Tensor {
    const shape: Shape = .init(.{ length, length }, .i32);
    const delta = Tensor.iota(shape, 1).sub(Tensor.iota(shape, 0));
    const offset = delta.cmp(.GT, .scalar(0, .i32)).convert(.i32).scale(@as(i32, buckets / 2));
    const distance = delta.abs();
    const exact = buckets / 4;
    const near = distance.cmp(.LT, .scalar(exact, .i32));
    const large = distance.maximum(.scalar(exact, .i32)).convert(.f32)
        .divByConst(@as(f32, exact)).log()
        .scale(@as(f32, buckets / 2 - exact) / @log(@as(f32, max_distance / exact)))
        .convert(.i32).addConstant(@as(i32, exact))
        .minimum(.scalar(buckets / 2 - 1, .i32));
    return offset.add(near.select(distance, large));
}

fn rms(x: Tensor, weight: Tensor) Tensor {
    const variance = x.mul(x).mean(2).addConstant(eps).rsqrt().broad(x.shape());
    return x.mul(variance).mul(weight.convert(.f32).broadcast(x.shape(), &.{2}));
}

fn linear(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    return t.linearParameters(x, w.get(name, ".weight").convert(.f32), null);
}

fn gelu(x: Tensor) Tensor {
    const gate = x.mul(x).mul(x).scale(@as(f32, 0.044715)).add(x)
        .scale(@as(f32, 0.7978845608028654)).tanh().addConstant(@as(f32, 1));
    return x.mul(gate).scale(@as(f32, 0.5));
}

pub fn Encoder(comptime profile: Profile) type {
    return EncoderWithStorage(profile, .f32);
}

pub fn EncoderWithStorage(comptime profile: Profile, comptime storage: runtime.checkpoint.Storage) type {
    const cfg = profile.config();
    return struct {
        const Self = @This();
        pub const schema = parameters(cfg, if (storage == .f16) .f16 else .f32);
        pub const width = cfg.width;
        graph: runtime.TensorModel,

        pub fn load(session: *const runtime.Session, path: []const u8) !Self {
            return .{ .graph = try .loadCheckpoint(session, &schema, path, storage) };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        fn forward(weights: []const Tensor, input: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = weights };
            const length = input.dim(1);
            const ids = input.slice(0, .single(0)).convert(.i32);
            const mask = input.slice(0, .single(1));
            const bias_shape: Shape = .init(.{length}, .f32);
            const hidden_positions = mask.cmp(.EQ, .scalar(0, .f32));
            const padding = hidden_positions.select(
                Tensor.scalar(-std.math.inf(f32), .f32).broad(bias_shape),
                Tensor.scalar(0, .f32).broad(bias_shape),
            );
            const relative = relativeBuckets(length);
            const embedding = w.get("shared", ".weight").withTags(.{ .vocab, .width });
            var x = embedding.gather(.{ .vocab = ids }, .{}).convert(.f32).reshape(.{ 1, length, cfg.width });
            var bias: Tensor = undefined;
            var prefix: [96]u8 = undefined;
            var key: [120]u8 = undefined;
            for (0..cfg.blocks) |i| {
                const name = l.name(&prefix, "encoder.block.{d}.layer", .{i});
                if (i == 0 or cfg.each_bias) {
                    const table = w.get(l.name(&key, "{s}.0.SelfAttention", .{name}), ".relative_attention_bias.weight");
                    bias = table.withTags(.{ .bucket, .head }).gather(.{ .bucket = relative }, .{})
                        .convert(.f32).transpose(.{ 2, 0, 1 });
                }
                const norm = rms(x, w.get(l.name(&key, "{s}.0.layer_norm", .{name}), ".weight"));
                const q = t.splitHeads(linear(w, l.name(&key, "{s}.0.SelfAttention.q", .{name}), norm), cfg.heads);
                const k = t.splitHeads(linear(w, l.name(&key, "{s}.0.SelfAttention.k", .{name}), norm), cfg.heads);
                const v = t.splitHeads(linear(w, l.name(&key, "{s}.0.SelfAttention.v", .{name}), norm), cfg.heads);
                const scores = q.dotGeneral(k, &.{.{ 3, 3 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
                const logits = scores.add(bias.broadcast(scores.shape(), &.{ 1, 2, 3 }))
                    .add(padding.broadcast(scores.shape(), &.{3}));
                const values = t.attend(logits, v);
                x = x.add(linear(w, l.name(&key, "{s}.0.SelfAttention.o", .{name}), values));
                const ff = rms(x, w.get(l.name(&key, "{s}.1.layer_norm", .{name}), ".weight"));
                const gate = linear(w, l.name(&key, "{s}.1.DenseReluDense.wi_0", .{name}), ff);
                const value = linear(w, l.name(&key, "{s}.1.DenseReluDense.wi_1", .{name}), ff);
                x = x.add(linear(w, l.name(&key, "{s}.1.DenseReluDense.wo", .{name}), gelu(gate).mul(value)));
            }
            return rms(x, w.get("encoder.final_layer_norm", ".weight"));
        }

        pub fn encode(self: *Self, tokens: usize, input: []const f32, output: []f32) !u32 {
            if (tokens == 0 or tokens > max_tokens or input.len != 2 * tokens) return error.InvalidTokenBuffer;
            var visible = false;
            for (input[0..tokens], input[tokens..]) |id, mask| {
                if (!std.math.isFinite(id) or id < 0 or id >= @as(f32, @floatFromInt(cfg.vocab)) or @trunc(id) != id)
                    return error.InvalidTokenId;
                if (mask != 0 and mask != 1) return error.InvalidAttentionMask;
                visible = visible or mask == 1;
            }
            if (!visible) return error.InvalidAttentionMask;
            return self.graph.infer(forward, .init(.{ 2, tokens }, .f32), .init(.{ 1, tokens, cfg.width }, .f32), input, output);
        }
    };
}
