const std = @import("std");
const runtime = @import("zml_lib/runtime");
const Shape = @import("zml/core").Shape;
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;

pub const Stage = enum { encoder, decoder };
pub const Profile = enum {
    fixture,
    tiny,
    large_v3_turbo,

    pub fn config(profile: Profile) Config {
        return switch (profile) {
            .fixture => .{ .width = 32, .ff = 64, .heads = 4, .encoder_layers = 2, .decoder_layers = 2, .mels = 8, .source = 16, .target = 24, .vocab = 96 },
            .tiny => .{ .width = 384, .ff = 1536, .heads = 6, .encoder_layers = 4, .decoder_layers = 4, .mels = 80, .source = 1500, .target = 448, .vocab = 51_865 },
            .large_v3_turbo => .{ .width = 1280, .ff = 5120, .heads = 20, .encoder_layers = 32, .decoder_layers = 4, .mels = 128, .source = 1500, .target = 448, .vocab = 51_866 },
        };
    }
};

pub const Config = struct {
    width: usize,
    ff: usize,
    heads: i64,
    encoder_layers: usize,
    decoder_layers: usize,
    mels: usize,
    source: usize,
    target: usize,
    vocab: usize,
};

fn pair(comptime prefix: []const u8, dims: anytype, width: usize) [2]l.Parameter {
    return .{
        .{ .name = prefix ++ ".weight", .shape = .init(dims, .f32) },
        .{ .name = prefix ++ ".bias", .shape = .init(.{width}, .f32) },
    };
}

fn attentionSchema(comptime prefix: []const u8, width: usize) [7]l.Parameter {
    return pair(prefix ++ ".q_proj", .{ width, width }, width) ++
        [_]l.Parameter{.{ .name = prefix ++ ".k_proj.weight", .shape = .init(.{ width, width }, .f32) }} ++
        pair(prefix ++ ".v_proj", .{ width, width }, width) ++ pair(prefix ++ ".out_proj", .{ width, width }, width);
}

fn parameters(comptime cfg: Config, comptime stage: Stage) [
    if (stage == .encoder)
        7 + 15 * cfg.encoder_layers
    else
        4 + 24 * cfg.decoder_layers
]l.Parameter {
    @setEvalBranchQuota(140_000);
    const width = cfg.width;
    const prefix = "model." ++ @tagName(stage);
    const base = if (stage == .encoder)
        pair(prefix ++ ".conv1", .{ width, cfg.mels, 3 }, width) ++ pair(prefix ++ ".conv2", .{ width, width, 3 }, width) ++
            [_]l.Parameter{.{ .name = prefix ++ ".embed_positions.weight", .shape = .init(.{ cfg.source, width }, .f32) }}
    else
        [_]l.Parameter{
            .{ .name = prefix ++ ".embed_tokens.weight", .shape = .init(.{ cfg.vocab, width }, .f32) },
            .{ .name = prefix ++ ".embed_positions.weight", .shape = .init(.{ cfg.target, width }, .f32) },
        };
    const layer_count = if (stage == .encoder) cfg.encoder_layers else cfg.decoder_layers;
    const items = if (stage == .encoder) 15 else 24;
    var schema: [base.len + 2 + items * layer_count]l.Parameter = undefined;
    schema[0..base.len].* = base;
    schema[base.len..][0..2].* = pair(prefix ++ ".layer_norm", .{width}, width);
    for (0..layer_count) |i| {
        const key = prefix ++ std.fmt.comptimePrint(".layers.{d}", .{i});
        const cross = if (stage == .decoder)
            attentionSchema(key ++ ".encoder_attn", width) ++ pair(key ++ ".encoder_attn_layer_norm", .{width}, width)
        else
            [_]l.Parameter{};
        schema[base.len + 2 + i * items ..][0..items].* =
            attentionSchema(key ++ ".self_attn", width) ++ pair(key ++ ".self_attn_layer_norm", .{width}, width) ++
            pair(key ++ ".fc1", .{ cfg.ff, width }, cfg.ff) ++ pair(key ++ ".fc2", .{ width, cfg.ff }, width) ++
            pair(key ++ ".final_layer_norm", .{width}, width) ++ cross;
    }
    return schema;
}

fn attention(w: l.Weights, name: []const u8, query: Tensor, source: Tensor, heads: i64, mask: ?Tensor) Tensor {
    var key: [112]u8 = undefined;
    const k = t.splitHeads(t.linear(w, l.name(&key, "{s}.k_proj", .{name}), source, false), heads);
    const v = t.splitHeads(t.linear(w, l.name(&key, "{s}.v_proj", .{name}), source, true), heads);
    return attendProjected(w, name, query, k, v, heads, mask);
}

fn attendProjected(w: l.Weights, name: []const u8, query: Tensor, k: Tensor, v: Tensor, heads: i64, mask: ?Tensor) Tensor {
    var key: [112]u8 = undefined;
    const q = t.splitHeads(t.linear(w, l.name(&key, "{s}.q_proj", .{name}), query, true), heads);
    var logits = t.scores(q, k);
    if (mask) |m| logits = logits.add(m.broadcast(logits.shape(), &.{ 2, 3 }));
    return t.linear(w, l.name(&key, "{s}.out_proj", .{name}), t.attend(logits, v), true);
}

fn layers(w: l.Weights, input: Tensor, comptime stage: Stage, cfg: Config, context: ?Tensor, cross_cache: ?Tensor, mask: ?Tensor) Tensor {
    const prefix = "model." ++ @tagName(stage);
    var x = input;
    var key: [100]u8 = undefined;
    var block: [70]u8 = undefined;
    for (0..if (stage == .encoder) cfg.encoder_layers else cfg.decoder_layers) |i| {
        const name = l.name(&block, "{s}.layers.{d}", .{ prefix, i });
        const a = t.norm(w, l.name(&key, "{s}.self_attn_layer_norm", .{name}), x, 1e-5);
        x = x.add(attention(w, l.name(&key, "{s}.self_attn", .{name}), a, a, cfg.heads, mask));
        if (stage == .decoder) {
            const c = t.norm(w, l.name(&key, "{s}.encoder_attn_layer_norm", .{name}), x, 1e-5);
            const cross = l.name(&key, "{s}.encoder_attn", .{name});
            if (cross_cache) |cache| {
                const slot: i64 = @intCast(2 * i);
                const k = cache.slice(0, .{ .start = slot, .end = slot + 1 });
                const v = cache.slice(0, .{ .start = slot + 1, .end = slot + 2 });
                x = x.add(attendProjected(w, cross, c, k, v, cfg.heads, null));
            } else {
                x = x.add(attention(w, cross, c, context.?, cfg.heads, null));
            }
        }
        const n = t.norm(w, l.name(&key, "{s}.final_layer_norm", .{name}), x, 1e-5);
        const f = t.gelu(t.linear(w, l.name(&key, "{s}.fc1", .{name}), n, true));
        x = x.add(t.linear(w, l.name(&key, "{s}.fc2", .{name}), f, true));
    }
    return t.norm(w, prefix ++ ".layer_norm", x, 1e-5);
}

fn decode(w: l.Weights, cfg: Config, ids: Tensor, context: ?Tensor, cross_cache: ?Tensor) Tensor {
    const length = ids.dim(0);
    const word = w.get("model.decoder.embed_tokens", ".weight");
    const x = word.withTags(.{ .vocab, .width }).gather(.{ .vocab = ids }, .{}).reshape(.{ 1, length, cfg.width });
    const position = w.get("model.decoder.embed_positions", ".weight").slice(0, .{ .end = length });
    const shape: Shape = .init(.{ length, length }, .i32);
    const blocked = Tensor.iota(shape, 1).cmp(.GT, Tensor.iota(shape, 0));
    const mask = blocked.select(.scalar(-std.math.inf(f32), .f32), .scalar(0, .f32));
    return layers(w, x.add(position.reshape(x.shape())), .decoder, cfg, context, cross_cache, mask);
}

fn audioConv(w: l.Weights, name: []const u8, source: Tensor, stride: i64) Tensor {
    const filter = w.get(name, ".weight");
    const kernel = filter.reshape(.{ filter.dim(0), filter.dim(1), 1, 3 });
    const right = if (stride == 1) 1 else @mod(source.dim(3), 2);
    const result = source.conv2d(kernel, .{ .window_strides = &.{ 1, stride }, .padding = &.{ 0, 0, 1, right } });
    return t.gelu(l.biasAdd(result, w.get(name, ".bias"), 1));
}

pub fn Graph(comptime profile: Profile, comptime stage: Stage) type {
    const cfg = profile.config();
    return struct {
        pub const schema = parameters(cfg, stage);
        pub const config = cfg;

        pub fn forward(params: []const Tensor, input: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = params };
            if (stage == .encoder) {
                const first = audioConv(w, "model.encoder.conv1", input.reshape(.{ 1, cfg.mels, 1, 2 * cfg.source }), 1);
                const second = audioConv(w, "model.encoder.conv2", first, 2)
                    .reshape(.{ 1, cfg.width, cfg.source }).transpose(.{ 0, 2, 1 });
                const positions = w.get("model.encoder.embed_positions", ".weight");
                return layers(w, second.add(positions.reshape(second.shape())), .encoder, cfg, null, null, null);
            }
            const cells = cfg.width * cfg.source;
            const context = input.slice(0, .{ .end = cells }).reshape(.{ 1, cfg.source, cfg.width });
            const ids = input.slice(0, .{ .start = cells }).convert(.i32);
            const length = ids.dim(0);
            const state = decode(w, cfg, ids, context, null);
            return t.linearParameters(state.slice(1, .single(length - 1)), w.get("model.decoder.embed_tokens", ".weight"), null);
        }

        pub fn projectContext(params: []const Tensor, context: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = params };
            var projected: [2 * cfg.decoder_layers]Tensor = undefined;
            var name: [112]u8 = undefined;
            for (0..cfg.decoder_layers) |i| {
                projected[2 * i] = t.splitHeads(t.linear(w, l.name(&name, "model.decoder.layers.{d}.encoder_attn.k_proj", .{i}), context, false), cfg.heads);
                projected[2 * i + 1] = t.splitHeads(t.linear(w, l.name(&name, "model.decoder.layers.{d}.encoder_attn.v_proj", .{i}), context, true), cfg.heads);
            }
            return Tensor.concatenate(&projected, 0);
        }

        pub fn forwardAt(params: []const Tensor, cross_cache: Tensor, ids: Tensor, index: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = params };
            const state = decode(w, cfg, ids, null, cross_cache);
            const last = state.withTags(.{ .batch, .position, .width }).gather(.{ .position = index }, .{})
                .reshape(.{ 1, cfg.width });
            return t.linearParameters(last, w.get("model.decoder.embed_tokens", ".weight"), null);
        }
    };
}
