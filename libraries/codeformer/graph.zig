const std = @import("std");
const l = @import("zml_lib/runtime").layers;
const Tensor = l.Tensor;
const schema = @import("weights.zig").schema;

pub const size = 512;
pub const input_count = 3 * size * size;
const side = 16;
const tokens = side * side;
const codebook_size = 1024;
const embedding = 512;
const latent_channels = 256;
pub const output_count = input_count + tokens * codebook_size;

const same = [4]i64{ 1, 1, 1, 1 };
const valid = [4]i64{ 0, 0, 0, 0 };

fn residual(w: l.Weights, prefix: []const u8, x: Tensor) Tensor {
    var key: [64]u8 = undefined;
    var y = l.groupNorm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 32, 1e-6).silu();
    y = l.conv(w, l.name(&key, "{s}.conv1", .{prefix}), y, 1, same, true);
    y = l.groupNorm(w, l.name(&key, "{s}.norm2", .{prefix}), y, 32, 1e-6).silu();
    y = l.conv(w, l.name(&key, "{s}.conv2", .{prefix}), y, 1, same, true);
    const shortcut = if (x.dim(1) == y.dim(1)) x else l.conv(w, l.name(&key, "{s}.conv_out", .{prefix}), x, 1, valid, true);
    return shortcut.add(y);
}

fn matrixAttention(q: Tensor, k: Tensor, v: Tensor, heads: i64) Tensor {
    const seq = q.dim(0);
    const dim = q.dim(1);
    const head_dim = @divExact(dim, heads);
    const qs = q.reshape(.{ seq, heads, head_dim }).transpose(.{ 1, 0, 2 });
    const ks = k.reshape(.{ seq, heads, head_dim }).transpose(.{ 1, 0, 2 });
    const vs = v.reshape(.{ seq, heads, head_dim }).transpose(.{ 1, 0, 2 });
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const logits = qs.dotGeneral(ks, &.{.{ 2, 2 }}, &.{.{ 0, 0 }}).scale(scale);
    return logits.softmax(2).dotGeneral(vs, &.{.{ 2, 1 }}, &.{.{ 0, 0 }}).transpose(.{ 1, 0, 2 })
        .reshape(.{ seq, dim });
}

fn spatialAttention(w: l.Weights, prefix: []const u8, x: Tensor) Tensor {
    var key: [64]u8 = undefined;
    const count = x.dim(2) * x.dim(3);
    const channels = x.dim(1);
    const h = l.groupNorm(w, l.name(&key, "{s}.norm", .{prefix}), x, 32, 1e-6);
    const q = l.conv(w, l.name(&key, "{s}.q", .{prefix}), h, 1, valid, true).reshape(.{ channels, count }).transpose(.{ 1, 0 });
    const k = l.conv(w, l.name(&key, "{s}.k", .{prefix}), h, 1, valid, true).reshape(.{ channels, count }).transpose(.{ 1, 0 });
    const v = l.conv(w, l.name(&key, "{s}.v", .{prefix}), h, 1, valid, true).reshape(.{ channels, count }).transpose(.{ 1, 0 });
    const attention = matrixAttention(q, k, v, 1).transpose(.{ 1, 0 }).reshape(x.shape());
    return x.add(l.conv(w, l.name(&key, "{s}.proj_out", .{prefix}), attention, 1, valid, true));
}

const Block = enum { conv, residual, attention, down, up, norm };

const encoder = [_]Block{
    .conv,     .residual,  .residual, .down,     .residual, .residual, .down,     .residual,  .residual, .down,
    .residual, .residual,  .down,     .residual, .residual, .down,     .residual, .attention, .residual, .attention,
    .residual, .attention, .residual, .norm,     .conv,
};

const generator = [_]Block{
    .conv,     .residual, .attention, .residual, .residual, .attention, .residual, .attention,
    .up,       .residual, .residual,  .up,       .residual, .residual,  .up,       .residual,
    .residual, .up,       .residual,  .residual, .up,       .residual,  .residual, .norm,
    .conv,
};

fn vqBlock(w: l.Weights, prefix: []const u8, x: Tensor, kind: Block) Tensor {
    var key: [64]u8 = undefined;
    return switch (kind) {
        .conv => l.conv(w, prefix, x, 1, same, true),
        .residual => residual(w, prefix, x),
        .attention => spatialAttention(w, prefix, x),
        .norm => l.groupNorm(w, prefix, x, 32, 1e-6),
        .down => l.conv(w, l.name(&key, "{s}.conv", .{prefix}), x, 2, .{ 0, 1, 0, 1 }, true),
        .up => l.conv(w, l.name(&key, "{s}.conv", .{prefix}), l.nearest2x(x), 1, same, true),
    };
}

fn selfAttentionProjection(x: Tensor, matrix: Tensor, bias: Tensor, start: i64) Tensor {
    const slice: Tensor.Slice = .{ .start = start, .end = start + embedding };
    const y = x.dotGeneral(matrix.slice(0, slice), &.{.{ 1, 1 }}, &.{});
    return l.biasAdd(y, bias.slice(0, slice), 1);
}

fn transformer(w: l.Weights, x: Tensor, pos: Tensor, index: usize) Tensor {
    var buffer: [24]u8 = undefined;
    var key: [64]u8 = undefined;
    const prefix = l.name(&buffer, "ft_layers.{d}", .{index});
    const normalized = l.layerNorm(w, l.name(&key, "{s}.norm1", .{prefix}), x);
    const positioned = normalized.add(pos);
    const att = l.name(&key, "{s}.self_attn", .{prefix});
    const matrix = w.get(att, ".in_proj_weight");
    const bias = w.get(att, ".in_proj_bias");
    const q = selfAttentionProjection(positioned, matrix, bias, 0);
    const k = selfAttentionProjection(positioned, matrix, bias, embedding);
    const v = selfAttentionProjection(normalized, matrix, bias, 2 * embedding);
    const y = x.add(l.linear(w, l.name(&key, "{s}.self_attn.out_proj", .{prefix}), matrixAttention(q, k, v, 8), true));
    const z = l.layerNorm(w, l.name(&key, "{s}.norm2", .{prefix}), y);
    const hidden = l.linear(w, l.name(&key, "{s}.linear1", .{prefix}), z, true);
    const gelu = hidden.mul(hidden.scale(@as(f32, 0.7071067811865476)).erf().addConstant(@as(f32, 1))).scale(@as(f32, 0.5));
    return y.add(l.linear(w, l.name(&key, "{s}.linear2", .{prefix}), gelu, true));
}

fn fuse(w: l.Weights, resolution: i64, enc: Tensor, dec: Tensor, fidelity: Tensor) Tensor {
    var base: [40]u8 = undefined;
    var key: [64]u8 = undefined;
    const prefix = l.name(&base, "fuse_convs_dict.{d}", .{resolution});
    const input = Tensor.concatenate(&.{ enc, dec }, 1);
    const encoded = residual(w, l.name(&key, "{s}.encode_enc", .{prefix}), input);
    const scale_hidden = l.leakyRelu(l.conv(w, l.name(&key, "{s}.scale.0", .{prefix}), encoded, 1, same, true));
    const scale = l.conv(w, l.name(&key, "{s}.scale.2", .{prefix}), scale_hidden, 1, same, true);
    const shift_hidden = l.leakyRelu(l.conv(w, l.name(&key, "{s}.shift.0", .{prefix}), encoded, 1, same, true));
    const shift = l.conv(w, l.name(&key, "{s}.shift.2", .{prefix}), shift_hidden, 1, same, true);
    return dec.add(dec.mul(scale).add(shift).mul(fidelity.broad(dec.shape())));
}

pub fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    var x = input.slice(0, .{ .end = input_count }).reshape(.{ 1, 3, size, size });
    const fidelity = input.slice(0, .single(input_count));
    var features: [4]Tensor = undefined;
    var key: [40]u8 = undefined;
    for (encoder, 0..) |kind, index| {
        x = vqBlock(w, l.name(&key, "encoder.blocks.{d}", .{index}), x, kind);
        switch (index) {
            5 => features[3] = x,
            8 => features[2] = x,
            11 => features[1] = x,
            14 => features[0] = x,
            else => {},
        }
    }
    var features_emb = l.linear(w, "feat_emb", x.reshape(.{ latent_channels, tokens }).transpose(.{ 1, 0 }), true);
    const pos = w.get("position_emb", "");
    for (0..9) |i| features_emb = transformer(w, features_emb, pos, i);
    const logits = l.linear(w, "idx_pred_layer.1", l.layerNorm(w, "idx_pred_layer.0", features_emb), false);
    const code_ids = Tensor.iota(logits.shape(), 1);
    const maxima = logits.max(1).broad(logits.shape());
    const absent = Tensor.scalar(codebook_size, .i32).broad(code_ids.shape());
    const indices = Tensor.select(logits.cmp(.EQ, maxima), code_ids, absent).min(1).reshape(.{ .token = tokens });
    x = w.get("quantize.embedding", ".weight").reshape(.{ .code = codebook_size, .feature = latent_channels })
        .gather(.{ .code = indices }, .{}).transpose(.{ 1, 0 }).reshape(.{ 1, latent_channels, side, side });
    for (generator, 0..) |kind, index| {
        x = vqBlock(w, l.name(&key, "generator.blocks.{d}", .{index}), x, kind);
        const feature_index: usize = switch (index) {
            9 => 0,
            12 => 1,
            15 => 2,
            18 => 3,
            else => continue,
        };
        x = fuse(w, x.dim(3), features[feature_index], x, fidelity);
    }
    return Tensor.concatenate(&.{ x.flatten(), logits.flatten() }, 0);
}
