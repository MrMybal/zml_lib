const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const decoder = @import("zml_lib/qwen_decoder");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const Parameter = l.Parameter;
const Weights = l.Weights;

pub const input_width = 16;
pub const output_width = 1;
pub const samples_per_frame = 1920;
pub const max_frames = 325;

const entries = 2048;
const embedding_width = 256;
const latent_width = 1024;
const model_width = 512;
const intermediate = 1024;
const head_count = 16;
const head_width = 64;
const layers = 8;
const sliding_window = 72;
const upsample_rates = [_]i64{ 8, 5, 4, 3 };
const synthesis_width = 1536;
const output_channels = synthesis_width / 16;
const quantizer = "decoder.quantizer.";
const transformer = "decoder.pre_transformer.";

fn p(comptime name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn affine(comptime name: []const u8, dims: anytype, width: i64) [2]Parameter {
    return .{ p(name ++ ".weight", dims), p(name ++ ".bias", .{width}) };
}

fn activation(comptime name: []const u8, width: i64) [2]Parameter {
    return .{ p(name ++ ".alpha", .{width}), p(name ++ ".beta", .{width}) };
}

fn put(parameters: []Parameter, offset: *usize, values: []const Parameter) void {
    @memcpy(parameters[offset.*..][0..values.len], values);
    offset.* += values.len;
}

pub const schema = blk: {
    @setEvalBranchQuota(200_000);
    var parameters: [269]Parameter = undefined;
    var offset: usize = 0;
    for (0..input_width) |group| {
        const key = codebookName(group);
        put(&parameters, &offset, &.{
            p(key ++ ".cluster_usage", .{entries}),
            p(key ++ ".embedding_sum", .{ entries, embedding_width }),
        });
    }
    put(&parameters, &offset, &.{
        p(quantizer ++ "rvq_first.output_proj.weight", .{ model_width, embedding_width, 1 }),
        p(quantizer ++ "rvq_rest.output_proj.weight", .{ model_width, embedding_width, 1 }),
        p(transformer ++ "norm.weight", .{model_width}),
    });
    put(&parameters, &offset, &affine("decoder.pre_conv.conv", .{ latent_width, model_width, 3 }, latent_width));
    put(&parameters, &offset, &affine(transformer ++ "input_proj", .{ model_width, latent_width }, model_width));
    put(&parameters, &offset, &affine(transformer ++ "output_proj", .{ latent_width, model_width }, latent_width));
    for (0..layers) |i| {
        const name = std.fmt.comptimePrint(transformer ++ "layers.{d}.", .{i});
        put(&parameters, &offset, &.{
            p(name ++ "input_layernorm.weight", .{model_width}),
            p(name ++ "self_attn.q_proj.weight", .{ head_count * head_width, model_width }),
            p(name ++ "self_attn.k_proj.weight", .{ head_count * head_width, model_width }),
            p(name ++ "self_attn.v_proj.weight", .{ head_count * head_width, model_width }),
            p(name ++ "self_attn.o_proj.weight", .{ model_width, head_count * head_width }),
            p(name ++ "self_attn_layer_scale.scale", .{model_width}),
            p(name ++ "post_attention_layernorm.weight", .{model_width}),
            p(name ++ "mlp.gate_proj.weight", .{ intermediate, model_width }),
            p(name ++ "mlp.up_proj.weight", .{ intermediate, model_width }),
            p(name ++ "mlp.down_proj.weight", .{ model_width, intermediate }),
            p(name ++ "mlp_layer_scale.scale", .{model_width}),
        });
    }
    for (0..2) |i| {
        const name = std.fmt.comptimePrint("decoder.upsample.{d}.", .{i});
        put(&parameters, &offset, &affine(name ++ "0.conv", .{ latent_width, latent_width, 2 }, latent_width));
        put(&parameters, &offset, &affine(name ++ "1.dwconv.conv", .{ latent_width, 1, 7 }, latent_width));
        put(&parameters, &offset, &affine(name ++ "1.norm", .{latent_width}, latent_width));
        put(&parameters, &offset, &affine(name ++ "1.pwconv1", .{ 4 * latent_width, latent_width }, 4 * latent_width));
        put(&parameters, &offset, &affine(name ++ "1.pwconv2", .{ latent_width, 4 * latent_width }, latent_width));
        put(&parameters, &offset, &.{p(name ++ "1.gamma", .{latent_width})});
    }
    put(&parameters, &offset, &affine("decoder.decoder.0.conv", .{ synthesis_width, latent_width, 7 }, synthesis_width));
    for (upsample_rates, 0..) |rate, i| {
        const width = synthesis_width >> i;
        const reduced = width / 2;
        const name = std.fmt.comptimePrint("decoder.decoder.{d}.block.", .{i + 1});
        put(&parameters, &offset, &activation(name ++ "0", width));
        put(&parameters, &offset, &affine(name ++ "1.conv", .{ width, reduced, 2 * rate }, reduced));
        for (2..5) |j| {
            const unit = std.fmt.comptimePrint("{s}{d}.", .{ name, j });
            put(&parameters, &offset, &activation(unit ++ "act1", reduced));
            put(&parameters, &offset, &affine(unit ++ "conv1.conv", .{ reduced, reduced, 7 }, reduced));
            put(&parameters, &offset, &activation(unit ++ "act2", reduced));
            put(&parameters, &offset, &affine(unit ++ "conv2.conv", .{ reduced, reduced, 1 }, reduced));
        }
    }
    put(&parameters, &offset, &activation("decoder.decoder.5", output_channels));
    put(&parameters, &offset, &affine("decoder.decoder.6.conv", .{ 1, output_channels, 7 }, 1));
    std.debug.assert(offset == parameters.len);
    break :blk parameters;
};

fn codebookName(comptime group: usize) []const u8 {
    if (group == 0) return quantizer ++ "rvq_first.vq.layers.0._codebook";
    return std.fmt.comptimePrint(quantizer ++ "rvq_rest.vq.layers.{d}._codebook", .{group - 1});
}

pub fn check(input: []const f32, count: usize) !void {
    if (count == 0 or count > max_frames or input.len != count * input_width) return error.InvalidFrameCount;
    for (input) |code| {
        if (!std.math.isFinite(code) or code < 0 or code >= entries or @trunc(code) != code)
            return error.InvalidSpeechCode;
    }
}

pub fn outputRows(count: usize) usize {
    return count * samples_per_frame;
}

fn embedding(w: Weights, codes: Tensor, comptime group: usize) Tensor {
    const name = codebookName(group);
    const ids = codes.slice(1, .single(group)).convert(.i32);
    const usage = decoder.tokenEmbeddings(w.get(name, ".cluster_usage").reshape(.{ entries, 1 }), ids);
    const minimum = Tensor.scalar(1e-5, .f32).broad(usage.shape());
    const values = decoder.tokenEmbeddings(w.get(name, ".embedding_sum"), ids);
    return values.div(usage.maximum(minimum).broad(values.shape()));
}

fn quantize(w: Weights, codes: Tensor) Tensor {
    const semantic = embedding(w, codes, 0);
    var acoustic = embedding(w, codes, 1);
    inline for (2..input_width) |group| acoustic = acoustic.add(embedding(w, codes, group));
    const first = w.get(quantizer ++ "rvq_first.output_proj", ".weight").squeeze(2);
    const rest = w.get(quantizer ++ "rvq_rest.output_proj", ".weight").squeeze(2);
    return t.linearParameters(semantic, first, null).add(t.linearParameters(acoustic, rest, null));
}

fn rms(w: Weights, name: []const u8, x: Tensor) Tensor {
    const inv_std = x.mul(x).mean(2).addConstant(@as(f32, 1e-5)).rsqrt();
    return x.mul(inv_std.broad(x.shape())).mul(w.get(name, ".weight").broadcast(x.shape(), &.{2}));
}

fn preTransformer(w: Weights, x: Tensor) Tensor {
    const count = x.dim(1);
    const pairs: zml.Shape = .init(.{ count, head_width / 2 }, .i32);
    const frequency_step = @as(f32, -2) / @as(f32, head_width);
    const scale = Tensor.iota(pairs, 1).convert(.f32).scale(frequency_step);
    const base = Tensor.scalar(10_000, .f32).broad(scale.shape());
    const phase = Tensor.iota(pairs, 0).convert(.f32).mul(base.pow(scale));
    const rotary = Tensor.concatenate(&.{ phase, phase }, 1);
    const sin = rotary.sin();
    const cos = rotary.cos();

    const square: zml.Shape = .init(.{ count, count }, .i32);
    const distance = Tensor.iota(square, 0).sub(Tensor.iota(square, 1));
    const valid = distance.cmp(.GE, Tensor.scalar(0, .i32).broad(square))
        .logical(.AND, distance.cmp(.LT, Tensor.scalar(sliding_window, .i32).broad(square)));
    const mask = valid.select(Tensor.scalar(0, .f32).broad(square.withDtype(.f32)), Tensor.scalar(-std.math.inf(f32), .f32).broad(square.withDtype(.f32)));

    var hidden = t.linear(w, transformer ++ "input_proj", x, true);
    for (0..layers) |i| {
        var scope: [56]u8 = undefined;
        var key: [96]u8 = undefined;
        const name = l.name(&scope, transformer ++ "layers.{d}", .{i});
        const before_attn = rms(w, l.name(&key, "{s}.input_layernorm", .{name}), hidden);
        const q = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.q_proj", .{name}), before_attn, false), head_count);
        const k = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.k_proj", .{name}), before_attn, false), head_count);
        const v = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.v_proj", .{name}), before_attn, false), head_count);
        const scores = decoder.scoresAfterDot(decoder.rotate(q, sin, cos), decoder.rotate(k, sin, cos));
        const attention = t.attend(scores.add(mask.broadcast(scores.shape(), &.{ 2, 3 })), v);
        const mixed = t.linear(w, l.name(&key, "{s}.self_attn.o_proj", .{name}), attention, false);
        hidden = hidden.add(mixed.mul(w.get(name, ".self_attn_layer_scale.scale").broadcast(mixed.shape(), &.{2})));

        const before_mlp = rms(w, l.name(&key, "{s}.post_attention_layernorm", .{name}), hidden);
        const gate = t.linear(w, l.name(&key, "{s}.mlp.gate_proj", .{name}), before_mlp, false);
        const up = t.linear(w, l.name(&key, "{s}.mlp.up_proj", .{name}), before_mlp, false);
        const down = t.linear(w, l.name(&key, "{s}.mlp.down_proj", .{name}), gate.mul(gate.sigmoid()).mul(up), false);
        hidden = hidden.add(down.mul(w.get(name, ".mlp_layer_scale.scale").broadcast(down.shape(), &.{2})));
    }
    return t.linear(w, transformer ++ "output_proj", rms(w, transformer ++ "norm", hidden), true);
}

fn causalConv(w: Weights, name: []const u8, x: Tensor, dilation: i64, groups: i64) Tensor {
    const kernel = w.get(name, ".weight");
    const y = x.conv1d(kernel, .{
        .padding = &.{ (kernel.dim(2) - 1) * dilation, 0 },
        .rhs_dilation = dilation,
        .feature_group_count = groups,
    });
    return l.biasAdd(y, w.get(name, ".bias"), 1);
}

fn upsample(w: Weights, name: []const u8, x: Tensor, stride: i64) Tensor {
    const kernel = w.get(name, ".weight");
    const length = x.dim(2);
    const channels = kernel.dim(1);
    const taps = kernel.dim(2);
    const weights = kernel.transpose(.{ 2, 1, 0 }).reshape(.{ taps * channels, x.dim(1) });
    const tiles = t.linearParameters(x.transpose(.{ 0, 2, 1 }), weights, null);
    const result = if (taps == stride) tiles else blk: {
        // The second half of each kernel overlaps the next input frame.
        const parts = tiles.reshape(.{ 1, length, 2, stride, channels }).withTags(.{ .batch, .time, .overlap, .tap, .channel });
        const previous = parts.slice(2, .single(1))
            .pad(@as(f32, 0), .{ .time = Tensor.Pad{ .low = 1 } }).slice(1, .{ .end = length });
        break :blk parts.slice(2, .single(0)).add(previous);
    };
    return l.biasAdd(result.reshape(.{ 1, length * stride, channels }), w.get(name, ".bias"), 2)
        .transpose(.{ 0, 2, 1 });
}

fn convNeXt(w: Weights, comptime name: []const u8, x: Tensor) Tensor {
    const depthwise = causalConv(w, name ++ ".dwconv.conv", x, 1, x.dim(1)).transpose(.{ 0, 2, 1 });
    const normalized = t.norm(w, name ++ ".norm", depthwise, 1e-6);
    const expanded = t.gelu(t.linear(w, name ++ ".pwconv1", normalized, true));
    const reduced = t.linear(w, name ++ ".pwconv2", expanded, true);
    return x.add(reduced.mul(w.get(name, ".gamma").broadcast(reduced.shape(), &.{2})).transpose(.{ 0, 2, 1 }));
}

fn snake(w: Weights, name: []const u8, x: Tensor) Tensor {
    const alpha = w.get(name, ".alpha").exp();
    const beta = w.get(name, ".beta").exp().addConstant(@as(f32, 1e-9));
    const sine = x.mul(alpha.broadcast(x.shape(), &.{1})).sin();
    return x.add(sine.mul(sine).div(beta.broadcast(x.shape(), &.{1})));
}

fn synthesisBlock(w: Weights, comptime index: usize, x: Tensor) Tensor {
    const name = std.fmt.comptimePrint("decoder.decoder.{d}.block.", .{index + 1});
    var hidden = upsample(w, name ++ "1.conv", snake(w, name ++ "0", x), upsample_rates[index]);
    inline for (.{ 1, 3, 9 }, 2..) |dilation, j| {
        const unit = std.fmt.comptimePrint("{s}{d}", .{ name, j });
        const first = causalConv(w, unit ++ ".conv1.conv", snake(w, unit ++ ".act1", hidden), dilation, 1);
        const second = causalConv(w, unit ++ ".conv2.conv", snake(w, unit ++ ".act2", first), 1, 1);
        hidden = hidden.add(second);
    }
    return hidden;
}

pub fn forward(parameters: []const Tensor, codes: Tensor) Tensor {
    const w: Weights = .{ .schema = &schema, .tensors = parameters };
    const embeddings = quantize(w, codes).transpose(.{ 0, 2, 1 });
    const projected = causalConv(w, "decoder.pre_conv.conv", embeddings, 1, 1).transpose(.{ 0, 2, 1 });
    var hidden = preTransformer(w, projected).transpose(.{ 0, 2, 1 });
    inline for (0..2) |i| {
        const name = std.fmt.comptimePrint("decoder.upsample.{d}.", .{i});
        hidden = convNeXt(w, name ++ "1", upsample(w, name ++ "0.conv", hidden, 2));
    }
    hidden = causalConv(w, "decoder.decoder.0.conv", hidden, 1, 1);
    inline for (0..upsample_rates.len) |i| hidden = synthesisBlock(w, i, hidden);
    const audio = causalConv(w, "decoder.decoder.6.conv", snake(w, "decoder.decoder.5", hidden), 1, 1);
    return audio.clamp(Tensor.scalar(-1, .f32), Tensor.scalar(1, .f32)).reshape(.{ audio.dim(2), 1 });
}
