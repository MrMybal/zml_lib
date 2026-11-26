const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const qwen = @import("zml_lib/qwen_decoder");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const Parameter = l.Parameter;
const Weights = l.Weights;

const conv_prefix = "encoder.encoder.layers.";
const attention_prefix = "encoder.encoder_transformer.layers.";
const semantic = "encoder.quantizer.semantic_residual_vector_quantizer";
const acoustic = "encoder.quantizer.acoustic_residual_vector_quantizer";
const strides = [_]i64{ 4, 5, 6, 8 };
const width = 512;
const head_width = 64;
const heads = width / head_width;
const blocks = 8;
const entries = 2048;
const code_width = 256;

pub const input_width = 1;
pub const output_width = 16;
pub const samples_per_frame = 1920;
pub const max_samples = 30 * 24_000;

fn p(comptime name: []const u8, shape: anytype) Parameter {
    return .{ .name = name, .shape = .init(shape, .f32) };
}

fn convParameters(comptime name: []const u8, out: i64, in: i64, kernel: i64) [2]Parameter {
    return .{ p(name ++ ".weight", .{ out, in, kernel }), p(name ++ ".bias", .{out}) };
}

fn codebook(comptime group: usize) []const u8 {
    return if (group == 0) semantic ++ ".layers.0.codebook" else std.fmt.comptimePrint(acoustic ++ ".layers.{d}.codebook", .{group - 1});
}

pub const schema = blk: {
    @setEvalBranchQuota(100_000);
    var result: [159]Parameter = undefined;
    result[0..2].* = convParameters(conv_prefix ++ "0.conv", 64, 1, 7);
    var offset: usize = 2;
    for (strides, 0..) |stride, i| {
        const channels = 64 << i;
        const res = std.fmt.comptimePrint(conv_prefix ++ "{d}.block.", .{3 * i + 1});
        const down = std.fmt.comptimePrint(conv_prefix ++ "{d}.conv", .{3 * i + 3});
        result[offset..][0..2].* = convParameters(res ++ "1.conv", channels / 2, channels, 3);
        result[offset + 2 ..][0..2].* = convParameters(res ++ "3.conv", channels, channels / 2, 1);
        result[offset + 4 ..][0..2].* = convParameters(down, 2 * channels, channels, 2 * stride);
        offset += 6;
    }
    result[offset..][0..2].* = convParameters(conv_prefix ++ "14.conv", width, 1024, 3);
    result[offset + 2] = p("encoder.downsample.conv.weight", .{ width, width, 4 });
    offset += 3;
    for (0..blocks) |i| {
        const base = std.fmt.comptimePrint(attention_prefix ++ "{d}.", .{i});
        result[offset..][0..12].* = .{
            p(base ++ "input_layernorm.weight", .{width}),
            p(base ++ "input_layernorm.bias", .{width}),
            p(base ++ "self_attn.q_proj.weight", .{ width, width }),
            p(base ++ "self_attn.k_proj.weight", .{ width, width }),
            p(base ++ "self_attn.v_proj.weight", .{ width, width }),
            p(base ++ "self_attn.o_proj.weight", .{ width, width }),
            p(base ++ "self_attn_layer_scale.scale", .{width}),
            p(base ++ "post_attention_layernorm.weight", .{width}),
            p(base ++ "post_attention_layernorm.bias", .{width}),
            p(base ++ "mlp.fc1.weight", .{ 2048, width }),
            p(base ++ "mlp.fc2.weight", .{ width, 2048 }),
            p(base ++ "mlp_layer_scale.scale", .{width}),
        };
        offset += 12;
    }
    result[offset] = p(semantic ++ ".input_proj.weight", .{ code_width, width, 1 });
    result[offset + 1] = p(acoustic ++ ".input_proj.weight", .{ code_width, width, 1 });
    offset += 2;
    for (0..output_width) |i| {
        const name = codebook(i);
        result[offset] = p(name ++ ".cluster_usage", .{entries});
        result[offset + 1] = p(name ++ ".embed_sum", .{ entries, code_width });
        offset += 2;
    }
    std.debug.assert(offset == result.len);
    break :blk result;
};

pub fn check(input: []const f32, count: usize) !void {
    if (count == 0 or count > max_samples or input.len != count) return error.InvalidAudioLength;
    for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteAudio;
}

pub fn outputRows(count: usize) usize {
    return @divFloor(count + samples_per_frame - 1, samples_per_frame);
}

fn elu(x: Tensor) Tensor {
    const zero = Tensor.scalar(0, .f32).broad(x.shape());
    return x.maximum(zero).add(x.minimum(zero).exp().addConstant(@as(f32, -1)));
}

fn causal(w: Weights, name: []const u8, x: Tensor, stride: i64) Tensor {
    const kernel = w.get(name, ".weight");
    const y = x.conv1d(kernel, .{
        .window_strides = stride,
        .padding = &.{ kernel.dim(2) - stride, @mod(-x.dim(2), stride) },
    });
    return l.biasAdd(y, w.get(name, ".bias"), 1);
}

fn seanet(w: Weights, samples: Tensor) Tensor {
    var x = causal(w, conv_prefix ++ "0.conv", samples.reshape(.{ 1, 1, samples.dim(0) }), 1);
    inline for (strides, 0..) |stride, i| {
        const res = std.fmt.comptimePrint(conv_prefix ++ "{d}.block.", .{3 * i + 1});
        const down = std.fmt.comptimePrint(conv_prefix ++ "{d}.conv", .{3 * i + 3});
        const hidden = causal(w, res ++ "1.conv", elu(x), 1);
        x = causal(w, down, elu(x.add(causal(w, res ++ "3.conv", elu(hidden), 1))), stride);
    }
    return causal(w, conv_prefix ++ "14.conv", elu(x), 1);
}

fn encodeAttention(w: Weights, encoded: Tensor) Tensor {
    const time = encoded.dim(1);
    const positions: zml.Shape = .init(.{ time, head_width / 2 }, .i32);
    const phases = Tensor.iota(positions, 0).convert(.f32).div(
        Tensor.scalar(10_000, .f32).broad(positions.withDtype(.f32))
            .pow(Tensor.iota(positions, 1).convert(.f32).scale(2 / @as(f32, head_width))),
    );
    const rotary = Tensor.concatenate(&.{ phases, phases }, 1);
    const sin = rotary.sin();
    const cos = rotary.cos();
    const square: zml.Shape = .init(.{ time, time }, .i32);
    const future = Tensor.iota(square, 1).cmp(.GT, Tensor.iota(square, 0));
    const floats = square.withDtype(.f32);
    const mask = future.select(Tensor.scalar(-std.math.inf(f32), .f32).broad(floats), Tensor.scalar(0, .f32).broad(floats));

    var x = encoded;
    for (0..blocks) |i| {
        var scope: [80]u8 = undefined;
        var key: [112]u8 = undefined;
        const name = l.name(&scope, attention_prefix ++ "{d}", .{i});
        const norm = t.norm(w, l.name(&key, "{s}.input_layernorm", .{name}), x, 1e-5);
        const q = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.q_proj", .{name}), norm, false), heads);
        const k = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.k_proj", .{name}), norm, false), heads);
        const v = t.splitHeads(t.linear(w, l.name(&key, "{s}.self_attn.v_proj", .{name}), norm, false), heads);
        const scores = qwen.scoresAfterDot(qwen.rotate(q, sin, cos), qwen.rotate(k, sin, cos));
        const attention = t.attend(scores.add(mask.broadcast(scores.shape(), &.{ 2, 3 })), v);
        const mixed = t.linear(w, l.name(&key, "{s}.self_attn.o_proj", .{name}), attention, false);
        x = x.add(mixed.mul(w.get(name, ".self_attn_layer_scale.scale").broadcast(mixed.shape(), &.{2})));
        const normalized = t.norm(w, l.name(&key, "{s}.post_attention_layernorm", .{name}), x, 1e-5);
        const hidden = t.gelu(t.linear(w, l.name(&key, "{s}.mlp.fc1", .{name}), normalized, false));
        const down = t.linear(w, l.name(&key, "{s}.mlp.fc2", .{name}), hidden, false);
        x = x.add(down.mul(w.get(name, ".mlp_layer_scale.scale").broadcast(down.shape(), &.{2})));
    }
    return x;
}

fn downsample(w: Weights, x: Tensor) Tensor {
    const time = x.dim(2);
    const first = x.slice(2, .{ .end = 1 });
    const left = first.broad(.init(.{ 1, width, 2 }, .f32));
    const padded = if (@mod(time, 2) == 0)
        Tensor.concatenate(&.{ left, x }, 2)
    else
        Tensor.concatenate(&.{ left, x, x.slice(2, .{ .start = time - 1 }) }, 2);
    return padded.conv1d(w.get("encoder.downsample.conv", ".weight"), .{ .window_strides = 2 });
}

const Quantized = struct {
    ids: Tensor,
    table: Tensor,
};

fn quantize(w: Weights, comptime group: usize, signal: Tensor) Quantized {
    const name = codebook(group);
    const usage = w.get(name, ".cluster_usage");
    const minimum = Tensor.scalar(1e-5, .f32).broad(usage.shape());
    const sums = w.get(name, ".embed_sum");
    const table = sums.div(usage.maximum(minimum).broadcast(sums.shape(), &.{0}));
    const dot = signal.dotGeneral(table, &.{.{ 1, 1 }}, &.{}).scale(@as(f32, -2));
    const squared = table.mul(table).sum(1).reshape(.{entries});
    const distance = dot.add(squared.broadcast(dot.shape(), &.{1}));
    return .{ .table = table, .ids = distance.negate().argMax(1).indices.reshape(.{signal.dim(0)}) };
}

pub fn forward(parameters: []const Tensor, samples: Tensor) Tensor {
    const w: Weights = .{ .schema = &schema, .tensors = parameters };
    const hidden = encodeAttention(w, seanet(w, samples).transpose(.{ 0, 2, 1 }));
    const sampled = downsample(w, hidden.transpose(.{ 0, 2, 1 }));
    const x = sampled.transpose(.{ 0, 2, 1 }).reshape(.{ sampled.dim(2), width });
    const semantic_input = t.linearParameters(x, w.get(semantic ++ ".input_proj", ".weight").squeeze(2), null);
    var residual = t.linearParameters(x, w.get(acoustic ++ ".input_proj", ".weight").squeeze(2), null);
    var codes: [output_width]Tensor = undefined;
    codes[0] = quantize(w, 0, semantic_input).ids.convert(.f32).reshape(.{ x.dim(0), 1 });
    inline for (1..output_width) |group| {
        const result = quantize(w, group, residual);
        codes[group] = result.ids.convert(.f32).reshape(.{ x.dim(0), 1 });
        if (group + 1 < output_width) {
            const selected = qwen.tokenEmbeddings(result.table, result.ids).reshape(residual.shape());
            residual = residual.sub(selected);
        }
    }
    return Tensor.concatenate(&codes, 1);
}
