const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const qwen = @import("zml_lib/qwen_decoder");
const Tensor = zml.Tensor;
const Parameter = runtime.layers.Parameter;
const Weights = runtime.layers.Weights;

pub const Profile = enum { qwen3_tts_0b6, qwen3_tts_1b7, fixture };
pub const Stage = enum { text, codec, talker, predictor };
pub const codebooks = 16;

fn parameter(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f16) };
}

fn decoder(width: i64, mlp: i64, layers: usize, vocab: i64, small: bool) qwen.Profile {
    return .{
        .vocab = vocab,
        .layers = layers,
        .width = width,
        .mlp = mlp,
        .heads = if (small) 4 else 16,
        .kv_heads = if (small) 2 else 8,
        .head_width = if (small) 8 else 128,
        .has_bias = false,
        .head_norm = true,
        .separate_output = false,
        .scale_attention_after_dot = true,
        .eos_ids = &.{},
    };
}

fn backbone(comptime config: qwen.Profile, weights: Weights, embeddings: Tensor) Tensor {
    const count = embeddings.dim(1);
    const coordinates: zml.Shape = .init(.{ count, count }, .i32);
    const future = Tensor.iota(coordinates, 1).cmp(.GT, Tensor.iota(coordinates, 0));
    const score_shape = coordinates.withDtype(.f32);
    const mask = future.select(Tensor.scalar(-std.math.inf(f32), .f32).broad(score_shape), Tensor.scalar(0, .f32).broad(score_shape));
    const rope_shape: zml.Shape = .init(.{ count, @divExact(config.head_width, 2) }, .i32);
    const power = Tensor.iota(rope_shape, 1).convert(.f32).scale(2 / @as(f32, @floatFromInt(config.head_width)));
    const frequency = Tensor.scalar(1, .f32).broad(power.shape()).div(
        Tensor.scalar(1_000_000, .f32).broad(power.shape()).pow(power),
    );
    // Text-only synthesis assigns identical positions to the three Talker RoPE axes.
    const angle = Tensor.iota(rope_shape, 0).convert(.f32).mul(frequency);
    const doubled = Tensor.concatenate(&.{ angle, angle }, 1);
    return qwen.layersForward(config, weights, embeddings, mask, doubled.sin(), doubled.cos());
}

pub fn Graph(comptime profile: Profile, comptime stage: Stage) type {
    @setEvalBranchQuota(2_000_000);
    const small = profile == .fixture;
    const width: i64 = switch (profile) {
        .qwen3_tts_0b6 => 1024,
        .qwen3_tts_1b7 => 2048,
        .fixture => 16,
    };
    const text_width: i64 = if (small) 12 else 2048;
    const text_vocab: i64 = if (small) 64 else 151_936;
    const talker_vocab: i64 = if (small) 32 else 3072;
    const codec_vocab: i64 = if (small) 16 else 2048;
    const predictor_width: i64 = if (small) 8 else 1024;
    const talker = decoder(width, width * 3, if (small) 2 else 28, talker_vocab, small);
    const predictor = decoder(predictor_width, predictor_width * 3, if (small) 2 else 5, codec_vocab, small);
    const talker_parameters = qwen.schema(talker, .f16);
    const predictor_parameters = qwen.schema(predictor, .f16);
    const projection = if (width == predictor_width) [_]Parameter{} else [_]Parameter{
        parameter("small_to_mtp_projection.weight", .{ predictor_width, width }),
        parameter("small_to_mtp_projection.bias", .{predictor_width}),
    };
    const codec_embeddings = blk: {
        var result: [codebooks]Parameter = undefined;
        result[0] = parameter("codec_embedding.weight", .{ talker_vocab, width });
        for (1..codebooks) |group| {
            const name = std.fmt.comptimePrint("codec_embedding.{d}.weight", .{group - 1});
            result[group] = parameter(name, .{ codec_vocab, width });
        }
        break :blk result;
    };
    const predictor_heads = blk: {
        var result: [codebooks - 1]Parameter = undefined;
        for (&result, 0..) |*head, i| {
            head.* = parameter(std.fmt.comptimePrint("lm_head.{d}.weight", .{i}), .{ codec_vocab, predictor_width });
        }
        break :blk result;
    };
    return struct {
        pub const schema = switch (stage) {
            .text => [_]Parameter{
                parameter("text_embedding.weight", .{ text_vocab, text_width }),
                parameter("text_projection.linear_fc1.weight", .{ text_width, text_width }),
                parameter("text_projection.linear_fc1.bias", .{text_width}),
                parameter("text_projection.linear_fc2.weight", .{ width, text_width }),
                parameter("text_projection.linear_fc2.bias", .{width}),
            },
            .codec => codec_embeddings,
            .talker => talker_parameters[1..].* ++ [_]Parameter{parameter("codec_head.weight", .{ talker_vocab, width })},
            .predictor => predictor_parameters[1..].* ++ projection ++ predictor_heads,
        };
        pub const input_width: usize = if (stage == .text or stage == .codec) 1 else @intCast(width);
        pub const output_width: usize = switch (stage) {
            .text, .codec => @intCast(width),
            .talker => @intCast(width + talker_vocab),
            .predictor => @intCast(codec_vocab),
        };

        pub fn check(input: []const f32, count: usize) !void {
            if (count == 0 or count > 1024 or input.len != count * input_width)
                return error.InvalidTokenCount;
            if ((stage == .predictor or stage == .codec) and count != codebooks)
                return error.ExpectedSixteenRows;
            for (input, 0..) |value, i| {
                if (!std.math.isFinite(value)) return error.NonfiniteInput;
                if (stage == .text or stage == .codec) {
                    const limit = if (stage == .text) text_vocab else if (i == 0) talker_vocab else codec_vocab;
                    if (value < 0 or @trunc(value) != value or value >= @as(f32, @floatFromInt(limit)))
                        return error.InvalidTokenId;
                }
            }
        }

        pub fn outputRows(count: usize) usize {
            return if (stage == .predictor) codebooks - 1 else count;
        }

        pub fn forward(parameters: []const Tensor, input: Tensor) Tensor {
            const weights: Weights = .{ .schema = &schema, .tensors = parameters };
            const count = input.dim(0);
            switch (stage) {
                .text => {
                    const ids = input.flatten().convert(.i32);
                    const embedded = qwen.tokenEmbeddings(weights.get("text_embedding", ".weight"), ids);
                    const expanded = qwen.linear(weights, "text_projection.linear_fc1", embedded, true);
                    return qwen.linear(weights, "text_projection.linear_fc2", expanded.mul(expanded.sigmoid()), true)
                        .reshape(.{ count, width });
                },
                .codec => {
                    const ids = input.flatten().convert(.i32);
                    var rows: [codebooks]Tensor = undefined;
                    for (&rows, 0..) |*row, group| {
                        row.* = qwen.tokenEmbeddings(parameters[group], ids.slice(0, .{ .start = @intCast(group), .end = @intCast(group + 1) }))
                            .reshape(.{ 1, width });
                    }
                    return Tensor.concatenate(&rows, 0);
                },
                .talker => {
                    const hidden = backbone(talker, weights, input.reshape(.{ 1, count, width }));
                    const logits = qwen.linear(weights, "codec_head", hidden, false);
                    return Tensor.concatenate(&.{ hidden, logits }, 2).reshape(.{ count, width + talker_vocab });
                },
                .predictor => {
                    const embedded = input.reshape(.{ 1, count, width });
                    const projected = if (width == predictor_width) embedded else qwen.linear(weights, "small_to_mtp_projection", embedded, true);
                    const hidden = backbone(predictor, weights, projected);
                    var rows: [codebooks - 1]Tensor = undefined;
                    for (&rows, 0..) |*row, i| {
                        const state = hidden.slice(1, .{ .start = @intCast(i + 1), .end = @intCast(i + 2) });
                        const head = parameters[parameters.len - rows.len + i].convert(.f32);
                        row.* = runtime.transformer.linearParameters(state, head, null).reshape(.{ 1, codec_vocab });
                    }
                    return Tensor.concatenate(&rows, 0);
                },
            }
        }
    };
}
