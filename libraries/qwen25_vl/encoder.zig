const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const qwen = @import("zml_lib/qwen_decoder");
const Tensor = zml.Tensor;

pub const vision = @import("vision.zig");
pub const max_tokens = 512;
const position_limit: f32 = 128_000;
const fields = 5;

const Config = struct {
    text: qwen.Profile,
    sections: [3]i64,
};

pub const Profile = enum {
    qwen25_vl_7b,
    fixture,

    fn config(profile: Profile) Config {
        return switch (profile) {
            .qwen25_vl_7b => .{ .sections = .{ 16, 24, 24 }, .text = .{
                .vocab = 152_064,
                .layers = 28,
                .width = 3584,
                .mlp = 18944,
                .heads = 28,
                .kv_heads = 4,
                .head_width = 128,
                .has_bias = true,
                .head_norm = false,
                .separate_output = false,
                .scale_attention_after_dot = true,
                .eos_ids = &.{},
            } },
            .fixture => .{ .sections = .{ 2, 1, 1 }, .text = .{
                .vocab = 64,
                .layers = 2,
                .width = 32,
                .mlp = 48,
                .heads = 4,
                .kv_heads = 2,
                .head_width = 8,
                .has_bias = true,
                .head_norm = false,
                .separate_output = false,
                .scale_attention_after_dot = true,
                .eos_ids = &.{},
            } },
        };
    }
};

fn angles(comptime cfg: Config, input: Tensor) Tensor {
    const half = @divExact(cfg.text.head_width, 2);
    const exponent = Tensor.iota(.init(.{half}, .i32), 0).convert(.f32)
        .scale(2 / @as(f32, @floatFromInt(cfg.text.head_width)));
    const base = Tensor.scalar(@as(f32, 1_000_000), .f32).broad(exponent.shape());
    const frequencies = Tensor.scalar(1, .f32).broad(exponent.shape()).div(base.pow(exponent));
    var phases: [3]Tensor = undefined;
    var start: i64 = 0;
    for (cfg.sections, &phases, 1..) |size, *phase, axis| {
        const shape: zml.Shape = .init(.{ input.dim(0), size }, .f32);
        const position = input.slice(1, .single(@intCast(axis)));
        phase.* = position.broadcast(shape, &.{0}).mul(frequencies.slice(0, .{ .start = start, .end = start + size })
            .broadcast(shape, &.{1}));
        start += size;
    }
    const merged = Tensor.concatenate(&phases, 1);
    return Tensor.concatenate(&.{ merged, merged }, 1);
}

fn inputMask(input: Tensor) Tensor {
    const length = input.dim(0);
    const coordinates: zml.Shape = .init(.{ length, length }, .i32);
    const after = Tensor.iota(coordinates, 1).cmp(.GT, Tensor.iota(coordinates, 0));
    const hidden = input.slice(1, .single(4)).cmp(.EQ, Tensor.scalar(0, .f32));
    // The finite minimum also gives a defined softmax for fully masked padding rows.
    const blocked = Tensor.scalar(-std.math.floatMax(f32), .f32);
    const key_mask = hidden.select(blocked.broad(.init(.{length}, .f32)), Tensor.scalar(0, .f32).broad(.init(.{length}, .f32)));
    const shape = coordinates.withDtype(.f32);
    return after.select(blocked.broad(shape), key_mask.broadcast(shape, &.{1}));
}

pub fn Encoder(comptime profile: Profile) type {
    const cfg = profile.config();
    return struct {
        const Self = @This();
        pub const schema = qwen.schema(cfg.text, .f16);
        pub const width: usize = @intCast(cfg.text.width);
        pub const input_width = fields + width;
        graph: runtime.TensorModel,

        pub fn load(session: *const runtime.Session, path: []const u8) !Self {
            return .{ .graph = try .loadCheckpoint(session, &schema, path, .f16) };
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit();
        }

        fn forward(parameters: []const Tensor, input: Tensor) Tensor {
            const weights: runtime.layers.Weights = .{ .schema = &schema, .tensors = parameters };
            const ids = input.slice(1, .single(0));
            const visual = ids.cmp(.LT, Tensor.scalar(0, .f32));
            const safe_ids = visual.select(Tensor.scalar(0, .f32).broad(ids.shape()), ids).convert(.i32);
            const text = qwen.tokenEmbeddings(weights.get("model.embed_tokens", ".weight"), safe_ids);
            const injected = input.slice(1, .{ .start = fields }).reshape(text.shape());
            const tokens = visual.broadcast(text.shape(), &.{1}).select(injected, text);
            const phase = angles(cfg, input);
            return qwen.layersForward(cfg.text, weights, tokens, inputMask(input), phase.sin(), phase.cos());
        }

        pub fn encode(self: *Self, count: usize, input: []const f32, output: []f32) !u32 {
            if (count == 0 or count > max_tokens or input.len != count * input_width)
                return error.InvalidTokenBuffer;
            var visible = false;
            for (0..count) |index| {
                const row = input[index * input_width ..][0..input_width];
                const id = row[0];
                if (!std.math.isFinite(id) or @trunc(id) != id or id < -1 or id >= @as(f32, @floatFromInt(cfg.text.vocab)))
                    return error.InvalidToken;
                for (row[1..4]) |position| {
                    if (!std.math.isFinite(position) or position < 0 or position >= position_limit or @trunc(position) != position)
                        return error.InvalidPosition;
                }
                if (row[4] != 0 and row[4] != 1) return error.InvalidAttentionMask;
                if (id == -1) {
                    for (row[fields..]) |value| if (!std.math.isFinite(value)) return error.InvalidEmbedding;
                }
                visible = visible or row[4] == 1;
            }
            if (!visible) return error.InvalidAttentionMask;
            return self.graph.infer(forward, .init(.{ count, input_width }, .f32), .init(.{ 1, count, width }, .f32), input, output);
        }
    };
}
