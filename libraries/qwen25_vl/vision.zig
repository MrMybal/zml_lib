const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const qwen = @import("zml_lib/qwen_decoder");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = zml.Tensor;

const depth = 32;
const width = 1280;
const heads = 16;
const head_width = width / heads;
const expansion = 3420;
pub const merge_size = 4;
pub const output_width = 3584;
pub const max_patches = 2048;
const patch_width = 3 * 2 * 14 * 14;
const metadata_columns = 4;
pub const input_width = metadata_columns + patch_width;

fn parameter(name: []const u8, shape: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(shape, .f16) };
}

fn linearParams(comptime name: []const u8, out: i64, in: i64) [2]l.Parameter {
    return .{ parameter(name ++ ".weight", .{ out, in }), parameter(name ++ ".bias", .{out}) };
}

pub const schema = blk: {
    @setEvalBranchQuota(100_000);
    var params: [6 + depth * 12]l.Parameter = undefined;
    params[0..6].* = [_]l.Parameter{
        parameter("visual.patch_embed.proj.weight", .{ width, 3, 2, 14, 14 }),
        parameter("visual.merger.ln_q.weight", .{width}),
    } ++ linearParams("visual.merger.mlp.0", merge_size * width, merge_size * width) ++
        linearParams("visual.merger.mlp.2", output_width, merge_size * width);
    for (0..depth) |i| {
        const base = std.fmt.comptimePrint("visual.blocks.{d}", .{i});
        params[6 + i * 12 ..][0..12].* = [_]l.Parameter{
            parameter(base ++ ".norm1.weight", .{width}), parameter(base ++ ".norm2.weight", .{width}),
        } ++ linearParams(base ++ ".attn.qkv", 3 * width, width) ++
            linearParams(base ++ ".attn.proj", width, width) ++
            linearParams(base ++ ".mlp.gate_proj", expansion, width) ++
            linearParams(base ++ ".mlp.up_proj", expansion, width) ++
            linearParams(base ++ ".mlp.down_proj", width, expansion);
    }
    break :blk params;
};

fn angles(input: Tensor) Tensor {
    const quadrant = head_width / 4;
    const exponent = Tensor.iota(.init(.{quadrant}, .i32), 0).convert(.f32)
        .scale(1 / @as(f32, @floatFromInt(quadrant)));
    const base = Tensor.scalar(@as(f32, 10_000), .f32).broad(exponent.shape());
    const frequency = Tensor.scalar(1, .f32).broad(exponent.shape()).div(base.pow(exponent));
    const shape: zml.Shape = .init(.{ input.dim(0), quadrant }, .f32);
    const row = input.slice(1, .single(0)).broadcast(shape, &.{0}).mul(frequency.broadcast(shape, &.{1}));
    const col = input.slice(1, .single(1)).broadcast(shape, &.{0}).mul(frequency.broadcast(shape, &.{1}));
    return Tensor.concatenate(&.{ row, col, row, col }, 1);
}

fn attentionMask(groups: Tensor) Tensor {
    const length = groups.dim(0);
    const shape: zml.Shape = .init(.{ length, length }, .f32);
    const same = groups.broadcast(shape, &.{0}).cmp(.EQ, groups.broadcast(shape, &.{1}));
    return same.select(Tensor.scalar(0, .f32).broad(shape), Tensor.scalar(-std.math.inf(f32), .f32).broad(shape));
}

fn block(w: l.Weights, x: Tensor, mask: Tensor, sin: Tensor, cos: Tensor, index: usize) Tensor {
    var prefix: [48]u8 = undefined;
    var key: [80]u8 = undefined;
    const name = l.name(&prefix, "visual.blocks.{d}", .{index});
    const norm = qwen.rms(x, w.get(l.name(&key, "{s}.norm1", .{name}), ".weight"));
    const qkv = qwen.linear(w, l.name(&key, "{s}.attn.qkv", .{name}), norm, true);
    const q = qwen.rotate(t.splitHeads(qkv.slice(2, .{ .end = width }), heads), sin, cos);
    const k = qwen.rotate(t.splitHeads(qkv.slice(2, .{ .start = width, .end = 2 * width }), heads), sin, cos);
    const v = t.splitHeads(qkv.slice(2, .{ .start = 2 * width }), heads);
    const scores = qwen.scoresAfterDot(q, k);
    const attended = t.attend(scores.add(mask.broadcast(scores.shape(), &.{ 2, 3 })), v);
    const after = x.add(qwen.linear(w, l.name(&key, "{s}.attn.proj", .{name}), attended, true));
    const ff = qwen.rms(after, w.get(l.name(&key, "{s}.norm2", .{name}), ".weight"));
    const gate = qwen.linear(w, l.name(&key, "{s}.mlp.gate_proj", .{name}), ff, true).silu();
    const up = qwen.linear(w, l.name(&key, "{s}.mlp.up_proj", .{name}), ff, true);
    return after.add(qwen.linear(w, l.name(&key, "{s}.mlp.down_proj", .{name}), gate.mul(up), true));
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const patches = input.dim(0);
    // Projection has stride equal to the full patch kernel, so each patch is an independent dot product.
    const pixels = input.slice(1, .{ .start = metadata_columns }).reshape(.{ 1, patches, patch_width });
    const projection = w.get("visual.patch_embed.proj", ".weight").convert(.f32).reshape(.{ width, patch_width });
    var x = t.linearParameters(pixels, projection, null);
    const phase = angles(input);
    const sin = phase.sin();
    const cos = phase.cos();
    const window_mask = attentionMask(input.slice(1, .single(2)));
    const frame_mask = attentionMask(input.slice(1, .single(3)));
    for (0..depth) |index| {
        const mask = if ((index + 1) % 8 == 0) frame_mask else window_mask;
        x = block(w, x, mask, sin, cos, index);
    }
    const normalized = qwen.rms(x, w.get("visual.merger.ln_q", ".weight"));
    const grouped = normalized.reshape(.{ 1, @divExact(patches, merge_size), merge_size * width });
    const hidden = t.gelu(qwen.linear(w, "visual.merger.mlp.0", grouped, true));
    return qwen.linear(w, "visual.merger.mlp.2", hidden, true);
}

pub const Encoder = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Encoder {
        return .{ .graph = try .loadCheckpoint(session, &schema, path, .f16) };
    }

    pub fn deinit(self: *Encoder) void {
        self.graph.deinit();
    }

    pub fn encode(self: *Encoder, count: usize, input: []const f32, output: []f32) !u32 {
        if (count < merge_size or count > max_patches or count % merge_size != 0 or input.len != count * input_width)
            return error.InvalidPatchBuffer;
        for (0..count) |index| {
            const row = input[index * input_width ..][0..input_width];
            for (row[0..metadata_columns]) |value| {
                if (!std.math.isFinite(value) or value < 0 or value >= max_patches or @trunc(value) != value)
                    return error.InvalidPatchPosition;
            }
            for (row[metadata_columns..]) |value| if (!std.math.isFinite(value)) return error.InvalidPixel;
        }
        return self.graph.infer(forward, .init(.{ count, input_width }, .f32), .init(.{ 1, count / merge_size, output_width }, .f32), input, output);
    }
};
