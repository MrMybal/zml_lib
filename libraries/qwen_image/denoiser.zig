const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const cuda_math = @import("cuda_math.zig");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = zml.Tensor;

pub const channels = 64;
const max_dense_image_tokens = 4096;
pub const max_image_tokens = 8192;
pub const max_text_tokens = 512;
const time_width = 256;
const eps: f32 = 1e-6;
const input_parameters = 9;
const top_parameters = 13;
const block_parameters = 32;

const Config = struct {
    width: i64,
    heads: i64,
    text: i64,
    axes: [3]i64,
    blocks: usize,
    ordered_first_qk: bool = false,
    center_first_scores: bool = false,
    align_inputs: bool = false,
    match_first_attention: bool = false,
    match_l4_first_block: bool = false,
};
pub const Profile = enum {
    edit_2511,
    fixture,
    edit_2511_cuda_qk,
    study_centered_scores,
    study_aligned_inputs,
    study_aligned_centered,
    study_ordered_attention,

    fn config(profile: Profile) Config {
        return switch (profile) {
            .edit_2511 => .{ .width = 3072, .heads = 24, .text = 3584, .axes = .{ 16, 56, 56 }, .blocks = 60 },
            .fixture => .{ .width = 48, .heads = 3, .text = 32, .axes = .{ 4, 6, 6 }, .blocks = 2 },
            .edit_2511_cuda_qk, .study_centered_scores, .study_aligned_inputs, .study_aligned_centered, .study_ordered_attention => blk: {
                var cfg = Profile.edit_2511.config();
                cfg.ordered_first_qk = true;
                cfg.center_first_scores = profile == .study_centered_scores or profile == .study_aligned_centered;
                cfg.align_inputs = profile == .study_aligned_inputs or profile == .study_aligned_centered or profile == .study_ordered_attention;
                cfg.match_first_attention = profile == .study_ordered_attention;
                break :blk cfg;
            },
        };
    }
};

pub const Geometry = struct {
    target: usize,
    image: usize,
    text: usize,

    pub fn valid(self: Geometry) bool {
        return self.target != 0 and self.target <= self.image and self.image <= max_image_tokens and
            self.text != 0 and self.text <= max_text_tokens;
    }
};

fn param(name: []const u8, dims: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn pair(comptime prefix: []const u8, outputs: i64, inputs: i64) [2]l.Parameter {
    return .{ param(prefix ++ ".weight", .{ outputs, inputs }), param(prefix ++ ".bias", .{outputs}) };
}

fn parameters(comptime cfg: Config) [top_parameters + cfg.blocks * block_parameters]l.Parameter {
    @setEvalBranchQuota(500_000);
    const d = cfg.width;
    const head = @divExact(d, cfg.heads);
    var params: [top_parameters + cfg.blocks * block_parameters]l.Parameter = undefined;
    params[0..top_parameters].* = pair("img_in", d, channels) ++ pair("txt_in", d, cfg.text) ++
        pair("time_text_embed.timestep_embedder.linear_1", d, time_width) ++
        pair("time_text_embed.timestep_embedder.linear_2", d, d) ++
        [_]l.Parameter{param("txt_norm.weight", .{cfg.text})} ++ pair("norm_out.linear", 2 * d, d) ++ pair("proj_out", channels, d);
    for (0..cfg.blocks) |i| {
        const name = std.fmt.comptimePrint("transformer_blocks.{d}.", .{i});
        const attn = name ++ "attn.";
        params[top_parameters + block_parameters * i ..][0..block_parameters].* =
            pair(name ++ "img_mod.1", 6 * d, d) ++ pair(name ++ "txt_mod.1", 6 * d, d) ++
            pair(attn ++ "to_q", d, d) ++ pair(attn ++ "to_k", d, d) ++ pair(attn ++ "to_v", d, d) ++
            pair(attn ++ "add_q_proj", d, d) ++ pair(attn ++ "add_k_proj", d, d) ++ pair(attn ++ "add_v_proj", d, d) ++
            pair(attn ++ "to_out.0", d, d) ++ pair(attn ++ "to_add_out", d, d) ++ [_]l.Parameter{
                param(attn ++ "norm_q.weight", .{head}),       param(attn ++ "norm_k.weight", .{head}),
                param(attn ++ "norm_added_q.weight", .{head}), param(attn ++ "norm_added_k.weight", .{head}),
            } ++ pair(name ++ "img_mlp.net.0.proj", 4 * d, d) ++ pair(name ++ "img_mlp.net.2", d, 4 * d) ++
            pair(name ++ "txt_mlp.net.0.proj", 4 * d, d) ++ pair(name ++ "txt_mlp.net.2", d, 4 * d);
    }
    return params;
}

const Weights = struct {
    base: l.Weights,
    scales: []const f32,
    projection_splits: i64 = 3,
    exact_cuda: bool = false,

    fn linearWeight(w: Weights, name: []const u8) Tensor {
        for (w.base.schema, w.base.tensors, w.scales) |parameter, weight, scale| {
            if (parameter.name.len == name.len + ".weight".len and
                std.mem.startsWith(u8, parameter.name, name) and std.mem.endsWith(u8, parameter.name, ".weight"))
            {
                return t.decodeE4M3(weight, scale);
            }
        }
        std.debug.panic("Missing Qwen Image matrix {s}", .{name});
    }

    fn linear(w: Weights, name: []const u8, x: Tensor) Tensor {
        const weight = w.linearWeight(name);
        const bias = w.base.get(name, ".bias");
        return if (w.exact_cuda) cuda_math.linear(x, weight, bias) else t.linearParameters(x, weight, bias);
    }

    fn orderedProjection(w: Weights, name: []const u8, x: Tensor) Tensor {
        return w.orderedLinear(name, x, w.projection_splits);
    }

    fn orderedLinear(w: Weights, name: []const u8, x: Tensor, splits: i64) Tensor {
        const weight = w.linearWeight(name);
        const width = weight.dim(1);
        const cols = weight.dim(0);
        const rows = @divExact(@as(i64, @intCast(x.shape().count())), width);
        const runs = @divExact(width, splits * 32);
        // Within each K partition, four lanes interleave runs of eight values.
        const lhs = x.reshape(.{ rows, splits, runs, 4, 8 }).transpose(.{ 1, 3, 0, 2, 4 }).reshape(.{ splits, 4, rows, runs * 8 });
        const rhs = weight.reshape(.{ cols, splits, runs, 4, 8 }).transpose(.{ 1, 3, 0, 2, 4 }).reshape(.{ splits, 4, cols, runs * 8 });
        const parts = orderedDot(lhs, rhs);
        const balanced = parts.slice(1, .single(0)).add(parts.slice(1, .single(1)))
            .add(parts.slice(1, .single(2)).add(parts.slice(1, .single(3))));
        var sum = balanced.slice(0, .single(0));
        for (1..@intCast(splits)) |i| sum = sum.add(balanced.slice(0, .single(@intCast(i))));
        return l.biasAdd(sum, w.base.get(name, ".bias"), 1).reshape(x.shape().setDim(x.rank() - 1, cols));
    }

    fn orderedVector(w: Weights, name: []const u8, x: Tensor) Tensor {
        const weight = w.linearWeight(name);
        const cols = weight.dim(0);
        const width = weight.dim(1);
        const values = @divExact(width, 16);
        const lhs = x.reshape(.{ 1, 1, values, 16 }).transpose(.{ 0, 3, 1, 2 });
        const rhs = weight.reshape(.{ 1, cols, values, 16 }).transpose(.{ 0, 3, 1, 2 });
        var sum = orderedDot(lhs, rhs).slice(0, .single(0));
        for ([_]i64{ 8, 4, 2, 1 }) |n| sum = sum.slice(0, .{ .end = n }).add(sum.slice(0, .{ .start = n }));
        return l.biasAdd(sum.reshape(.{ 1, cols }), w.base.get(name, ".bias"), 1);
    }
};

fn rms(x: Tensor, gain: Tensor) Tensor {
    const axis = x.rank() - 1;
    const norm = x.mul(x).mean(axis).addConstant(eps).rsqrt();
    return x.mul(norm.broad(x.shape())).mul(gain.broadcast(x.shape(), &.{axis}));
}

fn fusedProductAdd(accumulator: Tensor, lhs: Tensor, rhs: Tensor) Tensor {
    // Widen the product and its sum, then round once to FP32 before the next term.
    return lhs.convert(.f64).mul(rhs.convert(.f64)).add(accumulator.convert(.f64)).convert(.f32);
}

const ProjectionAccumulation = struct {
    wide: bool = false,
    pub const State = struct { index: Tensor, input: Tensor, weight: Tensor, sum: Tensor };

    pub fn cond(_: @This(), state: State) Tensor {
        return state.index.cmp(.LT, Tensor.scalar(state.input.dim(3), .i32));
    }

    pub fn body(self: @This(), state: State) State {
        const input = state.input.slice(3, .dyn(state.index, 8));
        const weight = state.weight.slice(3, .dyn(state.index, 8));
        var sum = state.sum;
        for (0..8) |i| {
            const lhs = input.slice(3, .single(@intCast(i))).broadcast(sum.shape(), &.{ 0, 1, 2 });
            const rhs = weight.slice(3, .single(@intCast(i))).broadcast(sum.shape(), &.{ 0, 1, 3 });
            sum = if (self.wide) sum.add(lhs.convert(.f64).mul(rhs.convert(.f64))) else fusedProductAdd(sum, lhs, rhs);
        }
        return .{ .index = state.index.addConstant(@as(i32, 8)), .input = state.input, .weight = state.weight, .sum = sum };
    }
};

fn orderedDot(lhs: Tensor, rhs: Tensor) Tensor {
    return zml.ops.while_(ProjectionAccumulation, .{}, .{
        .index = Tensor.scalar(0, .i32),
        .input = lhs,
        .weight = rhs,
        .sum = Tensor.scalar(0, .f32).broad(.init(.{ lhs.dim(0), lhs.dim(1), lhs.dim(2), rhs.dim(2) }, .f32)),
    }).sum;
}

fn centeredScores(q: Tensor, k: Tensor, mask: Tensor) Tensor {
    const shape: zml.Shape = .init(.{ q.dim(0), q.dim(1), q.dim(2), k.dim(2) }, .f64);
    const sums = zml.ops.while_(ProjectionAccumulation, .{ .wide = true }, .{
        .index = Tensor.scalar(0, .i32),
        .input = q,
        .weight = k,
        .sum = Tensor.scalar(0, .f64).broad(shape),
    }).sum;
    const scores = sums.scale(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(3))))).add(mask.convert(.f64).broadcast(shape, &.{3}));
    var maximum = scores;
    var n = maximum.dim(3);
    while (n > 1) {
        if (@mod(n, 2) != 0) {
            const pad = Tensor.scalar(-std.math.inf(f64), .f64).broad(shape.setDim(3, 1));
            maximum = Tensor.concatenate(&.{ maximum, pad }, 3);
            n += 1;
        }
        n = @divExact(n, 2);
        maximum = maximum.slice(3, .{ .end = n }).maximum(maximum.slice(3, .{ .start = n }));
    }
    return scores.sub(maximum.broad(shape)).convert(.f32);
}

fn rms128(x: Tensor, gain: Tensor) Tensor {
    // CUDA RMSNorm for a 128-wide head accumulates four squares per lane, then halves the lanes.
    const batch, const heads, const tokens, const width = x.dims()[0..4].*;
    std.debug.assert(width == 128);
    const groups = x.reshape(.{ batch, heads, tokens, 32, 4 });
    const first = groups.slice(4, .single(0));
    var partial = first.mul(first);
    for (1..4) |i| {
        const value = groups.slice(4, .single(@intCast(i)));
        partial = fusedProductAdd(partial, value, value);
    }
    var count: i64 = 32;
    while (count > 1) {
        count = @divExact(count, 2);
        partial = partial.slice(3, .{ .end = count }).add(partial.slice(3, .{ .start = count }));
    }
    const inv = partial.scale(@as(f32, 1.0 / 128.0)).addConstant(eps).rsqrt();
    return x.mul(inv.broad(x.shape())).mul(gain.broadcast(x.shape(), &.{3}));
}

fn rms3584(x: Tensor, gain: Tensor) Tensor {
    const batch, const tokens, const width = x.dims()[0..3].*;
    std.debug.assert(width == 3584);
    const groups = x.reshape(.{ batch, tokens, 7, 4, 32, 4 });
    var sum = Tensor.scalar(0, .f32).broad(.init(.{ batch, tokens, 4, 32 }, .f32));
    for (0..28) |i| {
        const v = groups.slice(2, .single(@intCast(i / 4))).slice(4, .single(@intCast(i % 4)));
        sum = fusedProductAdd(sum, v, v);
    }
    for ([_]i64{ 16, 8, 4, 2, 1 }) |n| sum = sum.slice(3, .{ .end = n }).add(sum.slice(3, .{ .start = n }));
    for ([_]i64{ 2, 1 }) |n| sum = sum.slice(2, .{ .end = n }).add(sum.slice(2, .{ .start = n }));
    const mean = sum.convert(.f64).scale(@as(f64, 1.0 / 3584.0)).convert(.f32);
    const inv = mean.addConstant(eps).rsqrt().reshape(.{ batch, tokens, 1 });
    return x.mul(inv.broad(x.shape())).mul(gain.broadcast(x.shape(), &.{2}));
}

const Moments = struct {
    mean: Tensor,
    sum: Tensor,

    fn merge(self: Moments, axis: i64, extent: i64, population: f32) Moments {
        const left = self.mean.slice(axis, .{ .end = extent });
        const right = self.mean.slice(axis, .{ .start = extent });
        const sum = self.sum.slice(axis, .{ .end = extent }).add(self.sum.slice(axis, .{ .start = extent }));
        const difference = right.sub(left);
        const correction = difference.mul(difference).scale(population);
        return .{
            .mean = left.scale(@as(f32, 0.5)).add(right.scale(@as(f32, 0.5))),
            .sum = fusedProductAdd(sum, correction, Tensor.scalar(0.5, .f32).broad(correction.shape())),
        };
    }
};

fn layerNorm3072(x: Tensor) Tensor {
    const batch, const tokens, const width = x.dims()[0..3].*;
    std.debug.assert(width == 3072);
    const groups = x.reshape(.{ batch, tokens, 6, 4, 32, 4 });
    const shape: zml.Shape = .init(.{ batch, tokens, 4, 32 }, .f32);
    const zero = Tensor.scalar(0, .f32).broad(shape);
    var moments: Moments = .{ .mean = zero, .sum = zero };
    for (0..24) |i| {
        const value = groups.slice(2, .single(@intCast(i / 4))).slice(4, .single(@intCast(i % 4)));
        const difference = value.sub(moments.mean);
        const reciprocal = Tensor.scalar(1.0 / @as(f32, @floatFromInt(i + 1)), .f32).broad(shape);
        const mean = fusedProductAdd(moments.mean, difference, reciprocal);
        moments = .{ .mean = mean, .sum = fusedProductAdd(moments.sum, difference, value.sub(mean)) };
    }
    var count: f32 = 24;
    for ([_]i64{ 16, 8, 4, 2, 1 }) |half| {
        moments = moments.merge(3, half, count);
        count *= 2;
    }
    for ([_]i64{ 2, 1 }) |half| {
        moments = moments.merge(2, half, count);
        count *= 2;
    }
    const denom = moments.sum.convert(.f64).scale(@as(f64, 1.0 / 3072.0)).convert(.f32);
    const inv = denom.addConstant(eps).rsqrt().reshape(.{ batch, tokens, 1 });
    return x.sub(moments.mean.reshape(.{ batch, tokens, 1 }).broad(x.shape())).mul(inv.broad(x.shape()));
}
fn divideSilu(x: Tensor) Tensor {
    return x.div(x.scale(@as(f32, -1)).exp().addConstant(@as(f32, 1)));
}

fn timeEmbedding(w: Weights, time: Tensor, ordered: bool) Tensor {
    if (w.exact_cuda) {
        const hidden = w.linear("time_text_embed.timestep_embedder.linear_1", cuda_math.phase(time));
        return cuda_math.silu(w.linear("time_text_embed.timestep_embedder.linear_2", cuda_math.silu(hidden)));
    }
    const half = time_width / 2;
    const frequency = Tensor.iota(.init(.{half}, .i32), 0).convert(.f32)
        .scale(@as(f32, -@log(10000.0)) / @as(f32, half)).exp();
    const shape: zml.Shape = .init(.{ 2, half }, .f32);
    const times = Tensor.concatenate(&.{ time.reshape(.{1}), Tensor.scalar(0, .f32).reshape(.{1}) }, 0);
    const phase = times.broadcast(shape, &.{0}).mul(frequency.broadcast(shape, &.{1})).scale(@as(f32, 1000));
    const embedding = Tensor.concatenate(&.{ phase.cos(), phase.sin() }, 1);
    const hidden = w.linear("time_text_embed.timestep_embedder.linear_1", embedding);
    const name = "time_text_embed.timestep_embedder.linear_2";
    return if (ordered) divideSilu(w.orderedLinear(name, divideSilu(hidden), 3)) else w.linear(name, hidden.silu()).silu();
}

const Rotary = struct { cos: Tensor, sin: Tensor };

fn rotation(cfg: Config, coordinates: Tensor) Rotary {
    const length = coordinates.dim(0);
    // ComfyUI evaluates the frequencies and angles in float64, then casts the rotations to float32.
    const positions = coordinates.convert(.f64);
    var axes: [3]Tensor = undefined;
    for (cfg.axes, &axes, 0..) |axis, *values, i| {
        const width = @divExact(axis, 2);
        const exponent = Tensor.iota(.init(.{width}, .i32), 0).convert(.f64)
            .scale(2 / @as(f64, @floatFromInt(axis)));
        const frequency = Tensor.scalar(1, .f64).broad(exponent.shape()).div(
            Tensor.scalar(@as(f64, 10_000), .f64).broad(exponent.shape()).pow(exponent),
        );
        const shape: zml.Shape = .init(.{ length, width }, .f64);
        values.* = positions.slice(1, .single(@intCast(i))).broadcast(shape, &.{0})
            .mul(frequency.broadcast(shape, &.{1}));
    }
    const phase = Tensor.concatenate(&axes, 1);
    return .{ .cos = phase.cos().convert(.f32), .sin = phase.sin().convert(.f32) };
}

fn rotate(x: Tensor, r: Rotary) Tensor {
    const shape = .{ x.dim(0), x.dim(1), x.dim(2), @divExact(x.dim(3), 2) };
    const pairs = x.reshape(shape ++ .{2});
    const even = pairs.slice(4, .single(0));
    const odd = pairs.slice(4, .single(1));
    const c = r.cos.broadcast(even.shape(), &.{ 2, 3 });
    const s = r.sin.broadcast(even.shape(), &.{ 2, 3 });
    return Tensor.concatenate(&.{
        even.mul(c).sub(odd.mul(s)).reshape(shape ++ .{1}),
        even.mul(s).add(odd.mul(c)).reshape(shape ++ .{1}),
    }, 4).reshape(x.shape());
}

const QKV = struct { q: Tensor, k: Tensor, v: Tensor };

fn project(w: Weights, prefix: []const u8, x: Tensor, heads: i64, ordered: bool, order_values: bool, comptime image: bool) QKV {
    var key: [96]u8 = undefined;
    const q = projectHead(w, l.name(&key, "{s}." ++ (if (image) "to_q" else "add_q_proj"), .{prefix}), x, heads, ordered);
    const k = projectHead(w, l.name(&key, "{s}." ++ (if (image) "to_k" else "add_k_proj"), .{prefix}), x, heads, ordered);
    const v = projectHead(w, l.name(&key, "{s}." ++ (if (image) "to_v" else "add_v_proj"), .{prefix}), x, heads, order_values);
    const q_gain = w.base.get(l.name(&key, "{s}." ++ (if (image) "norm_q" else "norm_added_q"), .{prefix}), ".weight");
    const k_gain = w.base.get(l.name(&key, "{s}." ++ (if (image) "norm_k" else "norm_added_k"), .{prefix}), ".weight");
    return .{
        .q = if (w.exact_cuda) cuda_math.rms(q, q_gain) else if (ordered) rms128(q, q_gain) else rms(q, q_gain),
        .k = if (w.exact_cuda) cuda_math.rms(k, k_gain) else if (ordered) rms128(k, k_gain) else rms(k, k_gain),
        .v = v,
    };
}

fn projectHead(w: Weights, name: []const u8, x: Tensor, heads: i64, ordered: bool) Tensor {
    const projection = if (ordered) w.orderedProjection(name, x) else w.linear(name, x);
    return t.splitHeads(projection, heads);
}

const Modulation = struct { shift: Tensor, scale: Tensor, gate: Tensor };

fn modulation(params: Tensor, offset: i64, width: i64) Modulation {
    return .{
        .shift = params.slice(2, .{ .start = offset, .end = offset + width }),
        .scale = params.slice(2, .{ .start = offset + width, .end = offset + 2 * width }),
        .gate = params.slice(2, .{ .start = offset + 2 * width, .end = offset + 3 * width }),
    };
}

fn modulate(x: Tensor, m: Modulation, exact_cuda: bool) Tensor {
    return modulateNormalized(if (exact_cuda) cuda_math.layerNorm(x) else t.normAffineFree(x, eps), m, false);
}

fn modulateNormalized(x: Tensor, m: Modulation, fused: bool) Tensor {
    const scale = m.scale.addConstant(@as(f32, 1));
    return if (fused) fusedProductAdd(m.shift, x, scale) else x.mul(scale).add(m.shift);
}

fn feedForward(w: Weights, prefix: []const u8, x: Tensor) Tensor {
    var key: [96]u8 = undefined;
    const expanded = w.linear(l.name(&key, "{s}.net.0.proj", .{prefix}), x);
    return w.linear(l.name(&key, "{s}.net.2", .{prefix}), if (w.exact_cuda) cuda_math.gelu(expanded) else t.geluTanh(expanded));
}

const Streams = struct { image: Tensor, text: Tensor };

pub const ProjectionProbe = struct {
    const cfg = Profile.edit_2511.config();
    const prefix = "transformer_blocks.0.attn.";
    const names = .{ "to_q", "to_k", "add_q_proj", "add_k_proj" };
    const norms = .{ "norm_q", "norm_k", "norm_added_q", "norm_added_k" };
    pub const schema = pair(prefix ++ names[0], cfg.width, cfg.width) ++
        pair(prefix ++ names[1], cfg.width, cfg.width) ++
        pair(prefix ++ names[2], cfg.width, cfg.width) ++
        pair(prefix ++ names[3], cfg.width, cfg.width) ++ [_]l.Parameter{
        param(prefix ++ norms[0] ++ ".weight", .{128}),
        param(prefix ++ norms[1] ++ ".weight", .{128}),
        param(prefix ++ norms[2] ++ ".weight", .{128}),
        param(prefix ++ norms[3] ++ ".weight", .{128}),
    };
    const stored = schema ++ [_]l.Parameter{param("weight_scales", .{schema.len})};

    pub fn inputCount(g: Geometry) usize {
        return (g.image + g.text) * (5 * @as(usize, cfg.width) + 3);
    }

    pub fn outputCount(g: Geometry) usize {
        return (g.image + g.text) * 6 * @as(usize, cfg.width);
    }

    fn forward(parameters_: []const Tensor, scales: []const f32, g: Geometry, split: bool, input: Tensor) Tensor {
        const w: Weights = .{ .base = .{ .schema = &schema, .tensors = parameters_ }, .scales = scales };
        const ni: i64 = @intCast(g.image);
        const nt: i64 = @intCast(g.text);
        const width = cfg.width;
        const base = (ni + nt) * (5 * width);
        const x_image = input.slice(0, .{ .end = ni * width }).reshape(.{ 1, ni, width });
        const x_text = input.slice(0, .{ .start = ni * width, .end = (ni + nt) * width }).reshape(.{ 1, nt, width });
        const r = rotation(cfg, input.slice(0, .{ .start = base }).reshape(.{ ni + nt, 3 }));
        const r_image: Rotary = .{ .cos = r.cos.slice(0, .{ .start = nt }), .sin = r.sin.slice(0, .{ .start = nt }) };
        const r_text: Rotary = .{ .cos = r.cos.slice(0, .{ .end = nt }), .sin = r.sin.slice(0, .{ .end = nt }) };
        var offset = (ni + nt) * width;
        var out: [12]Tensor = undefined;
        inline for (names, norms, 0..) |linear, norm, i| {
            const x = if (i < 2) x_image else x_text;
            const tokens = if (i < 2) ni else nt;
            const raw = if (split) w.orderedProjection(prefix ++ linear, x) else w.linear(prefix ++ linear, x);
            const raw_end = offset + tokens * width;
            const norm_end = raw_end + tokens * width;
            const reference_raw = input.slice(0, .{ .start = offset, .end = raw_end }).reshape(.{ 1, cfg.heads, tokens, 128 });
            const reference_norm = input.slice(0, .{ .start = raw_end, .end = norm_end }).reshape(reference_raw.shape());
            offset = norm_end;
            const gain = w.base.get(prefix ++ norm, ".weight");
            const rope = if (i < 2) r_image else r_text;
            out[3 * i ..][0..3].* = .{
                raw.flatten(),
                (if (split) rms128(reference_raw, gain) else rms(reference_raw, gain)).flatten(),
                rotate(reference_norm, rope).flatten(),
            };
        }
        return Tensor.concatenate(&out, 0);
    }

    pub fn run(session: *const runtime.Session, path: []const u8, g: Geometry, split: bool, input: []const f32, output: []f32) !u32 {
        const allocator = session.allocator;
        const weights = try runtime.checkpoint.loadMixedE4M3(allocator, session.io, session.platform, stored.len, &stored, l.Parameter.at, path);
        defer weights.deinit(allocator);
        var scales: [schema.len]f32 = undefined;
        const scale_buffer = weights.buffers[schema.len];
        try scale_buffer.toSlice(session.io, .init(scale_buffer.shape(), std.mem.asBytes(&scales)));
        for (scales, weights.tensors[0..schema.len]) |scale, value| {
            if (scale <= 0 or !std.math.isFinite(scale * 448) or (value.dtype() != .u8 and scale != 1)) return error.InvalidFP8Scale;
        }
        const input_shape: zml.Shape = .init(.{input.len}, .f32);
        var graph = try runtime.Compiled.init(session, forward, .{ weights.tensors[0..schema.len], &scales, g, split, Tensor.fromShape(input_shape) });
        defer graph.deinit(allocator);
        graph.args.bake(weights.buffers[0..schema.len]);
        var source = try zml.Buffer.fromBytesOpts(session.io, session.platform, input_shape, std.mem.sliceAsBytes(input), .{ .wait = true });
        defer source.deinit();
        try graph.run(.{source});
        var result = graph.results.get(zml.Buffer);
        defer result.deinit();
        const output_shape: zml.Shape = .init(.{output.len}, .f32);
        if (!result.shape().eql(output_shape)) return error.InvalidOutputShape;
        const shards = try session.verifyPlacement(&result);
        try result.toSlice(session.io, .init(output_shape, std.mem.sliceAsBytes(output)));
        return shards;
    }
};

fn block(weights: Weights, prefix: []const u8, x: Streams, condition: Tensor, cfg: Config, g: Geometry, r: Rotary, mask: Tensor, first: bool) Streams {
    var key: [96]u8 = undefined;
    var attention_key: [96]u8 = undefined;
    const match_l4 = first and cfg.match_l4_first_block;
    const exact_cuda = match_l4 and cuda_math.enabled();
    // The observed cuBLAS projection kernel uses a 32-row tile and two K partitions.
    const split_small_projections = match_l4 and !exact_cuda and g.image <= 32 and g.text <= 32;
    var w = weights;
    w.exact_cuda = exact_cuda;
    if (match_l4 and !exact_cuda) w.projection_splits = 2;
    const d = cfg.width;
    const shape: zml.Shape = .init(.{ 1, g.image, 6 * d }, .f32);
    const image_mod_name = l.name(&key, "{s}.img_mod.1", .{prefix});
    const times = if (match_l4 and !exact_cuda) w.orderedProjection(image_mod_name, condition) else w.linear(image_mod_name, condition);
    const target_params = times.slice(0, .single(0)).broadcast(shape, &.{2});
    const refs_params = times.slice(0, .single(1)).broadcast(shape, &.{2});
    const target = Tensor.iota(.init(.{g.image}, .i32), 0).cmp(.LT, Tensor.scalar(@as(i32, @intCast(g.target)), .i32));
    const image_params = target.broadcast(shape, &.{1}).select(target_params, refs_params);
    const text_name = l.name(&key, "{s}.txt_mod.1", .{prefix});
    const text_condition = condition.slice(0, .{ .end = 1 });
    const text_params = (if (first and cfg.align_inputs) w.orderedVector(text_name, text_condition) else w.linear(text_name, text_condition))
        .reshape(.{ 1, 1, 6 * d }).broad(.init(.{ 1, g.text, 6 * d }, .f32));
    const mi = modulation(image_params, 0, d);
    const mt = modulation(text_params, 0, d);
    const attn_prefix = l.name(&attention_key, "{s}.attn", .{prefix});
    const ordered = (first and cfg.align_inputs) or (match_l4 and !exact_cuda);
    const image_input = if (ordered) modulateNormalized(layerNorm3072(x.image), mi, match_l4) else modulate(x.image, mi, exact_cuda);
    const text_input = if (ordered) modulateNormalized(layerNorm3072(x.text), mt, match_l4) else modulate(x.text, mt, exact_cuda);
    const order_attention = first and cfg.match_first_attention;
    const order_qk = (first and cfg.ordered_first_qk) or split_small_projections;
    const order_values = order_attention or split_small_projections;
    const image = project(w, attn_prefix, image_input, cfg.heads, order_qk, order_values, true);
    const text = project(w, attn_prefix, text_input, cfg.heads, order_qk, order_values, false);
    const q = rotate(Tensor.concatenate(&.{ text.q, image.q }, 2), r);
    const k = rotate(Tensor.concatenate(&.{ text.k, image.k }, 2), r);
    const v = Tensor.concatenate(&.{ text.v, image.v }, 2);
    const tiled = g.image > max_dense_image_tokens;
    const attended = if (exact_cuda or tiled) cuda_math.attention(q, k, v, mask, tiled) else blk: {
        const logits = if (first and cfg.center_first_scores)
            centeredScores(q, k, mask)
        else if (order_attention)
            orderedScores(q, k, mask)
        else
            standardScores(q, k, mask, match_l4);
        break :blk if (order_attention) orderedAttend(logits, v) else t.attend(logits, v);
    };
    const text_count: i64 = @intCast(g.text);
    const image_context = attended.slice(1, .{ .start = text_count });
    const text_context = attended.slice(1, .{ .end = text_count });
    const image_name = l.name(&key, "{s}.to_out.0", .{attn_prefix});
    const image_update = if (order_attention) w.orderedProjection(image_name, image_context) else w.linear(image_name, image_context);
    const text_out_name = l.name(&key, "{s}.to_add_out", .{attn_prefix});
    const text_update = if (order_attention) w.orderedProjection(text_out_name, text_context) else w.linear(text_out_name, text_context);
    const image_after = x.image.add(image_update.mul(mi.gate));
    const text_after = x.text.add(text_update.mul(mt.gate));
    const fi = modulation(image_params, 3 * d, d);
    const ft = modulation(text_params, 3 * d, d);
    return .{
        .image = image_after.add(feedForward(w, l.name(&key, "{s}.img_mlp", .{prefix}), modulate(image_after, fi, exact_cuda)).mul(fi.gate)),
        .text = text_after.add(feedForward(w, l.name(&key, "{s}.txt_mlp", .{prefix}), modulate(text_after, ft, exact_cuda)).mul(ft.gate)),
    };
}

fn standardScores(q: Tensor, k: Tensor, mask: Tensor, scale_after_dot: bool) Tensor {
    const logits = if (scale_after_dot) blk: {
        // ComfyUI's sub-quadratic attention scales its accumulated FP32 scores.
        const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(q.dim(3))));
        break :blk q.dotGeneral(k, &.{.{ 3, 3 }}, &.{ .{ 0, 0 }, .{ 1, 1 } }).scale(scale);
    } else t.scores(q, k);
    return logits.add(mask.broadcast(logits.shape(), &.{3}));
}

fn orderedScores(q: Tensor, k: Tensor, mask: Tensor) Tensor {
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(q.dim(3))));
    const scores = orderedDot(q, k).scale(scale);
    return scores.add(mask.broadcast(scores.shape(), &.{3}));
}

fn orderedAttend(scores: Tensor, v: Tensor) Tensor {
    const weights = scores.softmax(3);
    const values = v.transpose(.{ 0, 1, 3, 2 });
    const tail = @mod(-weights.dim(3), 8);
    const a = if (tail == 0) weights else Tensor.concatenate(&.{ weights, Tensor.scalar(0, .f32).broad(weights.shape().setDim(3, tail)) }, 3);
    const b = if (tail == 0) values else Tensor.concatenate(&.{ values, Tensor.scalar(0, .f32).broad(values.shape().setDim(3, tail)) }, 3);
    const y = orderedDot(a, b);
    return y.transpose(.{ 0, 2, 1, 3 }).reshape(.{ y.dim(0), y.dim(2), y.dim(1) * y.dim(3) });
}

pub fn Model(comptime profile: Profile) type {
    const cfg = profile.config();
    return struct {
        const Self = @This();
        pub const schema = parameters(cfg);
        pub const text_width: usize = @intCast(cfg.text);
        pub const blocks = cfg.blocks;
        pub const stored = schema ++ [_]l.Parameter{param("weight_scales", .{schema.len})};
        const Blockwise = struct {
            stages: []runtime.Compiled,
            reader: ?runtime.checkpoint.Reader = null,
        };
        const Execution = union(enum) {
            whole: runtime.Compiled,
            blockwise: Blockwise,
        };
        const Input = struct {
            image: Tensor,
            image_positions: Tensor,
            text: Tensor,
            text_positions: Tensor,
            mask: Tensor,
            time: Tensor,
        };
        const condition_count = 2 * cfg.width;
        session: *const runtime.Session,
        weights: ?runtime.checkpoint.Parameters = null,
        execution: Execution,
        geometry: Geometry,
        trace_blocks: bool,

        pub fn inputCount(g: Geometry) usize {
            return g.image * (channels + 3) + g.text * (text_width + 4) + 1;
        }

        fn streamCount(g: Geometry) usize {
            return (g.image + g.text) * @as(usize, @intCast(cfg.width));
        }

        pub fn traceInputCount(g: Geometry) usize {
            return inputCount(g) + 1 + 2 * @as(usize, @intCast(cfg.width)) + (cfg.blocks + 1) * streamCount(g);
        }

        pub fn traceOutputCount(g: Geometry) usize {
            return g.target * channels + 2 * @as(usize, @intCast(cfg.width)) + (cfg.blocks + 1) * streamCount(g);
        }

        fn flattenStreams(streams: Streams) Tensor {
            return Tensor.concatenate(&.{ streams.image.flatten(), streams.text.flatten() }, 0);
        }

        fn unpackStreams(input: Tensor, offset: i64, g: Geometry) Streams {
            const image_end = offset + @as(i64, @intCast(g.image)) * cfg.width;
            const end = offset + @as(i64, @intCast(streamCount(g)));
            return .{
                .image = input.slice(0, .{ .start = offset, .end = image_end }).reshape(.{ 1, g.image, cfg.width }),
                .text = input.slice(0, .{ .start = image_end, .end = end }).reshape(.{ 1, g.text, cfg.width }),
            };
        }

        fn selectedStreams(input: Tensor, reference_offset: i64, g: Geometry, force: Tensor, streams: Streams) Streams {
            const selected = unpackStreams(input, reference_offset, g);
            return .{
                .image = force.broad(selected.image.shape()).select(selected.image, streams.image),
                .text = force.broad(selected.text.shape()).select(selected.text, streams.text),
            };
        }

        fn parseInput(input: Tensor, g: Geometry) Input {
            const images: i64 = @intCast(g.image);
            const texts: i64 = @intCast(g.text);
            const image_end = images * channels;
            const image_pos_end = image_end + images * 3;
            const text_end = image_pos_end + texts * cfg.text;
            const text_pos_end = text_end + texts * 3;
            const key_mask_shape: zml.Shape = .init(.{g.text}, .f32);
            const masked = input.slice(0, .{ .start = text_pos_end, .end = text_pos_end + texts }).cmp(.EQ, Tensor.scalar(0, .f32));
            const text_mask = masked.select(Tensor.scalar(-std.math.inf(f32), .f32).broad(key_mask_shape), Tensor.scalar(0, .f32).broad(key_mask_shape));
            return .{
                .image = input.slice(0, .{ .end = image_end }).reshape(.{ 1, g.image, channels }),
                .image_positions = input.slice(0, .{ .start = image_end, .end = image_pos_end }).reshape(.{ g.image, 3 }),
                .text = input.slice(0, .{ .start = image_pos_end, .end = text_end }).reshape(.{ 1, g.text, text_width }),
                .text_positions = input.slice(0, .{ .start = text_end, .end = text_pos_end }).reshape(.{ g.text, 3 }),
                .mask = Tensor.concatenate(&.{ text_mask, Tensor.scalar(0, .f32).broad(.init(.{g.image}, .f32)) }, 0),
                .time = input.slice(0, .single(text_pos_end + texts)),
            };
        }

        fn inputStreams(w: Weights, input: Input) Streams {
            const gain = w.base.get("txt_norm", ".weight");
            return .{
                .image = if (cfg.align_inputs) w.orderedLinear("img_in", input.image, 1) else w.linear("img_in", input.image),
                .text = w.linear("txt_in", if (w.exact_cuda) cuda_math.rms(input.text, gain) else if (cfg.align_inputs) rms3584(input.text, gain) else rms(input.text, gain)),
            };
        }

        fn outputVelocity(w: Weights, streams: Streams, condition: Tensor, g: Geometry) Tensor {
            const mod = w.linear("norm_out.linear", condition.slice(0, .{ .end = 1 }));
            const x = streams.image.slice(1, .{ .end = @intCast(g.target) });
            const scale = mod.slice(1, .{ .end = cfg.width }).addConstant(@as(f32, 1)).broadcast(x.shape(), &.{ 0, 2 });
            const shift = mod.slice(1, .{ .start = cfg.width }).broadcast(x.shape(), &.{ 0, 2 });
            return w.linear("proj_out", t.normAffineFree(x, eps).mul(scale).add(shift));
        }

        fn forward(parameters_: []const Tensor, scales: []const f32, g: Geometry, trace_blocks: bool, input: Tensor) Tensor {
            const w: Weights = .{ .base = .{ .schema = &schema, .tensors = parameters_ }, .scales = scales };
            const parsed = parseInput(input, g);
            const condition_zml = timeEmbedding(w, parsed.time, cfg.align_inputs);
            const base: i64 = @intCast(inputCount(g));
            const condition_end = base + 1 + 2 * cfg.width;
            const control = if (trace_blocks) input.slice(0, .single(base)) else undefined;
            const force = if (trace_blocks) control.cmp(.EQ, Tensor.scalar(1, .f32)) else undefined;
            const condition = if (trace_blocks)
                force.broad(condition_zml.shape()).select(
                    input.slice(0, .{ .start = base + 1, .end = condition_end }).reshape(condition_zml.shape()),
                    condition_zml,
                )
            else
                condition_zml;
            const rope = rotation(cfg, Tensor.concatenate(&.{ parsed.text_positions, parsed.image_positions }, 0));
            var streams = inputStreams(w, parsed);
            var trace: [cfg.blocks + 3]Tensor = undefined;
            if (trace_blocks) {
                trace[1] = condition_zml.flatten();
                trace[2] = flattenStreams(streams);
            }
            const stride: i64 = @intCast(streamCount(g));
            var key: [64]u8 = undefined;
            for (0..cfg.blocks) |i| {
                const offset = condition_end + @as(i64, @intCast(i)) * stride;
                const at_block = if (trace_blocks and i != 0) force.select(force, control.cmp(.EQ, Tensor.scalar(@as(f32, @floatFromInt(i + 2)), .f32))) else force;
                const x = if (trace_blocks) selectedStreams(input, offset, g, at_block, streams) else streams;
                streams = block(w, l.name(&key, "transformer_blocks.{d}", .{i}), x, condition, cfg, g, rope, parsed.mask, i == 0);
                if (trace_blocks) trace[i + 3] = flattenStreams(streams);
            }
            if (trace_blocks) {
                const at_head = force.select(force, control.cmp(.EQ, Tensor.scalar(@as(f32, @floatFromInt(cfg.blocks + 2)), .f32)));
                streams = selectedStreams(input, condition_end + @as(i64, @intCast(cfg.blocks)) * stride, g, at_head, streams);
            }
            const velocity = outputVelocity(w, streams, condition, g);
            if (trace_blocks) {
                trace[0] = velocity.flatten();
                return Tensor.concatenate(&trace, 0);
            }
            return velocity;
        }

        fn forwardInput(parameters_: []const Tensor, scales: []const f32, g: Geometry, input: Tensor) Tensor {
            const w: Weights = .{
                .base = .{ .schema = schema[0..input_parameters], .tensors = parameters_ },
                .scales = scales,
                .exact_cuda = profile == .edit_2511 and cuda_math.enabled(),
            };
            const parsed = parseInput(input, g);
            const condition = timeEmbedding(w, parsed.time, cfg.align_inputs);
            return Tensor.concatenate(&.{ condition.flatten(), flattenStreams(inputStreams(w, parsed)) }, 0);
        }

        fn forwardBlock(parameters_: []const Tensor, scales: []const f32, g: Geometry, index: usize, input: Tensor, state: Tensor) Tensor {
            const start = top_parameters + index * block_parameters;
            const w: Weights = .{ .base = .{ .schema = schema[start..][0..block_parameters], .tensors = parameters_ }, .scales = scales };
            const parsed = parseInput(input, g);
            const rope = rotation(cfg, Tensor.concatenate(&.{ parsed.text_positions, parsed.image_positions }, 0));
            const condition = state.slice(0, .{ .end = condition_count });
            var block_config = cfg;
            block_config.match_l4_first_block = profile == .edit_2511;
            var key: [64]u8 = undefined;
            const streams = block(
                w,
                l.name(&key, "transformer_blocks.{d}", .{index}),
                unpackStreams(state, condition_count, g),
                condition.reshape(.{ 2, cfg.width }),
                block_config,
                g,
                rope,
                parsed.mask,
                index == 0,
            );
            return Tensor.concatenate(&.{ condition, flattenStreams(streams) }, 0);
        }

        fn forwardOutput(parameters_: []const Tensor, scales: []const f32, g: Geometry, state: Tensor) Tensor {
            const w: Weights = .{ .base = .{ .schema = schema[input_parameters..top_parameters], .tensors = parameters_ }, .scales = scales };
            const condition = state.slice(0, .{ .end = condition_count }).reshape(.{ 2, cfg.width });
            return outputVelocity(w, unpackStreams(state, condition_count, g), condition, g);
        }

        fn compileBlockwise(session: *const runtime.Session, tensors: []const Tensor, buffers: ?[]zml.Buffer, scales: []const f32, g: Geometry) ![]runtime.Compiled {
            const stages = try session.allocator.alloc(runtime.Compiled, cfg.blocks + 2);
            errdefer session.allocator.free(stages);
            var compiled: usize = 0;
            errdefer for (stages[0..compiled]) |*stage| stage.deinit(session.allocator);
            const input = Tensor.fromShape(.init(.{inputCount(g)}, .f32));
            const state = Tensor.fromShape(.init(.{streamCount(g) + @as(usize, @intCast(condition_count))}, .f32));
            stages[0] = try runtime.Compiled.init(session, forwardInput, .{
                tensors[0..input_parameters], scales[0..input_parameters], g, input,
            });
            if (buffers) |resident| stages[0].args.bake(resident[0..input_parameters]);
            compiled += 1;
            for (0..cfg.blocks) |index| {
                const start = top_parameters + index * block_parameters;
                const end = start + block_parameters;
                const stage = &stages[index + 1];
                stage.* = try runtime.Compiled.init(session, forwardBlock, .{
                    tensors[start..end], scales[start..end], g, index, input, state,
                });
                if (buffers) |resident| stage.args.bake(resident[start..end]);
                compiled += 1;
            }
            const head = &stages[cfg.blocks + 1];
            head.* = try runtime.Compiled.init(session, forwardOutput, .{
                tensors[input_parameters..top_parameters], scales[input_parameters..top_parameters], g, state,
            });
            if (buffers) |resident| head.args.bake(resident[input_parameters..top_parameters]);
            return stages;
        }

        pub fn load(session: *const runtime.Session, path: []const u8, g: Geometry) !Self {
            return loadGraph(session, path, g, false, false);
        }

        pub fn loadBlockwise(session: *const runtime.Session, path: []const u8, g: Geometry) !Self {
            return loadGraph(session, path, g, false, true);
        }

        pub fn loadTrace(session: *const runtime.Session, path: []const u8, g: Geometry) !Self {
            return loadGraph(session, path, g, true, false);
        }

        fn validateScales(scales: []const f32, tensors: []const Tensor) !void {
            var quantized: usize = 0;
            for (scales, tensors[0..schema.len]) |scale, value| {
                if (scale <= 0 or !std.math.isFinite(scale * 448)) return error.InvalidFP8Scale;
                if (value.dtype() == .u8) quantized += 1 else if (scale != 1) return error.InvalidFP8Scale;
            }
            std.log.info("weight_storage=e4m3_scaled quantized={d}", .{quantized});
        }

        fn loadGraph(session: *const runtime.Session, path: []const u8, g: Geometry, trace_blocks: bool, blockwise: bool) !Self {
            if (!g.valid()) return error.InvalidGeometry;
            if (cfg.ordered_first_qk and session.platform.target != .cuda) return error.ExpectedCudaBackend;
            const tiled = g.image > max_dense_image_tokens;
            if (tiled and !(blockwise and profile == .edit_2511 and cuda_math.enabled()))
                return error.LargeSequenceRequiresCudaReferenceMath;
            var scales: [schema.len]f32 = undefined;
            // Keep workspace available for attention and FP32 projections on long sequences.
            if (blockwise and g.image + g.text >= 1024) {
                const reader = try runtime.checkpoint.openMixedE4M3(session.allocator, session.io, stored.len, &stored, l.Parameter.at, path);
                errdefer reader.deinit();
                try reader.readF32(schema.len, &scales);
                try validateScales(&scales, reader.tensors);
                const stages = try compileBlockwise(session, reader.tensors, null, &scales, g);
                std.log.info("denoiser_graph=blockwise blocks={d} weights=streamed", .{cfg.blocks});
                if (tiled) std.log.info("denoiser_attention=tiled", .{});
                return .{ .session = session, .execution = .{ .blockwise = .{ .stages = stages, .reader = reader } }, .geometry = g, .trace_blocks = false };
            }
            const weights = try runtime.checkpoint.loadMixedE4M3(session.allocator, session.io, session.platform, stored.len, &stored, l.Parameter.at, path);
            errdefer weights.deinit(session.allocator);
            const scale_buffer = weights.buffers[schema.len];
            try scale_buffer.toSlice(session.io, .init(scale_buffer.shape(), std.mem.asBytes(&scales)));
            try validateScales(&scales, weights.tensors);
            const execution: Execution = if (blockwise) blk: {
                const stages = try compileBlockwise(session, weights.tensors, weights.buffers, &scales, g);
                std.log.info("denoiser_graph=blockwise blocks={d}", .{cfg.blocks});
                break :blk .{ .blockwise = .{ .stages = stages } };
            } else blk: {
                const count = if (trace_blocks) traceInputCount(g) else inputCount(g);
                const shape: zml.Shape = .init(.{count}, .f32);
                var graph = try runtime.Compiled.init(session, forward, .{ weights.tensors[0..schema.len], &scales, g, trace_blocks, Tensor.fromShape(shape) });
                graph.args.bake(weights.buffers[0..schema.len]);
                break :blk .{ .whole = graph };
            };
            return .{ .session = session, .weights = weights, .execution = execution, .geometry = g, .trace_blocks = trace_blocks };
        }

        fn runStage(session: *const runtime.Session, stage: *runtime.Compiled, reader: ?runtime.checkpoint.Reader, start: usize, end: usize, inputs: anytype) !zml.Buffer {
            const buffers = if (reader) |file| try file.loadRange(session.platform, start, end) else null;
            defer if (buffers) |loaded| {
                for (loaded) |*buffer| buffer.deinit();
                session.allocator.free(loaded);
            };
            if (buffers) |loaded| try stage.run(.{loaded} ++ inputs) else try stage.run(inputs);
            var result = stage.results.get(zml.Buffer);
            errdefer result.deinit();
            try result.await(session.io);
            return result;
        }

        fn runBlockwise(session: *const runtime.Session, graph: Blockwise, source: zml.Buffer) !zml.Buffer {
            const stages = graph.stages;
            var state = try runStage(session, &stages[0], graph.reader, 0, input_parameters, .{source});
            defer state.deinit();
            for (stages[1 .. stages.len - 1], 0..) |*stage, index| {
                const start = top_parameters + index * block_parameters;
                const next = try runStage(session, stage, graph.reader, start, start + block_parameters, .{ source, state });
                state.deinit();
                state = next;
            }
            const head = &stages[stages.len - 1];
            return runStage(session, head, graph.reader, input_parameters, top_parameters, .{state});
        }

        pub fn predict(self: *Self, input: []const f32, output: []f32) !u32 {
            const g = self.geometry;
            const count = if (self.trace_blocks) traceInputCount(g) else inputCount(g);
            const output_count = if (self.trace_blocks) traceOutputCount(g) else g.target * channels;
            if (input.len != count or output.len != output_count)
                return error.InvalidBufferSize;
            const time = input[inputCount(g) - 1];
            if (!std.math.isFinite(time) or time < 0 or time > 1) return error.InvalidTime;
            const session = self.session;
            var source = try zml.Buffer.fromBytesOpts(session.io, session.platform, .init(.{input.len}, .f32), std.mem.sliceAsBytes(input), .{ .wait = true });
            defer source.deinit();
            var result = switch (self.execution) {
                .whole => |*graph| blk: {
                    try graph.run(.{source});
                    break :blk graph.results.get(zml.Buffer);
                },
                .blockwise => |graph| try runBlockwise(session, graph, source),
            };
            defer result.deinit();
            const shape: zml.Shape = if (self.trace_blocks) .init(.{output_count}, .f32) else .init(.{ 1, g.target, channels }, .f32);
            if (!result.shape().eql(shape)) return error.InvalidOutputShape;
            const shards = try session.verifyPlacement(&result);
            try result.toSlice(session.io, .init(shape, std.mem.sliceAsBytes(output)));
            return shards;
        }

        pub fn deinit(self: *Self) void {
            switch (self.execution) {
                .whole => |*graph| graph.deinit(self.session.allocator),
                .blockwise => |graph| {
                    for (graph.stages) |*stage| stage.deinit(self.session.allocator);
                    self.session.allocator.free(graph.stages);
                    if (graph.reader) |reader| reader.deinit();
                },
            }
            if (self.weights) |weights| weights.deinit(self.session.allocator);
        }
    };
}
