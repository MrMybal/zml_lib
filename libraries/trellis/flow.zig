const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const t = rt.transformer;
const Tensor = l.Tensor;
const weights = @import("flow_weights.zig");

pub const Profile = weights.Profile;
pub const official = weights.official;
pub const fixture = weights.fixture;
pub const pipeline_fixture = weights.pipeline_fixture;
pub const schema = weights.schema;
pub const latent_width = weights.latent_width;
pub const fine_width = weights.fine_width;
pub const coarse_width = weights.coarse_width;

fn norm(x: Tensor, eps: f32) Tensor {
    const axis = x.rank() - 1;
    const centered = x.sub(x.mean(axis).broad(x.shape()));
    return centered.mul(centered.mul(centered).mean(axis).addConstant(eps).rsqrt().broad(x.shape()));
}

pub fn gelu(x: Tensor) Tensor {
    const argument = x.add(x.mul(x).mul(x).scale(@as(f32, 0.044715))).scale(@as(f32, 0.7978845608028654));
    return x.mul(argument.tanh().addConstant(@as(f32, 1))).scale(@as(f32, 0.5));
}

fn frequencies(n: i64) Tensor {
    const scale = -@log(@as(f32, 10000)) / @as(f32, @floatFromInt(n));
    return Tensor.iota(.init(.{n}, .i32), 0).convert(.f32).scale(scale).exp();
}

fn timeEmbedding(w: l.Weights, timestep: Tensor) Tensor {
    const n = weights.time_width / 2;
    const angle = frequencies(n).mul(timestep.reshape(.{}));
    const value = Tensor.concatenate(&.{ angle.cos(), angle.sin() }, 0).reshape(.{ 1, weights.time_width });
    return t.linear(w, "t_embedder.mlp.2", t.linear(w, "t_embedder.mlp.0", value, true).silu(), true);
}

pub fn positionEmbedding(coords: Tensor, width: i64) Tensor {
    const count = coords.dim(0);
    const n = @divFloor(width, 6);
    const axes = coords.reshape(.{ count * 3, 1 });
    const grid: zml.Shape = .init(.{ count * 3, n }, .f32);
    const angle = axes.broad(grid).mul(frequencies(n).broadcast(grid, &.{1}));
    const values = Tensor.concatenate(&.{ angle.sin(), angle.cos() }, 1).reshape(.{ count, n * 6 });
    if (width == n * 6) return values;
    const zeros = Tensor.scalar(@as(f32, 0), .f32).broad(.init(.{ count, width - n * 6 }, .f32));
    return Tensor.concatenate(&.{ values, zeros }, 1);
}

fn headedNorm(x: Tensor, gamma: Tensor) Tensor {
    const length = x.mul(x).sum(3).sqrt().maximum(.scalar(@as(f32, 1e-12), .f32)).broad(x.shape());
    const gain: f32 = @sqrt(@as(f32, @floatFromInt(x.dim(3))));
    return x.div(length).mul(gamma.broadcast(x.shape(), &.{ 1, 3 })).scale(gain);
}

fn adapt(x: Tensor, modulation: Tensor, shift: i64, scale: i64) Tensor {
    const bias = modulation.slice(1, .single(shift)).broadcast(x.shape(), &.{ 0, 2 });
    const strength = modulation.slice(1, .single(scale)).broadcast(x.shape(), &.{ 0, 2 }).addConstant(@as(f32, 1));
    return x.mul(strength).add(bias);
}

fn block(comptime p: Profile, w: l.Weights, i: usize, x: Tensor, context: Tensor, time: Tensor) Tensor {
    var prefix: [32]u8 = undefined;
    var buf: [96]u8 = undefined;
    const key = l.name(&prefix, "blocks.{d}", .{i});
    const mod = t.linear(w, l.name(&buf, "{s}.adaLN_modulation.1", .{key}), time.silu(), true).reshape(.{ 1, 6, p.width });
    const a = adapt(norm(x, 1e-6), mod, 0, 1);
    const qkv = t.linear(w, l.name(&buf, "{s}.self_attn.to_qkv", .{key}), a, true).reshape(.{ 1, x.dim(1), 3, p.width });
    const q = headedNorm(t.splitHeads(qkv.slice(2, .single(0)), p.heads), w.get(l.name(&buf, "{s}.self_attn.q_rms_norm", .{key}), ".gamma"));
    const k = headedNorm(t.splitHeads(qkv.slice(2, .single(1)), p.heads), w.get(l.name(&buf, "{s}.self_attn.k_rms_norm", .{key}), ".gamma"));
    const v = t.splitHeads(qkv.slice(2, .single(2)), p.heads);
    const self = t.linear(w, l.name(&buf, "{s}.self_attn.to_out", .{key}), t.attend(t.scores(q, k), v), true);
    const y = x.add(self.mul(mod.slice(1, .single(2)).broadcast(x.shape(), &.{ 0, 2 })));
    const n2 = t.norm(w, l.name(&buf, "{s}.norm2", .{key}), y, 1e-6);
    const cross_q = t.linear(w, l.name(&buf, "{s}.cross_attn.to_q", .{key}), n2, true);
    const kv = t.linear(w, l.name(&buf, "{s}.cross_attn.to_kv", .{key}), context, true).reshape(.{ 1, context.dim(1), 2, p.width });
    const attention = t.attention(cross_q, kv.slice(2, .single(0)), kv.slice(2, .single(1)), p.heads);
    const after = y.add(t.linear(w, l.name(&buf, "{s}.cross_attn.to_out", .{key}), attention, true));
    const feed = adapt(norm(after, 1e-6), mod, 3, 4);
    const hidden = gelu(t.linear(w, l.name(&buf, "{s}.mlp.mlp.0", .{key}), feed, true));
    const update = t.linear(w, l.name(&buf, "{s}.mlp.mlp.2", .{key}), hidden, true);
    return after.add(update.mul(mod.slice(1, .single(5)).broadcast(after.shape(), &.{ 0, 2 })));
}

fn transform(comptime p: Profile, w: l.Weights, x: Tensor, context: Tensor, time: Tensor) Tensor {
    var value = x.reshape(.{ 1, x.dim(0), p.width });
    const cond = context.reshape(.{ 1, context.dim(0), p.context });
    for (0..p.blocks) |i| value = block(p, w, i, value, cond, time);
    return value.reshape(x.shape());
}

pub fn gather(x: Tensor, indices: Tensor, comptime rank: u8) Tensor {
    const table = x.withTags(.{ .site, .channel });
    const values = indices.convert(.i32);
    const tagged = if (rank == 1) values.withTags(.{.row}) else values.withTags(.{ .row, .neighbor });
    return table.gather(.{ .site = tagged }, .{});
}

pub fn zeroRow(x: Tensor) Tensor {
    const zero = Tensor.scalar(@as(f32, 0), .f32).broad(.init(.{ 1, x.dim(1) }, .f32));
    return Tensor.concatenate(&.{ x, zero }, 0);
}

pub fn convolution(w: l.Weights, comptime key: []const u8, x: Tensor, neighbors: Tensor) Tensor {
    const features = gather(zeroRow(x), neighbors, 2).reshape(.{ x.dim(0), 27 * x.dim(1) });
    const kernel = w.get(key, ".weight");
    return t.linearParameters(features, kernel.reshape(.{ kernel.dim(0), 27 * x.dim(1) }), w.get(key, ".bias"));
}

fn residual(
    w: l.Weights,
    comptime key: []const u8,
    x: Tensor,
    time: Tensor,
    neighbors: Tensor,
    comptime out_width: i64,
    comptime has_skip: bool,
) Tensor {
    const modulation = t.linear(w, key ++ ".emb_layers.1", time.silu(), true);
    const inner = convolution(w, key ++ ".conv1.conv", t.norm(w, key ++ ".norm1", x, 1e-6).silu(), neighbors);
    const scale = modulation.slice(1, .{ .end = out_width }).broad(inner.shape()).addConstant(@as(f32, 1));
    const shift = modulation.slice(1, .{ .start = out_width }).broad(inner.shape());
    const update = convolution(w, key ++ ".conv2.conv", norm(inner, 1e-6).mul(scale).add(shift).silu(), neighbors);
    return update.add(if (has_skip) t.linear(w, key ++ ".skip_connection", x, true) else x);
}

pub fn Network(comptime p: Profile, comptime sparse: bool) type {
    return struct {
        pub const parameters = schema(p, sparse);

        pub fn forward(params: []const Tensor, sample: Tensor, context: Tensor, timestep: Tensor, fine: Tensor, coarse: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &parameters, .tensors = params };
            const time = timeEmbedding(w, timestep);
            const x = t.linear(w, "input_layer", sample, true);
            if (!sparse) {
                const tokens = transform(p, w, x.add(w.get("pos_emb", "")), context, time);
                return t.linear(w, "out_layer", norm(tokens, 1e-5), true);
            }
            const fine_neighbors = fine.slice(1, .{ .start = 1 });
            const coarse_neighbors = coarse.slice(1, .{ .start = 3, .end = 30 });
            const children = coarse.slice(1, .{ .start = 30, .end = 38 });
            const parent = fine.slice(1, .single(0));
            const local = residual(w, "input_blocks.0", x, time, fine_neighbors, p.io, false);
            const summed = gather(zeroRow(local), children, 2).sum(1).reshape(.{ coarse.dim(0), p.io });
            const pooled = summed.div(coarse.slice(1, .{ .start = 38 }).broad(summed.shape()));
            const down = residual(w, "input_blocks.1", pooled, time, coarse_neighbors, p.width, true);
            const positioned = down.add(positionEmbedding(coarse.slice(1, .{ .end = 3 }), p.width));
            const tokens = transform(p, w, positioned, context, time);
            const coarse_skip = Tensor.concatenate(&.{ tokens, down }, 1);
            const expanded = gather(coarse_skip, parent, 1).reshape(.{ sample.dim(0), 2 * p.width });
            const fine_out = residual(w, "out_blocks.0", expanded, time, fine_neighbors, p.io, true);
            const fine_skip = Tensor.concatenate(&.{ fine_out, local }, 1);
            const decoded = residual(w, "out_blocks.1", fine_skip, time, fine_neighbors, p.io, true);
            return t.linear(w, "out_layer", norm(decoded, 1e-5), true);
        }
    };
}
