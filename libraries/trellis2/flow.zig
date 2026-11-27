const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const t = rt.transformer;
const wdef = @import("weights.zig");
const Tensor = zml.Tensor;

pub const Profile = wdef.Profile;
const width = wdef.width;
const heads = wdef.heads;
const depth = wdef.head_width;
pub const context_width = wdef.context_width;

fn linear(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    const weight = w.get(name, ".weight");
    const kernel = if (weight.dtype() == .u8) t.decodeE4M3(weight, 1) else weight;
    return t.linearParameters(x, kernel, w.get(name, ".bias"));
}

fn time(w: l.Weights, timestep: Tensor) Tensor {
    const ratio = -@log(@as(f32, 10000)) / 128;
    const i = Tensor.iota(.init(.{128}, .i32), 0).convert(.f32);
    const angle = i.scale(ratio).exp().mul(timestep.reshape(.{}));
    const encoded = Tensor.concatenate(&.{ angle.cos(), angle.sin() }, 0).reshape(.{ 1, 256 });
    const embedding = linear(w, "t_embedder.mlp.2", linear(w, "t_embedder.mlp.0", encoded).silu());
    return linear(w, "adaLN_modulation.1", embedding.silu());
}

fn phases(coords: Tensor) Tensor {
    const rows = coords.dim(0);
    const per_axis = depth / 2 / 3;
    const ratio = -@log(@as(f32, 10000)) / @as(f32, per_axis);
    const pos: zml.Shape = .init(.{ rows, 3, per_axis }, .f32);
    const frequencies = Tensor.iota(pos.withDtype(.i32), 2).convert(.f32).scale(ratio).exp();
    const angle = coords.broadcast(pos, &.{ 0, 1 }).mul(frequencies).reshape(.{ rows, 3 * per_axis });
    const zeros = Tensor.scalar(@as(f32, 0), .f32).broad(.init(.{ rows, depth / 2 - 3 * per_axis }, .f32));
    return Tensor.concatenate(&.{ angle, zeros }, 1);
}

fn rotate(x: Tensor, sin: Tensor, cos: Tensor) Tensor {
    const split = x.reshape(.{ 1, heads, x.dim(2), depth / 2, 2 });
    const re = split.slice(4, .single(0)).reshape(.{ 1, heads, x.dim(2), depth / 2 });
    const im = split.slice(4, .single(1)).reshape(re.shape());
    const a = cos.broadcast(re.shape(), &.{ 2, 3 });
    const b = sin.broadcast(re.shape(), &.{ 2, 3 });
    return Tensor.concatenate(&.{ re.mul(a).sub(im.mul(b)).reshape(.{ 1, heads, x.dim(2), depth / 2, 1 }), re.mul(b).add(im.mul(a)).reshape(.{ 1, heads, x.dim(2), depth / 2, 1 }) }, 4)
        .reshape(x.shape());
}

fn normalizeHeads(x: Tensor, gain: Tensor) Tensor {
    const l2 = x.mul(x).sum(3).sqrt().maximum(.scalar(@as(f32, 1e-12), .f32));
    return x.div(l2.broad(x.shape())).mul(gain.broadcast(x.shape(), &.{ 1, 3 })).scale(@sqrt(@as(f32, depth)));
}

fn normalizedProjection(w: l.Weights, scope: []const u8, comptime part: []const u8, x: Tensor) Tensor {
    return normalizeHeads(t.splitHeads(x, heads), w.get(scope, "." ++ part ++ "_rms_norm.gamma"));
}

fn affine(x: Tensor, mod: Tensor, slot: i64) Tensor {
    return mod.slice(1, .single(slot)).broadcast(x.shape(), &.{ 0, 2 });
}

fn adaptive(x: Tensor, mod: Tensor, slot: i64) Tensor {
    const normalized = t.normAffineFree(x, 1e-6);
    return normalized.mul(affine(x, mod, slot + 1).addConstant(@as(f32, 1))).add(affine(x, mod, slot));
}

const QueryTiles = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,

    const tile = 256;
    pub const State = struct { start: Tensor, out: Tensor };

    pub fn cond(self: QueryTiles, state: State) Tensor {
        return state.start.cmp(.LT, .scalar(self.q.dim(2), .i32));
    }

    pub fn body(self: QueryTiles, state: State) State {
        const zero: Tensor = .scalar(0, .i32);
        const query = self.q.slice(2, .dyn(state.start, tile));
        const answer = t.attend(t.scores(query, self.k), self.v);
        return .{
            .start = state.start.addConstant(@as(i32, tile)),
            .out = state.out.dynamicUpdateSlice(.{ zero, state.start, zero }, answer),
        };
    }
};

fn attention(q: Tensor, k: Tensor, v: Tensor) Tensor {
    const rows = q.dim(2);
    if (rows <= 4096) return t.attend(t.scores(q, k), v);
    // A tile retains the full key sequence and hence the same softmax denominator.
    const padded_rows = @divTrunc(rows + QueryTiles.tile - 1, QueryTiles.tile) * QueryTiles.tile;
    const padded_q = if (padded_rows == rows) q else Tensor.concatenate(&.{
        q, Tensor.zeroes(.init(.{ q.dim(0), heads, padded_rows - rows, depth }, q.dtype())),
    }, 2);
    const state = zml.ops.@"while"(QueryTiles, .{ .q = padded_q, .k = k, .v = v }, .{
        .start = .scalar(0, .i32),
        .out = .zeroes(.init(.{ q.dim(0), padded_rows, width }, q.dtype())),
    });
    return state.out.slice(1, .{ .end = rows });
}

fn block(comptime p: Profile, w: l.Weights, index: usize, x: Tensor, step: Tensor, ctx: Tensor, projected: Tensor, sin: Tensor, cos: Tensor) Tensor {
    var base: [24]u8 = undefined;
    var self_base: [48]u8 = undefined;
    var cross_base: [64]u8 = undefined;
    var key: [96]u8 = undefined;
    const scope = l.name(&base, "blocks.{d}", .{index});
    const self_scope = l.name(&self_base, "{s}.self_attn", .{scope});
    const cross = l.name(&cross_base, "{s}.cross_attn{s}", .{ scope, if (p.projectionWidth() != 0) ".cross_attn_block" else "" });
    const mod = step.add(w.get(scope, ".modulation").broadcast(step.shape(), &.{1})).reshape(.{ 1, 6, width });
    const norm = adaptive(x, mod, 0);
    const qkv = linear(w, l.name(&key, "{s}.to_qkv", .{self_scope}), norm).reshape(.{ 1, x.dim(1), 3, width });
    const q = rotate(normalizedProjection(w, self_scope, "q", qkv.slice(2, .single(0))), sin, cos);
    const k = rotate(normalizedProjection(w, self_scope, "k", qkv.slice(2, .single(1))), sin, cos);
    const v = t.splitHeads(qkv.slice(2, .single(2)), heads);
    const self_update = linear(w, l.name(&key, "{s}.to_out", .{self_scope}), attention(q, k, v));
    const first = x.add(self_update.mul(affine(x, mod, 2)));

    const normalized = t.norm(w, l.name(&key, "{s}.norm2", .{scope}), first, 1e-6);
    const cross_q = normalizedProjection(w, cross, "q", linear(w, l.name(&key, "{s}.to_q", .{cross}), normalized));
    const cross_kv = linear(w, l.name(&key, "{s}.to_kv", .{cross}), ctx).reshape(.{ 1, ctx.dim(1), 2, width });
    const cross_k = normalizedProjection(w, cross, "k", cross_kv.slice(2, .single(0)));
    const cross_v = t.splitHeads(cross_kv.slice(2, .single(1)), heads);
    const cross_update = linear(w, l.name(&key, "{s}.to_out", .{cross}), attention(cross_q, cross_k, cross_v));
    const fused = if (p.projectionWidth() != 0)
        cross_update.add(linear(w, l.name(&key, "{s}.cross_attn.proj_linear", .{scope}), projected))
    else
        cross_update;
    const second = first.add(fused);
    const hidden = t.geluTanh(linear(w, l.name(&key, "{s}.mlp.mlp.0", .{scope}), adaptive(second, mod, 3)));
    const update = linear(w, l.name(&key, "{s}.mlp.mlp.2", .{scope}), hidden);
    return second.add(update.mul(affine(second, mod, 5)));
}

pub fn Denoiser(comptime p: Profile) type {
    return struct {
        pub const schema = wdef.schema(p);

        pub fn forward(parameters: []const Tensor, latent: Tensor, coords: Tensor, timestep: Tensor, context: Tensor, projection: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
            const rows = latent.dim(0);
            var x = linear(w, "input_layer", latent.reshape(.{ 1, rows, p.inputWidth() }));
            const cond = context.reshape(.{ 1, context.dim(0), context_width });
            const view = projection.reshape(.{ 1, rows, @max(1, p.projectionWidth()) });
            const pos = phases(coords);
            const sin = pos.sin();
            const cos = pos.cos();
            const step = time(w, timestep);
            for (0..wdef.blocks) |i| x = block(p, w, i, x, step, cond, view, sin, cos);
            return linear(w, "out_layer", t.normAffineFree(x, 1e-5)).reshape(.{ rows, p.outputWidth() });
        }
    };
}
