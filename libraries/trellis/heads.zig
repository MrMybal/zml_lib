const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const flow = @import("zml_lib/trellis/flow");
const weights = @import("head_weights.zig");
const l = rt.layers;
const t = rt.transformer;
const Tensor = l.Tensor;

pub const Kind = weights.Kind;
pub const Profile = weights.Profile;
pub const official = weights.official;
pub const fixture = weights.fixture;

pub fn outputWidth(kind: Kind) i64 {
    return weights.channels(kind) + (if (kind == .radiance) @as(i64, 3) else 0);
}

fn windowAttention(w: l.Weights, prefix: []const u8, p: Profile, x: Tensor, indices: Tensor) Tensor {
    var buf: [64]u8 = undefined;
    const rows = x.dim(0);
    const count = indices.dim(0);
    const size = indices.dim(1);
    const qkv_sorted = flow.gather(flow.zeroRow(t.linear(w, l.name(&buf, "{s}.attn.to_qkv", .{prefix}), x, true)), indices, 2)
        .reshape(.{ count, size, 3, p.width });
    const q = t.splitHeads(qkv_sorted.slice(2, .single(0)), p.heads);
    const k = t.splitHeads(qkv_sorted.slice(2, .single(1)), p.heads);
    const v = t.splitHeads(qkv_sorted.slice(2, .single(2)), p.heads);
    const scores = t.scores(q, k);
    const valid = indices.cmp(.LT, .scalar(@as(f32, @floatFromInt(rows)), .f32))
        .broadcast(scores.shape().withDtype(.bool), &.{ 0, 3 });
    const masked = Tensor.select(valid, scores, Tensor.scalar(@as(f32, -1e30), .f32).broad(scores.shape()));
    return t.attend(masked, v).reshape(.{ count * size, p.width });
}

fn transformerBlock(w: l.Weights, p: Profile, i: usize, x: Tensor, indices: Tensor, inverse: Tensor) Tensor {
    var key: [32]u8 = undefined;
    var buf: [64]u8 = undefined;
    const prefix = l.name(&key, "blocks.{d}", .{i});
    const ordered = windowAttention(w, prefix, p, t.normAffineFree(x, 1e-6), indices);
    const attended = flow.gather(ordered, inverse, 1).reshape(x.shape());
    const y = x.add(t.linear(w, l.name(&buf, "{s}.attn.to_out", .{prefix}), attended, true));
    const hidden = flow.gelu(t.linear(w, l.name(&buf, "{s}.mlp.mlp.0", .{prefix}), t.normAffineFree(y, 1e-6), true));
    return y.add(t.linear(w, l.name(&buf, "{s}.mlp.mlp.2", .{prefix}), hidden, true));
}

fn groupNorm(w: l.Weights, comptime key: []const u8, x: Tensor, groups: i64) Tensor {
    const rows = x.dim(0);
    const width = x.dim(1);
    const grouped = x.transpose(.{ 1, 0 }).reshape(.{ groups, @divExact(width, groups) * rows });
    const normalized = t.normAffineFree(grouped, 1e-5).reshape(.{ width, rows }).transpose(.{ 1, 0 });
    return l.biasAdd(normalized.mul(w.get(key, ".weight").broadcast(x.shape(), &.{1})), w.get(key, ".bias"), 1);
}

fn subdivide(x: Tensor) Tensor {
    const rows = x.dim(0);
    const width = x.dim(1);
    return x.broadcast(.init(.{ rows, 8, width }, .f32), &.{ 0, 2 }).reshape(.{ 8 * rows, width });
}

fn upsample(w: l.Weights, comptime key: []const u8, x: Tensor, neighbors: Tensor, groups: i64) Tensor {
    const skip = w.get(key ++ ".skip_connection.conv", ".weight");
    const skip_values = t.linearParameters(subdivide(x), skip.reshape(.{ skip.dim(0), x.dim(1) }), w.get(key ++ ".skip_connection.conv", ".bias"));
    const before = subdivide(groupNorm(w, key ++ ".act_layers.0", x, groups).silu());
    const first = flow.convolution(w, key ++ ".out_layers.0.conv", before, neighbors);
    const result = flow.convolution(w, key ++ ".out_layers.3.conv", groupNorm(w, key ++ ".out_layers.1", first, groups).silu(), neighbors);
    return result.add(skip_values);
}

fn represent(comptime kind: Kind, w: l.Weights, values: Tensor, coords: Tensor, resolution: i64) Tensor {
    const rows = coords.dim(0);
    const scale: f32 = 1 / @as(f32, @floatFromInt(resolution));
    const centers = coords.addConstant(@as(f32, 0.5)).scale(scale);
    if (kind == .gaussian) {
        const limit = weights.gaussians * 3;
        const offset = values.slice(1, .{ .end = limit }).reshape(.{ rows, weights.gaussians, 3 });
        const perturbation = w.get("offset_perturbation", "").broadcast(offset.shape(), &.{ 1, 2 });
        const position = centers.broadcast(offset.shape(), &.{ 0, 2 }).add(offset.add(perturbation).tanh().scale(scale * 0.75));
        const rotation_start = limit * 3;
        const rotation_end = rotation_start + weights.gaussians * 4;
        return Tensor.concatenate(&.{
            position.reshape(.{ rows, limit }),                                                      values.slice(1, .{ .start = limit, .end = rotation_start }),
            values.slice(1, .{ .start = rotation_start, .end = rotation_end }).scale(@as(f32, 0.1)), values.slice(1, .{ .start = rotation_end }),
        }, 1);
    }
    const trivec = weights.rank * 3 * weights.coefficients;
    return Tensor.concatenate(&.{ centers, values.slice(1, .{ .end = trivec }).addConstant(@as(f32, 1)), values.slice(1, .{ .start = trivec }) }, 1);
}

pub fn Decoder(comptime p: Profile, comptime kind: Kind) type {
    return struct {
        pub const parameters = weights.schema(p, kind);

        pub fn forward(params: []const Tensor, features: Tensor, coords: Tensor, windows: [2]Tensor, inverse: [2]Tensor, neighbors: [2]Tensor) Tensor {
            const w: l.Weights = .{ .schema = &parameters, .tensors = params };
            var x = t.linear(w, "input_layer", features, true).add(flow.positionEmbedding(coords, p.width));
            for (0..p.blocks) |i| x = transformerBlock(w, p, i, x, windows[i % 2], inverse[i % 2]);
            if (kind == .mesh) {
                const first = upsample(w, "upsample.0", x, neighbors[0], p.groups);
                return t.linear(w, "out_layer", upsample(w, "upsample.1", first, neighbors[1], p.groups), true);
            }
            const values = t.linear(w, "out_layer", t.normAffineFree(x, 1e-5), true);
            return represent(kind, w, values, coords, p.resolution);
        }
    };
}
