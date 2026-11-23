const std = @import("std");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const t = rt.transformer;
const Tensor = l.Tensor;
const p = @import("weights.zig");

pub const schema = p.schema;
pub const frames_max = 4096;
const heads = 8;
const head_width = p.width / heads;
const max_window = 120;
const observation_count = p.keypoints * 3;
const crop_offset = observation_count;
const angular_offset = crop_offset + p.camera_width;
const features_offset = angular_offset + p.motion_features;
pub const input_width = features_offset + p.image_features;
pub const output_width = p.motion_width + p.camera_width + p.contact_width;

fn geluTanh(x: Tensor) Tensor {
    const cubic = x.mul(x).mul(x).scale(@as(f32, 0.044715));
    const y = x.add(cubic).scale(@as(f32, 0.7978845608028654)).tanh();
    return x.mul(y.addConstant(@as(f32, 1))).scale(@as(f32, 0.5));
}

fn mlp(w: l.Weights, prefix: []const u8, x: Tensor, comptime act: anytype) Tensor {
    var key: [70]u8 = undefined;
    const hidden = act(t.linear(w, l.name(&key, "{s}.fc1", .{prefix}), x, true));
    return t.linear(w, l.name(&key, "{s}.fc2", .{prefix}), hidden, true);
}

fn condition(w: l.Weights, comptime prefix: []const u8, values: Tensor) Tensor {
    return t.linear(w, prefix ++ ".3", t.linear(w, prefix ++ ".0", values, true).silu(), true);
}

const Angles = struct {
    cosine: Tensor,
    sine: Tensor,

    fn init(frames: i64) Angles {
        const dim = head_width / 2;
        const scale = -@log(@as(f32, 10000)) / @as(f32, @floatFromInt(dim));
        const freqs = Tensor.iota(.init(.{dim}, .i32), 0).convert(.f32)
            .scale(scale).exp();
        const times = Tensor.iota(.init(.{frames}, .i32), 0).convert(.f32);
        const shape = times.shape().append(.{dim});
        const radians = times.broadcast(shape, &.{0}).mul(freqs.broadcast(shape, &.{1}));
        return .{ .cosine = radians.cos(), .sine = radians.sin() };
    }

    fn apply(self: Angles, x: Tensor) Tensor {
        const pairs = x.reshape(.{ 1, heads, x.dim(2), head_width / 2, 2 });
        const real = pairs.slice(4, .single(0));
        const imag = pairs.slice(4, .single(1));
        const cosine = self.cosine.broadcast(real.shape(), &.{ 2, 3 });
        const sine = self.sine.broadcast(real.shape(), &.{ 2, 3 });
        const re = real.mul(cosine).sub(imag.mul(sine));
        const im = real.mul(sine).add(imag.mul(cosine));
        return Tensor.stack(&.{ re, im }, .last, .pair).reshape(x.shape());
    }
};

fn attentionMask(frames: i64) ?Tensor {
    if (frames <= max_window) return null;
    const row = Tensor.iota(.init(.{ frames, frames }, .i32), 0);
    const col = Tensor.iota(row.shape(), 1);
    const edge: i32 = @intCast(frames);
    // Near either end, keep the reference window anchored inside the sequence.
    const lo = row.subConstant(@as(i32, max_window / 2)).maximum(.scalar(@as(i32, 0), .i32))
        .minimum(.scalar(edge - max_window, .i32));
    const hi = row.addConstant(@as(i32, max_window / 2)).minimum(.scalar(edge, .i32))
        .maximum(.scalar(@as(i32, max_window), .i32));
    const invalid = col.cmp(.LT, lo).logical(.OR, col.cmp(.GE, hi));
    const values = row.shape().withDtype(.f32);
    return invalid.select(Tensor.scalar(-std.math.inf(f32), .f32).broad(values), Tensor.scalar(@as(f32, 0), .f32).broad(values));
}

fn block(w: l.Weights, i: usize, x: Tensor, angles: Angles, mask: ?Tensor) Tensor {
    var base: [32]u8 = undefined;
    var key: [80]u8 = undefined;
    const prefix = l.name(&base, "blocks.{d}", .{i});
    const n1 = t.norm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 1e-6);
    const q = t.splitHeads(t.linear(w, l.name(&key, "{s}.attn.query", .{prefix}), n1, true), heads);
    const k = t.splitHeads(t.linear(w, l.name(&key, "{s}.attn.key", .{prefix}), n1, true), heads);
    const v = t.splitHeads(t.linear(w, l.name(&key, "{s}.attn.value", .{prefix}), n1, true), heads);
    var logits = t.scores(angles.apply(q), angles.apply(k));
    if (mask) |m| logits = logits.add(m.broadcast(logits.shape(), &.{ 2, 3 }));
    const projected = t.linear(w, l.name(&key, "{s}.attn.proj", .{prefix}), t.attend(logits, v), true);
    const y = x.add(projected.mul(w.get(prefix, ".gate_msa").broad(x.shape())));
    const n2 = t.norm(w, l.name(&key, "{s}.norm2", .{prefix}), y, 1e-6);
    const update = mlp(w, l.name(&key, "{s}.mlp", .{prefix}), n2, geluTanh);
    return y.add(update.mul(w.get(prefix, ".gate_mlp").broad(y.shape())));
}

fn forward(parameters: []const Tensor, features: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &p.schema, .tensors = parameters };
    const frames = features.dim(0);
    const input = features.reshape(.{ 1, frames, input_width });
    const obs = input.slice(2, .{ .end = crop_offset }).reshape(.{ 1, frames, p.keypoints, 3 });
    const conf = obs.slice(3, .{ .start = 2 });
    const visible = conf.cmp(.GT, .scalar(@as(f32, 0.5), .f32));
    const embeddings = t.linear(w, "learned_pos_linear", obs.slice(3, .{ .end = 2 }), true);
    const learned = w.get("learned_pos_params", "").broadcast(embeddings.shape(), &.{ 2, 3 });
    const per_joint = visible.broad(embeddings.shape().withDtype(.bool)).select(embeddings, learned);
    var encoded = mlp(w, "embed_noisyobs", per_joint.reshape(.{ 1, frames, p.keypoints * p.position_width }), t.gelu);
    encoded = encoded.add(condition(w, "cliffcam_embedder", input.slice(2, .{ .start = crop_offset, .end = angular_offset })));
    encoded = encoded.add(condition(w, "cam_angvel_embedder", input.slice(2, .{ .start = angular_offset, .end = features_offset })));
    const img_norm = t.norm(w, "imgseq_embedder.0", input.slice(2, .{ .start = features_offset }), 1e-5);
    encoded = encoded.add(t.linear(w, "imgseq_embedder.1", img_norm, true));
    const angles = Angles.init(frames);
    const mask = attentionMask(frames);
    for (0..p.blocks) |i| encoded = block(w, i, encoded, angles, mask);

    const raw_pose = mlp(w, "final_layer", encoded, t.gelu);
    const per_frame_betas = raw_pose.slice(2, .{ .start = 126, .end = 136 });
    const betas = per_frame_betas.mean(1).broad(per_frame_betas.shape());
    const motion = Tensor.concatenate(&.{ raw_pose.slice(2, .{ .end = 126 }), betas, raw_pose.slice(2, .{ .start = 136 }) }, 2);
    const raw_camera = mlp(w, "pred_cam_head", encoded, t.gelu);
    const s = raw_camera.slice(2, .{ .end = 1 }).scale(@as(f32, 0.1784)).addConstant(@as(f32, 1.0606))
        .maximum(.scalar(@as(f32, 0.25), .f32));
    const x = raw_camera.slice(2, .{ .start = 1, .end = 2 }).scale(@as(f32, 0.0956)).addConstant(@as(f32, -0.0027));
    const y = raw_camera.slice(2, .{ .start = 2 }).scale(@as(f32, 0.0764)).addConstant(@as(f32, 0.2702));
    const contacts = mlp(w, "static_conf_head", encoded, t.gelu);
    return Tensor.concatenate(&.{ motion, s, x, y, contacts }, 2).reshape(.{ frames, output_width });
}

pub const Model = struct {
    graph: rt.TensorModel,

    pub fn load(session: *const rt.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn predict(self: *Model, frames: i64, features: []const f32, output: []f32) !u32 {
        if (frames < 1 or frames > frames_max) return error.InvalidFrameCount;
        return self.graph.infer(forward, .init(.{ frames, input_width }, .f32), .init(.{ frames, output_width }, .f32), features, output);
    }
};
