const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = zml.Tensor;
const Weights = l.Weights;

pub const mels = 128;
pub const width = 1024;
pub const max_frames = 3000;
pub const live_frames = 2000;
const channels = 256;
const expansion = 4096;
const heads = 8;
const head_width = width / heads;
const blocks = 24;
const per_block = 28;
const subsampling = "encoder.subsampling";

fn parameter(name: []const u8, dims: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

pub const schema = blk: {
    @setEvalBranchQuota(2_000_000);
    var result: [12 + blocks * per_block]l.Parameter = undefined;
    result[0..12].* = .{
        parameter(subsampling ++ ".layers.0.weight", .{ channels, 1, 3, 3 }),
        parameter(subsampling ++ ".layers.0.bias", .{channels}),
        parameter(subsampling ++ ".layers.2.weight", .{ channels, 1, 3, 3 }),
        parameter(subsampling ++ ".layers.2.bias", .{channels}),
        parameter(subsampling ++ ".layers.3.weight", .{ channels, channels, 1, 1 }),
        parameter(subsampling ++ ".layers.3.bias", .{channels}),
        parameter(subsampling ++ ".layers.5.weight", .{ channels, 1, 3, 3 }),
        parameter(subsampling ++ ".layers.5.bias", .{channels}),
        parameter(subsampling ++ ".layers.6.weight", .{ channels, channels, 1, 1 }),
        parameter(subsampling ++ ".layers.6.bias", .{channels}),
        parameter(subsampling ++ ".linear.weight", .{ width, channels * (mels / 8) }),
        parameter(subsampling ++ ".linear.bias", .{width}),
    };
    for (0..blocks) |i| {
        const name = std.fmt.comptimePrint("encoder.layers.{d}", .{i});
        const start = 12 + i * per_block;
        result[start..][0..per_block].* = .{
            parameter(name ++ ".feed_forward1.linear1.weight", .{ expansion, width }),
            parameter(name ++ ".feed_forward1.linear2.weight", .{ width, expansion }),
            parameter(name ++ ".feed_forward2.linear1.weight", .{ expansion, width }),
            parameter(name ++ ".feed_forward2.linear2.weight", .{ width, expansion }),
            parameter(name ++ ".self_attn.q_proj.weight", .{ width, width }),
            parameter(name ++ ".self_attn.k_proj.weight", .{ width, width }),
            parameter(name ++ ".self_attn.v_proj.weight", .{ width, width }),
            parameter(name ++ ".self_attn.o_proj.weight", .{ width, width }),
            parameter(name ++ ".self_attn.relative_k_proj.weight", .{ width, width }),
            parameter(name ++ ".self_attn.bias_u", .{ heads, head_width }),
            parameter(name ++ ".self_attn.bias_v", .{ heads, head_width }),
            parameter(name ++ ".conv.pointwise_conv1.weight", .{ 2 * width, width }),
            parameter(name ++ ".conv.pointwise_conv2.weight", .{ width, width }),
            parameter(name ++ ".conv.depthwise_conv.weight", .{ width, 1, 9 }),
            parameter(name ++ ".conv.norm.weight", .{width}),
            parameter(name ++ ".conv.norm.bias", .{width}),
            parameter(name ++ ".conv.norm.running_mean", .{width}),
            parameter(name ++ ".conv.norm.running_var", .{width}),
            parameter(name ++ ".norm_feed_forward1.weight", .{width}),
            parameter(name ++ ".norm_feed_forward1.bias", .{width}),
            parameter(name ++ ".norm_feed_forward2.weight", .{width}),
            parameter(name ++ ".norm_feed_forward2.bias", .{width}),
            parameter(name ++ ".norm_conv.weight", .{width}),
            parameter(name ++ ".norm_conv.bias", .{width}),
            parameter(name ++ ".norm_self_att.weight", .{width}),
            parameter(name ++ ".norm_self_att.bias", .{width}),
            parameter(name ++ ".norm_out.weight", .{width}),
            parameter(name ++ ".norm_out.bias", .{width}),
        };
    }
    break :blk result;
};

fn relu(x: Tensor) Tensor {
    return x.maximum(Tensor.scalar(0, .f32).broad(x.shape()));
}

fn validFrames(steps: i64, valid: Tensor, stride: f32) Tensor {
    const index = Tensor.iota(.init(.{steps}, .i32), 0).convert(.f32).scale(stride);
    return index.cmp(.LT, valid.broad(index.shape()));
}

fn maskSignal(x: Tensor, mask: Tensor, axis: i64) Tensor {
    return mask.broadcast(x.shape().withDtype(.bool), &.{axis})
        .select(x, Tensor.scalar(0, .f32).broad(x.shape()));
}

fn subsample(w: Weights, input: Tensor, valid: ?Tensor) Tensor {
    var x = input.reshape(.{ 1, 1, input.dim(0), mels });
    x = relu(l.conv3x3Stride2(w, subsampling ++ ".layers.0", x, true));
    const depths = [_][]const u8{ subsampling ++ ".layers.2", subsampling ++ ".layers.5" };
    const points = [_][]const u8{ subsampling ++ ".layers.3", subsampling ++ ".layers.6" };
    const strides = [_]f32{ 2, 4 };
    for (depths, points, strides) |depthwise, pointwise, stride| {
        // Convolution biases can otherwise make padded frames nonzero.
        if (valid) |length| x = maskSignal(x, validFrames(x.dim(2), length, stride), 2);
        const padding = [4]i64{ 1, @mod(x.dim(2), 2), 1, @mod(x.dim(3), 2) };
        x = l.convGroups(w, depthwise, x, 2, padding, true, channels);
        x = relu(l.conv(w, pointwise, x, 1, .{ 0, 0, 0, 0 }, true));
    }
    const steps = x.dim(2);
    return t.linear(w, subsampling ++ ".linear", x.transpose(.{ 0, 2, 1, 3 }).reshape(.{ 1, steps, channels * (mels / 8) }), true);
}

fn positions(steps: i64) Tensor {
    const size: zml.Shape = .init(.{ 2 * steps - 1, width / 2 }, .i32);
    const power = Tensor.iota(size, 1).convert(.f32).scale(@as(f32, -2) / @as(f32, width));
    const base = Tensor.scalar(10_000, .f32).broad(power.shape());
    const shift: f32 = @floatFromInt(steps - 1);
    const distances = Tensor.iota(size, 0).convert(.f32).scale(@as(f32, -1)).addConstant(shift);
    const angle = distances.mul(base.pow(power));
    const pair_shape = .{ 2 * steps - 1, width / 2, 1 };
    return Tensor.concatenate(&.{ angle.sin().reshape(pair_shape), angle.cos().reshape(pair_shape) }, 2)
        .reshape(.{ 1, 2 * steps - 1, width });
}

fn halfFeedForward(w: Weights, name: []const u8, x: Tensor) Tensor {
    var key: [96]u8 = undefined;
    const expanded = t.linear(w, l.name(&key, "{s}.linear1", .{name}), x, false);
    return t.linear(w, l.name(&key, "{s}.linear2", .{name}), expanded.mul(expanded.sigmoid()), false).scale(@as(f32, 0.5));
}

fn attention(w: Weights, name: []const u8, x: Tensor, positional: Tensor, mask: ?Tensor) Tensor {
    var key: [96]u8 = undefined;
    const q = t.splitHeads(t.linear(w, l.name(&key, "{s}.q_proj", .{name}), x, false), heads);
    const k = t.splitHeads(t.linear(w, l.name(&key, "{s}.k_proj", .{name}), x, false), heads);
    const v = t.splitHeads(t.linear(w, l.name(&key, "{s}.v_proj", .{name}), x, false), heads);
    const pos = t.splitHeads(t.linear(w, l.name(&key, "{s}.relative_k_proj", .{name}), positional, false), heads);
    const u = w.get(name, ".bias_u").broadcast(q.shape(), &.{ 1, 3 });
    const b = w.get(name, ".bias_v").broadcast(q.shape(), &.{ 1, 3 });
    const content = q.add(u).dotGeneral(k, &.{.{ 3, 3 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
    const relative = q.add(b).dotGeneral(pos, &.{.{ 3, 3 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
    const steps = x.dim(1);
    const zero = Tensor.scalar(0, .f32).broad(.init(.{ 1, heads, steps, 1 }, .f32));
    const shifted = Tensor.concatenate(&.{ zero, relative }, 3).reshape(.{ 1, heads, 2 * steps, steps })
        .slice(2, .{ .start = 1 }).reshape(relative.shape()).slice(3, .{ .end = steps });
    const scale: f32 = 1 / @sqrt(@as(f32, head_width));
    const scores = content.scale(scale).add(shifted.scale(scale));
    const valid_scores = if (mask) |valid|
        valid.broadcast(scores.shape().withDtype(.bool), &.{3})
            .select(scores, Tensor.scalar(-std.math.inf(f32), .f32).broad(scores.shape()))
    else
        scores;
    const mixed = t.attend(valid_scores, v);
    return t.linear(w, l.name(&key, "{s}.o_proj", .{name}), mixed, false);
}

fn convolution(w: Weights, name: []const u8, x: Tensor, mask: ?Tensor) Tensor {
    var key: [96]u8 = undefined;
    const paired = t.linear(w, l.name(&key, "{s}.pointwise_conv1", .{name}), x, false);
    const gated = paired.slice(2, .{ .end = width }).mul(paired.slice(2, .{ .start = width }).sigmoid());
    const signal = (if (mask) |valid| maskSignal(gated, valid, 1) else gated).transpose(.{ 0, 2, 1 });
    const kernel = w.get(name, ".depthwise_conv.weight");
    const filtered = signal.conv1d(kernel, .{ .padding = &.{ 4, 4 }, .feature_group_count = width });
    const norm = l.batchNorm(w, l.name(&key, "{s}.norm", .{name}), filtered);
    const activated = norm.mul(norm.sigmoid()).transpose(.{ 0, 2, 1 });
    return t.linear(w, l.name(&key, "{s}.pointwise_conv2", .{name}), activated, false);
}

fn forwardMasked(parameters: []const Tensor, input: Tensor, valid: ?Tensor) Tensor {
    const w: Weights = .{ .schema = &schema, .tensors = parameters };
    var x = subsample(w, input, valid);
    const positional = positions(x.dim(1));
    const mask = if (valid) |length| validFrames(x.dim(1), length, 8) else null;
    for (0..blocks) |i| {
        var base: [48]u8 = undefined;
        var key: [96]u8 = undefined;
        const name = l.name(&base, "encoder.layers.{d}", .{i});
        const ff1 = t.norm(w, l.name(&key, "{s}.norm_feed_forward1", .{name}), x, 1e-5);
        x = x.add(halfFeedForward(w, l.name(&key, "{s}.feed_forward1", .{name}), ff1));
        const attn = t.norm(w, l.name(&key, "{s}.norm_self_att", .{name}), x, 1e-5);
        x = x.add(attention(w, l.name(&key, "{s}.self_attn", .{name}), attn, positional, mask));
        const conv = t.norm(w, l.name(&key, "{s}.norm_conv", .{name}), x, 1e-5);
        x = x.add(convolution(w, l.name(&key, "{s}.conv", .{name}), conv, mask));
        const ff2 = t.norm(w, l.name(&key, "{s}.norm_feed_forward2", .{name}), x, 1e-5);
        x = x.add(halfFeedForward(w, l.name(&key, "{s}.feed_forward2", .{name}), ff2));
        x = t.norm(w, l.name(&key, "{s}.norm_out", .{name}), x, 1e-5);
    }
    return x.reshape(.{ x.dim(1), width });
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    return forwardMasked(parameters, input, null);
}

fn forwardLive(parameters: []const Tensor, input: Tensor) Tensor {
    const length = input.slice(0, .{ .end = 1 }).slice(1, .{ .start = mels }).reshape(.{});
    return forwardMasked(parameters, input.slice(1, .{ .end = mels }), length);
}

pub fn outputFrames(frames: usize) usize {
    return (frames + 7) / 8;
}

pub const Encoder = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Encoder {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Encoder) void {
        self.graph.deinit();
    }

    pub fn encode(self: *Encoder, frames: usize, input: []const f32, output: []f32) !u32 {
        if (frames == 0 or frames > max_frames) return error.InvalidMelFrameCount;
        for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteMelInput;
        return self.graph.infer(forward, .init(.{ frames, mels }, .f32), .init(.{ outputFrames(frames), width }, .f32), input, output);
    }

    pub fn encodeLive(self: *Encoder, frames: usize, input: []const f32, padded_input: []f32, output: []f32) !u32 {
        if (frames == 0 or frames > live_frames or input.len != frames * mels) return error.InvalidMelFrameCount;
        if (padded_input.len != live_frames * (mels + 1)) return error.InvalidBufferSize;
        for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteMelInput;
        @memset(padded_input, 0);
        for (0..frames) |frame| {
            @memcpy(padded_input[frame * (mels + 1) ..][0..mels], input[frame * mels ..][0..mels]);
        }
        padded_input[mels] = @floatFromInt(frames);
        return self.graph.infer(forwardLive, .init(.{ live_frames, mels + 1 }, .f32), .init(.{ outputFrames(live_frames), width }, .f32), padded_input, output);
    }
};
