const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Ddim = @import("ddim.zig").Ddim;
const Tensor = l.Tensor;
const widths = [_]i64{ 128, 256, 384, 640 };
const epsilon: f32 = 1e-5;
pub const condition_width = 512;
pub const latent_channels = 8;
pub const latent_bands = 16;

const Schema = struct {
    entries: [690]l.Parameter = undefined,
    next: usize = 0,

    fn parameter(self: *Schema, comptime name: []const u8, dims: anytype) void {
        self.entries[self.next] = .{ .name = name, .shape = .init(dims, .f32) };
        self.next += 1;
    }

    fn affine(self: *Schema, comptime name: []const u8, dims: anytype, channels: i64) void {
        self.parameter(name ++ ".weight", dims);
        self.parameter(name ++ ".bias", .{channels});
    }

    fn resnet(self: *Schema, comptime name: []const u8, from: i64, to: i64) void {
        self.affine(name ++ ".norm1", .{from}, from);
        self.affine(name ++ ".conv1", .{ to, from, 3, 3 }, to);
        self.affine(name ++ ".norm2", .{to}, to);
        self.affine(name ++ ".conv2", .{ to, to, 3, 3 }, to);
        self.affine(name ++ ".time_emb_proj", .{ to, 1024 }, to);
        if (from != to) self.affine(name ++ ".conv_shortcut", .{ to, from, 1, 1 }, to);
    }

    fn attention(self: *Schema, comptime name: []const u8, c: i64) void {
        self.affine(name ++ ".norm", .{c}, c);
        self.affine(name ++ ".proj_in", .{ c, c, 1, 1 }, c);
        self.affine(name ++ ".proj_out", .{ c, c, 1, 1 }, c);
        const prefix = name ++ ".transformer_blocks.0.";
        inline for (.{ "norm1", "norm2", "norm3" }) |norm| self.affine(prefix ++ norm, .{c}, c);
        inline for (.{ "attn1", "attn2" }) |attn| {
            inline for (.{ ".to_q.weight", ".to_k.weight", ".to_v.weight" }) |projection| {
                self.parameter(prefix ++ attn ++ projection, .{ c, c });
            }
            self.affine(prefix ++ attn ++ ".to_out.0", .{ c, c }, c);
        }
        self.affine(prefix ++ "ff.net.0.proj", .{ 8 * c, c }, 8 * c);
        self.affine(prefix ++ "ff.net.2", .{ c, 4 * c }, c);
    }
};

const schema = blk: {
    @setEvalBranchQuota(200_000);
    var builder: Schema = .{};
    builder.affine("conv_in", .{ 128, latent_channels, 3, 3 }, 128);
    builder.affine("conv_norm_out", .{128}, 128);
    builder.affine("conv_out", .{ latent_channels, 128, 3, 3 }, latent_channels);
    builder.affine("time_embedding.linear_1", .{ 512, 128 }, 512);
    builder.affine("time_embedding.linear_2", .{ 512, 512 }, 512);
    builder.affine("class_embedding", .{ 512, condition_width }, 512);
    var c: i64 = 128;
    var skip_widths: [12]i64 = undefined;
    skip_widths[0] = c;
    var depth: usize = 1;
    for (widths, 0..) |width, i| {
        const prefix = std.fmt.comptimePrint("down_blocks.{d}.", .{i});
        for (0..2) |j| {
            builder.resnet(prefix ++ std.fmt.comptimePrint("resnets.{d}", .{j}), c, width);
            if (i > 0) builder.attention(prefix ++ std.fmt.comptimePrint("attentions.{d}", .{j}), width);
            c = width;
            skip_widths[depth] = c;
            depth += 1;
        }
        if (i + 1 < widths.len) {
            builder.affine(prefix ++ "downsamplers.0.conv", .{ c, c, 3, 3 }, c);
            skip_widths[depth] = c;
            depth += 1;
        }
    }
    builder.resnet("mid_block.resnets.0", c, c);
    builder.attention("mid_block.attentions.0", c);
    builder.resnet("mid_block.resnets.1", c, c);
    for (0..widths.len) |i| {
        const prefix = std.fmt.comptimePrint("up_blocks.{d}.", .{i});
        const width = widths[widths.len - 1 - i];
        for (0..3) |j| {
            depth -= 1;
            builder.resnet(prefix ++ std.fmt.comptimePrint("resnets.{d}", .{j}), c + skip_widths[depth], width);
            if (i + 1 < widths.len) builder.attention(prefix ++ std.fmt.comptimePrint("attentions.{d}", .{j}), width);
            c = width;
        }
        if (i + 1 < widths.len) builder.affine(prefix ++ "upsamplers.0.conv", .{ c, c, 3, 3 }, c);
    }
    if (builder.next != builder.entries.len) @compileError("AudioLDM UNet schema is incomplete");
    break :blk builder.entries;
};

fn silu(x: Tensor) Tensor {
    return x.mul(x.sigmoid());
}

fn resnet(w: l.Weights, prefix: []const u8, x: Tensor, time: Tensor) Tensor {
    var key: [96]u8 = undefined;
    const normalized = l.groupNorm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 32, epsilon);
    const a = l.conv(w, l.name(&key, "{s}.conv1", .{prefix}), silu(normalized), 1, @splat(1), true);
    const time_proj = t.linear(w, l.name(&key, "{s}.time_emb_proj", .{prefix}), time, true);
    const conditioned = a.add(time_proj.broadcast(a.shape(), &.{ 0, 1 }));
    const y = l.groupNorm(w, l.name(&key, "{s}.norm2", .{prefix}), conditioned, 32, epsilon);
    const output = l.conv(w, l.name(&key, "{s}.conv2", .{prefix}), silu(y), 1, @splat(1), true);
    const residual = if (output.dim(1) == x.dim(1)) x else l.conv(w, l.name(&key, "{s}.conv_shortcut", .{prefix}), x, 1, @splat(0), true);
    return output.add(residual);
}

fn attend(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    var key: [120]u8 = undefined;
    const q = t.linear(w, l.name(&key, "{s}.to_q", .{name}), x, false);
    const k = t.linear(w, l.name(&key, "{s}.to_k", .{name}), x, false);
    const v = t.linear(w, l.name(&key, "{s}.to_v", .{name}), x, false);
    return t.linear(w, l.name(&key, "{s}.to_out.0", .{name}), t.attention(q, k, v, 8), true);
}

fn transform(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    var key: [112]u8 = undefined;
    const n = l.groupNorm(w, l.name(&key, "{s}.norm", .{name}), x, 32, 1e-6);
    const a = l.conv(w, l.name(&key, "{s}.proj_in", .{name}), n, 1, @splat(0), true);
    const h = a.dim(2);
    const width = a.dim(3);
    const c = a.dim(1);
    var tokens = a.transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, h * width, c });
    var prefix_buf: [92]u8 = undefined;
    const prefix = l.name(&prefix_buf, "{s}.transformer_blocks.0", .{name});
    const n1 = t.norm(w, l.name(&key, "{s}.norm1", .{prefix}), tokens, epsilon);
    tokens = tokens.add(attend(w, l.name(&key, "{s}.attn1", .{prefix}), n1));
    // AudioLDM supplies CLAP conditioning through class_embedding. Both attentions act on the current sequence.
    const n2 = t.norm(w, l.name(&key, "{s}.norm2", .{prefix}), tokens, epsilon);
    tokens = tokens.add(attend(w, l.name(&key, "{s}.attn2", .{prefix}), n2));
    const n3 = t.norm(w, l.name(&key, "{s}.norm3", .{prefix}), tokens, epsilon);
    const pair = t.linear(w, l.name(&key, "{s}.ff.net.0.proj", .{prefix}), n3, true);
    const hidden = pair.slice(2, .{ .end = 4 * c }).mul(t.gelu(pair.slice(2, .{ .start = 4 * c })));
    tokens = tokens.add(t.linear(w, l.name(&key, "{s}.ff.net.2", .{prefix}), hidden, true));
    const restored = tokens.reshape(.{ 1, h, width, c }).transpose(.{ 0, 3, 1, 2 });
    return x.add(l.conv(w, l.name(&key, "{s}.proj_out", .{name}), restored, 1, @splat(0), true));
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const count = input.dim(0) - 128 - condition_width;
    const signal = input.slice(0, .{ .end = count }).reshape(.{ 1, latent_channels, @divExact(count, latent_channels * latent_bands), latent_bands });
    const time_input = input.slice(0, .{ .start = count, .end = count + 128 }).reshape(.{ 1, 128 });
    const text = input.slice(0, .{ .start = count + 128 }).reshape(.{ 1, condition_width });
    const projected = t.linear(w, "time_embedding.linear_2", silu(t.linear(w, "time_embedding.linear_1", time_input, true)), true);
    const time = silu(Tensor.concatenate(&.{ projected, t.linear(w, "class_embedding", text, true) }, 1));
    var x = l.conv(w, "conv_in", signal, 1, @splat(1), true);
    var skip: [12]Tensor = undefined;
    skip[0] = x;
    var depth: usize = 1;
    var name: [64]u8 = undefined;
    for (0..widths.len) |i| {
        for (0..2) |j| {
            x = resnet(w, l.name(&name, "down_blocks.{d}.resnets.{d}", .{ i, j }), x, time);
            if (i > 0) x = transform(w, l.name(&name, "down_blocks.{d}.attentions.{d}", .{ i, j }), x);
            skip[depth] = x;
            depth += 1;
        }
        if (i + 1 < widths.len) {
            x = l.conv(w, l.name(&name, "down_blocks.{d}.downsamplers.0.conv", .{i}), x, 2, @splat(1), true);
            skip[depth] = x;
            depth += 1;
        }
    }
    x = resnet(w, "mid_block.resnets.0", x, time);
    x = transform(w, "mid_block.attentions.0", x);
    x = resnet(w, "mid_block.resnets.1", x, time);
    for (0..widths.len) |i| {
        for (0..3) |j| {
            depth -= 1;
            x = resnet(w, l.name(&name, "up_blocks.{d}.resnets.{d}", .{ i, j }), Tensor.concatenate(&.{ x, skip[depth] }, 1), time);
            if (i + 1 < widths.len) x = transform(w, l.name(&name, "up_blocks.{d}.attentions.{d}", .{ i, j }), x);
        }
        if (i + 1 < widths.len) x = l.conv(w, l.name(&name, "up_blocks.{d}.upsamplers.0.conv", .{i}), l.nearest2x(x), 1, @splat(1), true);
    }
    return l.conv(w, "conv_out", silu(l.groupNorm(w, "conv_norm_out", x, 32, epsilon)), 1, @splat(1), true).flatten();
}

pub const Denoiser = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Denoiser {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Denoiser) void {
        self.graph.deinit();
    }

    pub fn predict(self: *Denoiser, frames: u32, timestep: usize, latent: []const f32, condition: []const f32, noise: []f32) !u32 {
        if (frames == 0 or frames % 8 != 0 or timestep >= 1000) return error.InvalidDiffusionShape;
        const count: usize = @as(usize, frames) * latent_channels * latent_bands;
        if (latent.len != count or noise.len != count or condition.len != condition_width) return error.InvalidBufferSize;
        const allocator = self.graph.session.allocator;
        const input = try allocator.alloc(f32, count + 128 + condition_width);
        defer allocator.free(input);
        @memcpy(input[0..count], latent);
        for (0..64) |i| {
            const phase = @as(f32, @floatFromInt(timestep)) * @exp(-@log(@as(f32, 10000)) * @as(f32, @floatFromInt(i)) / 64);
            input[count + i] = @cos(phase);
            input[count + 64 + i] = @sin(phase);
        }
        @memcpy(input[count + 128 ..], condition);
        return self.graph.infer(forward, .init(.{@as(i64, @intCast(input.len))}, .f32), .init(.{@as(i64, @intCast(count))}, .f32), input, noise);
    }

    pub fn generateLatent(self: *Denoiser, frames: u32, steps: usize, latent: []f32, condition: []const f32, negative: ?[]const f32, guidance: f32) !u32 {
        const schedule = try Ddim.init(steps);
        if (!std.math.isFinite(guidance) or guidance < 1 or (guidance > 1 and negative == null)) return error.InvalidGuidance;
        const allocator = self.graph.session.allocator;
        const noise = try allocator.alloc(f32, latent.len);
        defer allocator.free(noise);
        const unconditioned = try allocator.alloc(f32, if (guidance > 1) latent.len else 0);
        defer allocator.free(unconditioned);
        var shards: u32 = 0;
        for (0..steps) |i| {
            const index = try schedule.timestep(i);
            shards = try self.predict(frames, index, latent, condition, noise);
            if (guidance > 1) {
                if (try self.predict(frames, index, latent, negative.?, unconditioned) != shards) return error.OutputPlacementMismatch;
                for (noise, unconditioned) |*positive, none| positive.* = none + guidance * (positive.* - none);
            }
            try schedule.step(i, latent, noise);
        }
        return shards;
    }
};
