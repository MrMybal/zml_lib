const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;

pub const Ddim = @import("ddim.zig").Ddim;
pub const Denoiser = @import("unet.zig").Denoiser;
pub const scale_factor: f32 = 0.9392935633659363;
pub const latent_channels = 8;
pub const latent_bands = 16;
pub const mel_bands = 64;
const expand = 4;
const epsilon: f32 = 1e-6;
const widths = [_]i64{ 512, 256, 128 };

fn pair(comptime name: []const u8, dims: anytype, bias: i64) [2]l.Parameter {
    return .{
        .{ .name = name ++ ".weight", .shape = .init(dims, .f32) },
        .{ .name = name ++ ".bias", .shape = .init(.{bias}, .f32) },
    };
}

fn resnetSchema(comptime name: []const u8, comptime input: i64, comptime output: i64) [8 + 2 * @as(usize, @intFromBool(input != output))]l.Parameter {
    return pair(name ++ ".norm1", .{input}, input) ++
        pair(name ++ ".conv1", .{ output, input, 3, 3 }, output) ++
        pair(name ++ ".norm2", .{output}, output) ++
        pair(name ++ ".conv2", .{ output, output, 3, 3 }, output) ++
        (if (input == output) [_]l.Parameter{} else pair(name ++ ".conv_shortcut", .{ output, input, 1, 1 }, output));
}

const schema = blk: {
    @setEvalBranchQuota(75_000);
    const middle = "decoder.mid_block.";
    const attn = middle ++ "attentions.0.";
    const first = pair("post_quant_conv", .{ 8, 8, 1, 1 }, 8) ++
        pair("decoder.conv_in", .{ 512, 8, 3, 3 }, 512) ++
        pair("decoder.conv_norm_out", .{128}, 128) ++
        pair("decoder.conv_out", .{ 1, 128, 3, 3 }, 1) ++
        resnetSchema(middle ++ "resnets.0", 512, 512) ++
        pair(attn ++ "group_norm", .{512}, 512) ++
        pair(attn ++ "query", .{ 512, 512 }, 512) ++
        pair(attn ++ "key", .{ 512, 512 }, 512) ++
        pair(attn ++ "value", .{ 512, 512 }, 512) ++
        pair(attn ++ "proj_attn", .{ 512, 512 }, 512) ++
        resnetSchema(middle ++ "resnets.1", 512, 512);
    var entries: [114]l.Parameter = undefined;
    entries[0..first.len].* = first;
    var offset = first.len;
    var input: i64 = widths[0];
    for (widths, 0..) |width, i| {
        const stage = std.fmt.comptimePrint("decoder.up_blocks.{d}", .{i});
        for (0..3) |j| {
            const name = stage ++ std.fmt.comptimePrint(".resnets.{d}", .{j});
            const block = resnetSchema(name, input, width);
            entries[offset..][0..block.len].* = block;
            offset += block.len;
            input = width;
        }
        if (i + 1 < widths.len) {
            entries[offset..][0..2].* = pair(stage ++ ".upsamplers.0.conv", .{ width, width, 3, 3 }, width);
            offset += 2;
        }
    }
    if (offset != entries.len) @compileError("AudioLDM decoder schema is incomplete");
    break :blk entries;
};

fn silu(x: Tensor) Tensor {
    return x.mul(x.sigmoid());
}

fn conv3(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    return l.conv(w, name, x, 1, @splat(1), true);
}

fn resnet(w: l.Weights, prefix: []const u8, x: Tensor) Tensor {
    var key: [80]u8 = undefined;
    const a = l.groupNorm(w, l.name(&key, "{s}.norm1", .{prefix}), x, 32, epsilon);
    const b = conv3(w, l.name(&key, "{s}.conv1", .{prefix}), silu(a));
    const c = l.groupNorm(w, l.name(&key, "{s}.norm2", .{prefix}), b, 32, epsilon);
    const y = conv3(w, l.name(&key, "{s}.conv2", .{prefix}), silu(c));
    const residual = if (x.dim(1) == y.dim(1)) x else l.conv(w, l.name(&key, "{s}.conv_shortcut", .{prefix}), x, 1, @splat(0), true);
    return y.add(residual);
}

fn attention(w: l.Weights, x: Tensor) Tensor {
    const prefix = "decoder.mid_block.attentions.0.";
    const dims = x.dims();
    const channels = dims[1];
    const normalized = l.groupNorm(w, prefix ++ "group_norm", x, 32, epsilon)
        .transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, dims[2] * dims[3], channels });
    const q = t.linear(w, prefix ++ "query", normalized, true);
    const k = t.linear(w, prefix ++ "key", normalized, true);
    const v = t.linear(w, prefix ++ "value", normalized, true);
    const y = t.linear(w, prefix ++ "proj_attn", t.attention(q, k, v, 1), true)
        .reshape(.{ 1, dims[2], dims[3], channels }).transpose(.{ 0, 3, 1, 2 });
    return x.add(y);
}

fn forward(parameters: []const Tensor, latent: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const post = l.conv(w, "post_quant_conv", latent.divByConst(scale_factor), 1, @splat(0), true);
    const start = conv3(w, "decoder.conv_in", post);
    const mid = resnet(w, "decoder.mid_block.resnets.0", start);
    var x = resnet(w, "decoder.mid_block.resnets.1", attention(w, mid));
    for (0..widths.len) |i| {
        var key: [64]u8 = undefined;
        for (0..3) |j| x = resnet(w, l.name(&key, "decoder.up_blocks.{d}.resnets.{d}", .{ i, j }), x);
        if (i + 1 < widths.len) {
            x = conv3(w, l.name(&key, "decoder.up_blocks.{d}.upsamplers.0.conv", .{i}), l.nearest2x(x));
        }
    }
    const y = conv3(w, "decoder.conv_out", silu(l.groupNorm(w, "decoder.conv_norm_out", x, 32, epsilon)));
    return y.reshape(.{ 1, y.dim(2), y.dim(3) });
}

pub fn melFrames(latent_frames: u32) !u32 {
    if (latent_frames == 0) return error.InvalidFrameCount;
    return std.math.mul(u32, latent_frames, expand);
}

pub const Decoder = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Decoder {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Decoder) void {
        self.graph.deinit();
    }

    pub fn decode(self: *Decoder, latent_frames: u32, latent: []const f32, mel: []f32) !u32 {
        const frames = try melFrames(latent_frames);
        return self.graph.infer(
            forward,
            .init(.{ 1, latent_channels, @as(i64, latent_frames), latent_bands }, .f32),
            .init(.{ 1, @as(i64, frames), mel_bands }, .f32),
            latent,
            mel,
        );
    }
};
