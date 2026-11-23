const std = @import("std");
const l = @import("layers.zig");
const t = @import("transformer.zig");
const TensorModel = @import("tensor_model.zig").TensorModel;
const Tensor = l.Tensor;

pub const Config = struct {
    channels: i64,
    scaling: f32,
    shift: f32 = 0,
    post_quant: bool = false,
};

pub fn Decoder(comptime config: Config) type {
    return struct {
        pub const channels = config.channels;
        pub const factor = 8;
        pub const schema = if (config.post_quant)
            decoder_schema ++ convolution("post_quant_conv", channels, channels, 1)
        else
            decoder_schema;

        const widths = [_]i64{ 128, 256, 512, 512 };
        const epsilon: f32 = 1e-6;

        fn convolution(comptime name: []const u8, out: i64, in: i64, k: i64) [2]l.Parameter {
            return .{
                .{ .name = name ++ ".weight", .shape = .init(.{ out, in, k, k }, .f32) },
                .{ .name = name ++ ".bias", .shape = .init(.{out}, .f32) },
            };
        }

        fn normalization(comptime name: []const u8, width: i64) [2]l.Parameter {
            return .{
                .{ .name = name ++ ".weight", .shape = .init(.{width}, .f32) },
                .{ .name = name ++ ".bias", .shape = .init(.{width}, .f32) },
            };
        }

        fn residualSchema(comptime name: []const u8, out: i64, in: i64) [8]l.Parameter {
            return normalization(name ++ ".norm1", in) ++ convolution(name ++ ".conv1", out, in, 3) ++
                normalization(name ++ ".norm2", out) ++ convolution(name ++ ".conv2", out, out, 3);
        }

        const decoder_schema = blk: {
            @setEvalBranchQuota(80_000);
            var specs: [138]l.Parameter = undefined;
            const mid = widths[3];
            const prefix = "decoder.mid.attn_1.";
            specs[0..32].* = convolution("decoder.conv_in", mid, channels, 3) ++
                convolution("decoder.conv_out", 3, widths[0], 3) ++ normalization("decoder.norm_out", widths[0]) ++
                residualSchema("decoder.mid.block_1", mid, mid) ++ residualSchema("decoder.mid.block_2", mid, mid) ++
                normalization(prefix ++ "norm", mid) ++ convolution(prefix ++ "q", mid, mid, 1) ++
                convolution(prefix ++ "k", mid, mid, 1) ++ convolution(prefix ++ "v", mid, mid, 1) ++
                convolution(prefix ++ "proj_out", mid, mid, 1);
            var index: usize = 32;
            var in = mid;
            for (0..4) |stage| {
                const level = 3 - stage;
                const out = widths[level];
                for (0..3) |block| {
                    const name = std.fmt.comptimePrint("decoder.up.{d}.block.{d}", .{ level, block });
                    specs[index..][0..8].* = residualSchema(name, out, in);
                    index += 8;
                    if (in != out) {
                        specs[index..][0..2].* = convolution(name ++ ".nin_shortcut", out, in, 1);
                        index += 2;
                    }
                    in = out;
                }
                if (level != 0) {
                    const name = std.fmt.comptimePrint("decoder.up.{d}.upsample.conv", .{level});
                    specs[index..][0..2].* = convolution(name, out, out, 3);
                    index += 2;
                }
            }
            std.debug.assert(index == specs.len);
            break :blk specs;
        };

        fn conv3(w: l.Weights, name: []const u8, x: Tensor) Tensor {
            return l.conv(w, name, x, 1, @splat(1), true);
        }

        fn conv1(w: l.Weights, name: []const u8, x: Tensor) Tensor {
            return l.conv(w, name, x, 1, @splat(0), true);
        }

        fn residual(w: l.Weights, name: []const u8, x: Tensor) Tensor {
            var key: [88]u8 = undefined;
            const a = l.groupNorm(w, l.name(&key, "{s}.norm1", .{name}), x, 32, epsilon).silu();
            const b = conv3(w, l.name(&key, "{s}.conv1", .{name}), a);
            const c = l.groupNorm(w, l.name(&key, "{s}.norm2", .{name}), b, 32, epsilon).silu();
            const result = conv3(w, l.name(&key, "{s}.conv2", .{name}), c);
            const skip = if (result.dim(1) == x.dim(1)) x else conv1(w, l.name(&key, "{s}.nin_shortcut", .{name}), x);
            return skip.add(result);
        }

        fn attention(w: l.Weights, x: Tensor) Tensor {
            const prefix = "decoder.mid.attn_1.";
            const n = l.groupNorm(w, prefix ++ "norm", x, 32, epsilon);
            const shape = .{ 1, x.dim(2) * x.dim(3), x.dim(1) };
            const q = conv1(w, prefix ++ "q", n).transpose(.{ 0, 2, 3, 1 }).reshape(shape);
            const k = conv1(w, prefix ++ "k", n).transpose(.{ 0, 2, 3, 1 }).reshape(shape);
            const v = conv1(w, prefix ++ "v", n).transpose(.{ 0, 2, 3, 1 }).reshape(shape);
            const a = t.attention(q, k, v, 1).reshape(.{ 1, x.dim(2), x.dim(3), x.dim(1) })
                .transpose(.{ 0, 3, 1, 2 });
            return x.add(conv1(w, prefix ++ "proj_out", a));
        }

        fn forward(parameters: []const Tensor, latent: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
            const normalized = latent.divByConst(config.scaling);
            const scaled = if (config.shift == 0) normalized else normalized.addConstant(config.shift);
            const source = if (config.post_quant) conv1(w, "post_quant_conv", scaled) else scaled;
            var x = conv3(w, "decoder.conv_in", source);
            x = residual(w, "decoder.mid.block_1", x);
            x = attention(w, x);
            x = residual(w, "decoder.mid.block_2", x);
            var key: [80]u8 = undefined;
            for (0..4) |stage| {
                const level = 3 - stage;
                for (0..3) |block| x = residual(w, l.name(&key, "decoder.up.{d}.block.{d}", .{ level, block }), x);
                if (level != 0) x = conv3(w, l.name(&key, "decoder.up.{d}.upsample.conv", .{level}), l.nearest2x(x));
            }
            return conv3(w, "decoder.conv_out", l.groupNorm(w, "decoder.norm_out", x, 32, epsilon).silu());
        }

        pub fn decode(
            model: *TensorModel,
            height: i64,
            width: i64,
            latent: []const f32,
            output: []f32,
        ) !u32 {
            return model.infer(forward, .init(.{ 1, channels, height, width }, .f32), .init(.{ 1, 3, height * factor, width * factor }, .f32), latent, output);
        }
    };
}
