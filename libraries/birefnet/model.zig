const rt = @import("zml_lib/runtime");
const wd = @import("weights.zig");
const swin = @import("swin.zig");
const deform = @import("deform.zig");
const l = rt.layers;
const Tensor = l.Tensor;
pub const parameters = wd.schema;

fn spatial(x: Tensor, target: Tensor) Tensor {
    return rt.spatial.bilinearAlignCorners(x, target.dim(2), target.dim(3));
}

fn unit(w: l.Weights, x: Tensor, scope: []const u8) Tensor {
    var key: [128]u8 = undefined;
    var branch: [96]u8 = undefined;
    const conv_in = l.conv(w, l.name(&key, "{s}.conv_in", .{scope}), x, 1, @splat(1), true);
    const a = l.batchNorm(w, l.name(&key, "{s}.bn_in", .{scope}), conv_in).relu();
    var parts: [5]Tensor = undefined;
    parts[0] = deform.branch(w, a, l.name(&branch, "{s}.dec_att.aspp1", .{scope}), 1);
    for ([_]i64{ 1, 3, 7 }, 0..) |kernel, i|
        parts[i + 1] = deform.branch(w, a, l.name(&branch, "{s}.dec_att.aspp_deforms.{d}", .{ scope, i }), kernel);
    const average = a.mean(3).mean(2).reshape(.{ 1, 64, 1, 1 });
    const conv_pool = l.conv(w, l.name(&key, "{s}.dec_att.global_avg_pool.1", .{scope}), average, 1, @splat(0), false);
    const pool = l.batchNorm(w, l.name(&key, "{s}.dec_att.global_avg_pool.2", .{scope}), conv_pool).relu();
    parts[4] = pool.broad(parts[0].shape());
    const fused = l.conv(w, l.name(&key, "{s}.dec_att.conv1", .{scope}), Tensor.concatenate(&parts, 1), 1, @splat(0), false);
    const attention = l.batchNorm(w, l.name(&key, "{s}.dec_att.bn1", .{scope}), fused).relu();
    const conv_out = l.conv(w, l.name(&key, "{s}.conv_out", .{scope}), attention, 1, @splat(1), true);
    return l.batchNorm(w, l.name(&key, "{s}.bn_out", .{scope}), conv_out);
}

fn gate(w: l.Weights, x: Tensor, level: usize) Tensor {
    var key: [80]u8 = undefined;
    const conv = l.conv(w, l.name(&key, "decoder.gdt_convs_{d}.0", .{level}), x, 1, @splat(1), true);
    const residual = l.batchNorm(w, l.name(&key, "decoder.gdt_convs_{d}.1", .{level}), conv).relu();
    const mask = l.conv(w, l.name(&key, "decoder.gdt_convs_attn_{d}.0", .{level}), residual, 1, @splat(0), true).sigmoid();
    return x.mul(mask.broad(x.shape()));
}

fn imageInput(w: l.Weights, image: Tensor, target: Tensor, level: usize) Tensor {
    var key: [80]u8 = undefined;
    const h = target.dim(2);
    const width = target.dim(3);
    const hg = @divExact(image.dim(2), h);
    const wg = @divExact(image.dim(3), width);
    const blocks = image.reshape(.{ 1, 3, hg, h, wg, width }).transpose(.{ 0, 1, 2, 4, 3, 5 })
        .reshape(.{ 1, 3 * hg * wg, h, width });
    const first = l.conv(w, l.name(&key, "decoder.ipt_blk{d}.conv1", .{level}), blocks, 1, @splat(1), true);
    return l.conv(w, l.name(&key, "decoder.ipt_blk{d}.conv_out", .{level}), first, 1, @splat(1), true);
}

pub fn logits(params: []const Tensor, image: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &parameters, .tensors = params };
    const primary = swin.forward(w, image);
    const minor = swin.forward(w, rt.spatial.bilinearAlignCorners(image, @divExact(image.dim(2), 2), @divExact(image.dim(3), 2)));
    var pyramid: [4]Tensor = undefined;
    for (&pyramid, primary, minor) |*features, big, small|
        features.* = Tensor.concatenate(&.{ big, spatial(small, big) }, 1);
    const joint = Tensor.concatenate(&.{
        spatial(pyramid[0], pyramid[3]), spatial(pyramid[1], pyramid[3]), spatial(pyramid[2], pyramid[3]), pyramid[3],
    }, 1);
    var result = unit(w, joint, "squeeze_module.0");
    var key: [72]u8 = undefined;
    for (0..4) |i| {
        const level = 4 - i;
        const input = Tensor.concatenate(&.{ result, imageInput(w, image, result, level + 1) }, 1);
        result = unit(w, input, l.name(&key, "decoder.decoder_block{d}", .{level}));
        if (level > 1) {
            const side = pyramid[level - 2];
            const gated = spatial(gate(w, result, level), side);
            result = gated.add(l.conv(w, l.name(&key, "decoder.lateral_block{d}.conv", .{level}), side, 1, @splat(0), true));
        }
    }
    // A pointwise linear projection commutes with bilinear upsampling.
    // Upsample one logit channel instead of materializing 192 features at
    // full resolution and concatenating them with the 48 image features.
    const scope = "decoder.conv_out1.0";
    const kernel = w.get(scope, ".weight");
    const channels = result.dim(1);
    const coarse = l.conv2dParameters(result, kernel.slice(1, .{ .end = channels }), 1, @splat(0), 1);
    const detail = l.conv2dParameters(
        imageInput(w, image, image, 1),
        kernel.slice(1, .{ .start = channels }),
        1,
        @splat(0),
        1,
    );
    return l.biasAdd(spatial(coarse, image).add(detail), w.get(scope, ".bias"), 1);
}

pub fn forward(params: []const Tensor, image: Tensor) Tensor {
    return logits(params, image).sigmoid();
}
