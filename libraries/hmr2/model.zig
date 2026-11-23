const std = @import("std");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const t = rt.transformer;
const vit = rt.pose_vit;
const Tensor = l.Tensor;
const parameter = vit.parameter;
const prefix = "smpl_head.transformer";
const width = 1024;
const heads = 8;
const inner = 512;
const depth = 6;

pub const input_dims = .{ 3, vit.height, vit.width };
pub const output_dims = .{width};
pub const schema = blk: {
    @setEvalBranchQuota(100_000);
    var layers: [depth * 17]l.Parameter = undefined;
    for (0..depth) |idx| {
        const base = std.fmt.comptimePrint(prefix ++ ".transformer.layers.{d}.", .{idx});
        layers[idx * 17 ..][0..17].* = .{
            parameter(base ++ "0.norm.weight", .{width}),
            parameter(base ++ "0.norm.bias", .{width}),
            parameter(base ++ "0.fn.to_qkv.weight", .{ inner * 3, width }),
            parameter(base ++ "0.fn.to_out.0.weight", .{ width, inner }),
            parameter(base ++ "0.fn.to_out.0.bias", .{width}),
            parameter(base ++ "1.norm.weight", .{width}),
            parameter(base ++ "1.norm.bias", .{width}),
            parameter(base ++ "1.fn.to_kv.weight", .{ inner * 2, vit.channels }),
            parameter(base ++ "1.fn.to_q.weight", .{ inner, width }),
            parameter(base ++ "1.fn.to_out.0.weight", .{ width, inner }),
            parameter(base ++ "1.fn.to_out.0.bias", .{width}),
            parameter(base ++ "2.norm.weight", .{width}),
            parameter(base ++ "2.norm.bias", .{width}),
            parameter(base ++ "2.fn.net.0.weight", .{ width, width }),
            parameter(base ++ "2.fn.net.0.bias", .{width}),
            parameter(base ++ "2.fn.net.3.weight", .{ width, width }),
            parameter(base ++ "2.fn.net.3.bias", .{width}),
        };
    }
    break :blk vit.schema ++ [_]l.Parameter{
        parameter(prefix ++ ".to_token_embedding.weight", .{ width, 1 }),
        parameter(prefix ++ ".to_token_embedding.bias", .{width}),
        parameter(prefix ++ ".pos_embedding", .{ 1, 1, width }),
    } ++ layers;
};

fn block(w: l.Weights, token: Tensor, context: Tensor, index: usize) Tensor {
    var buffer: [100]u8 = undefined;
    const layer = prefix ++ ".transformer.layers.{d}.";
    const normalized = t.norm(w, l.name(&buffer, layer ++ "0.norm", .{index}), token, 1e-5);
    // The self-attention has exactly one token, so its softmax is one.
    const value = t.linear(w, l.name(&buffer, layer ++ "0.fn.to_qkv", .{index}), normalized, false)
        .slice(2, .{ .start = 2 * inner });
    const self = token.add(t.linear(w, l.name(&buffer, layer ++ "0.fn.to_out.0", .{index}), value, true));
    const cross_input = t.norm(w, l.name(&buffer, layer ++ "1.norm", .{index}), self, 1e-5);
    const query = t.linear(w, l.name(&buffer, layer ++ "1.fn.to_q", .{index}), cross_input, false);
    const kv = t.linear(w, l.name(&buffer, layer ++ "1.fn.to_kv", .{index}), context, false)
        .reshape(.{ context.dim(0), vit.patch_count, 2, inner });
    const cross = t.attention(query, kv.slice(2, .single(0)), kv.slice(2, .single(1)), heads);
    const attended = self.add(t.linear(w, l.name(&buffer, layer ++ "1.fn.to_out.0", .{index}), cross, true));
    const latent = t.norm(w, l.name(&buffer, layer ++ "2.norm", .{index}), attended, 1e-5);
    const hidden = t.gelu(t.linear(w, l.name(&buffer, layer ++ "2.fn.net.0", .{index}), latent, true));
    return attended.add(t.linear(w, l.name(&buffer, layer ++ "2.fn.net.3", .{index}), hidden, true));
}

pub fn forward(params: []const Tensor, image: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = params };
    const context = vit.encode(w, image);
    const zeros = Tensor.scalar(@as(f32, 0), .f32).broad(.init(.{ image.dim(0), 1, 1 }, .f32));
    var token = t.linear(w, prefix ++ ".to_token_embedding", zeros, true);
    token = token.add(w.get(prefix ++ ".pos_embedding", "").broad(token.shape()));
    for (0..depth) |index| token = block(w, token, context, index);
    return token.squeeze(1);
}
