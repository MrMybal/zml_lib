const std = @import("std");
const rt = @import("zml_lib/runtime");
const Parameter = rt.layers.Parameter;

pub const Profile = struct {
    width: i64,
    heads: i64,
    blocks: usize,
    context: i64,
    grid: i64,
    io: i64,
};
pub const official: Profile = .{ .width = 1024, .heads = 16, .blocks = 24, .context = 1024, .grid = 16, .io = 128 };
pub const fixture: Profile = .{ .width = 16, .heads = 2, .blocks = 2, .context = 12, .grid = 3, .io = 8 };
// Matches the reduced DINOv2 token width and the decoder's 8³ output grid.
pub const pipeline_fixture: Profile = .{ .width = 16, .heads = 2, .blocks = 2, .context = 24, .grid = 2, .io = 8 };
pub const latent_width = 8;
pub const time_width = 256;
pub const fine_width = 28;
pub const coarse_width = 39;

fn tensor(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn pair(comptime name: []const u8, output: i64, dims: anytype) [2]Parameter {
    return .{ tensor(name ++ ".weight", dims), tensor(name ++ ".bias", .{output}) };
}

fn linear(comptime name: []const u8, input: i64, output: i64) [2]Parameter {
    return pair(name, output, .{ output, input });
}

fn spatialBlock(comptime name: []const u8, comptime input: i64, comptime output: i64, width: i64) [if (input == output) 8 else 10]Parameter {
    return pair(name ++ ".norm1", input, .{input}) ++
        pair(name ++ ".conv1.conv", output, .{ output, 3, 3, 3, input }) ++
        pair(name ++ ".conv2.conv", output, .{ output, 3, 3, 3, output }) ++
        linear(name ++ ".emb_layers.1", width, output * 2) ++
        (if (input == output) [0]Parameter{} else linear(name ++ ".skip_connection", input, output));
}

pub fn schema(comptime p: Profile, comptime sparse: bool) [p.blocks * 20 + (if (sparse) @as(usize, 46) else 9)]Parameter {
    @setEvalBranchQuota(200_000);
    const base = linear("t_embedder.mlp.0", time_width, p.width) ++ linear("t_embedder.mlp.2", p.width, p.width);
    const endpoints = if (sparse) linear("input_layer", latent_width, p.io) ++ linear("out_layer", p.io, latent_width) else linear("input_layer", latent_width, p.width) ++ linear("out_layer", p.width, latent_width) ++
        [1]Parameter{tensor("pos_emb", .{ p.grid * p.grid * p.grid, p.width })};
    const io = if (sparse) spatialBlock("input_blocks.0", p.io, p.io, p.width) ++
        spatialBlock("input_blocks.1", p.io, p.width, p.width) ++
        spatialBlock("out_blocks.0", 2 * p.width, p.io, p.width) ++
        spatialBlock("out_blocks.1", 2 * p.io, p.io, p.width) else [0]Parameter{};
    const prefix = base ++ endpoints ++ io;
    var result: [prefix.len + p.blocks * 20]Parameter = undefined;
    result[0..prefix.len].* = prefix;
    for (0..p.blocks) |i| {
        const k = std.fmt.comptimePrint("blocks.{d}.", .{i});
        const w = p.width;
        const head = @divExact(w, p.heads);
        result[prefix.len + i * 20 ..][0..20].* =
            linear(k ++ "adaLN_modulation.1", w, 6 * w) ++
            pair(k ++ "norm2", w, .{w}) ++
            linear(k ++ "self_attn.to_qkv", w, 3 * w) ++ linear(k ++ "self_attn.to_out", w, w) ++
            [_]Parameter{
                tensor(k ++ "self_attn.q_rms_norm.gamma", .{ p.heads, head }),
                tensor(k ++ "self_attn.k_rms_norm.gamma", .{ p.heads, head }),
            } ++
            linear(k ++ "cross_attn.to_q", w, w) ++ linear(k ++ "cross_attn.to_kv", p.context, 2 * w) ++
            linear(k ++ "cross_attn.to_out", w, w) ++
            linear(k ++ "mlp.mlp.0", w, 4 * w) ++ linear(k ++ "mlp.mlp.2", 4 * w, w);
    }
    return result;
}
