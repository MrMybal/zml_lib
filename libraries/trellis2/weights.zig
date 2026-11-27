const std = @import("std");
const rt = @import("zml_lib/runtime");
const Parameter = rt.layers.Parameter;

pub const width = 1536;
pub const heads = 12;
pub const head_width = width / heads;
pub const blocks = 30;
pub const context_width = 1024;
const intermediate = 8192;

pub const Profile = enum {
    structure,
    shape,
    texture,
    structure_multiview,
    shape_multiview,
    texture_multiview,

    pub fn inputWidth(self: Profile) i64 {
        return switch (self) {
            .structure, .structure_multiview => 8,
            .shape, .shape_multiview => 32,
            .texture, .texture_multiview => 64,
        };
    }

    pub fn outputWidth(self: Profile) i64 {
        return if (self == .texture or self == .texture_multiview) 32 else self.inputWidth();
    }

    pub fn projectionWidth(self: Profile) i64 {
        return switch (self) {
            .structure_multiview => 1024,
            .shape_multiview, .texture_multiview => 2048,
            else => 0,
        };
    }
};

fn spec(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn linear(comptime name: []const u8, fan_in: i64, fan_out: i64) [2]Parameter {
    return .{ spec(name ++ ".weight", .{ fan_out, fan_in }), spec(name ++ ".bias", .{fan_out}) };
}

fn attention(comptime scope: []const u8, comptime is_self: bool) [if (is_self) 6 else 8]Parameter {
    const input = if (is_self) linear(scope ++ ".to_qkv", width, 3 * width) else linear(scope ++ ".to_q", width, width) ++ linear(scope ++ ".to_kv", context_width, 2 * width);
    return input ++ linear(scope ++ ".to_out", width, width) ++ @as([2]Parameter, .{
        spec(scope ++ ".q_rms_norm.gamma", .{ heads, head_width }),
        spec(scope ++ ".k_rms_norm.gamma", .{ heads, head_width }),
    });
}

fn blockCount(comptime p: Profile) usize {
    return if (p.projectionWidth() == 0) 21 else 23;
}

pub fn schema(comptime p: Profile) [10 + blocks * blockCount(p)]Parameter {
    @setEvalBranchQuota(200_000);
    const prefix = linear("t_embedder.mlp.0", 256, width) ++ linear("t_embedder.mlp.2", width, width) ++
        linear("adaLN_modulation.1", width, 6 * width) ++
        linear("input_layer", p.inputWidth(), width) ++ linear("out_layer", width, p.outputWidth());
    var result: [prefix.len + blocks * blockCount(p)]Parameter = undefined;
    result[0..prefix.len].* = prefix;
    for (0..blocks) |i| {
        const k = std.fmt.comptimePrint("blocks.{d}.", .{i});
        const cross = if (p.projectionWidth() != 0) k ++ "cross_attn.cross_attn_block" else k ++ "cross_attn";
        const projection = if (p.projectionWidth() != 0) linear(k ++ "cross_attn.proj_linear", p.projectionWidth(), width) else [0]Parameter{};
        result[prefix.len + i * blockCount(p) ..][0..blockCount(p)].* =
            [3]Parameter{ spec(k ++ "modulation", .{6 * width}), spec(k ++ "norm2.weight", .{width}), spec(k ++ "norm2.bias", .{width}) } ++
            attention(k ++ "self_attn", true) ++ attention(cross, false) ++ projection ++
            linear(k ++ "mlp.mlp.0", width, intermediate) ++ linear(k ++ "mlp.mlp.2", intermediate, width);
    }
    return result;
}
