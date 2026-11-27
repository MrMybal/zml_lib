const std = @import("std");
const rt = @import("zml_lib/runtime");
const P = rt.layers.Parameter;

pub const window = 12;
pub const channels = [4]i64{ 192, 384, 768, 1536 };
pub const depths = [4]usize{ 2, 2, 18, 2 };
pub const decoder_input = [4]i64{ 480, 960, 1920, 3456 };
pub const decoder_output = [4]i64{ 192, 384, 768, 1536 };

fn entry(comptime key: []const u8, dims: anytype) P {
    return .{ .name = key, .shape = .init(dims, .f32) };
}

fn pair(comptime key: []const u8, dims: anytype, n: i64) [2]P {
    return .{ entry(key ++ ".weight", dims), entry(key ++ ".bias", .{n}) };
}

fn bn(comptime key: []const u8, n: i64) [4]P {
    return pair(key, .{n}, n) ++ .{
        entry(key ++ ".running_mean", .{n}), entry(key ++ ".running_var", .{n}),
    };
}

fn deform(comptime key: []const u8, comptime k: i64) [9]P {
    return pair(key ++ ".atrous_conv.offset_conv", .{ k * k * 2, 64, k, k }, k * k * 2) ++
        pair(key ++ ".atrous_conv.modulator_conv", .{ k * k, 64, k, k }, k * k) ++
        .{entry(key ++ ".atrous_conv.regular_conv.weight", .{ 256, 64, k, k })} ++ bn(key ++ ".bn", 256);
}

fn unit(comptime key: []const u8, in: i64, out: i64) [58]P {
    const attention = key ++ ".dec_att";
    return pair(key ++ ".conv_in", .{ 64, in, 3, 3 }, 64) ++ bn(key ++ ".bn_in", 64) ++
        deform(attention ++ ".aspp1", 1) ++ deform(attention ++ ".aspp_deforms.0", 1) ++
        deform(attention ++ ".aspp_deforms.1", 3) ++ deform(attention ++ ".aspp_deforms.2", 7) ++
        .{entry(attention ++ ".global_avg_pool.1.weight", .{ 256, 64, 1, 1 })} ++ bn(attention ++ ".global_avg_pool.2", 256) ++
        .{entry(attention ++ ".conv1.weight", .{ 64, 1280, 1, 1 })} ++ bn(attention ++ ".bn1", 64) ++
        pair(key ++ ".conv_out", .{ out, 64, 3, 3 }, out) ++ bn(key ++ ".bn_out", out);
}

fn backbone() [333]P {
    var out: [333]P = undefined;
    out[0..4].* = pair("bb.patch_embed.proj", .{ 192, 3, 4, 4 }, 192) ++ pair("bb.patch_embed.norm", .{192}, 192);
    var i: usize = 4;
    inline for (channels, depths, 0..) |c, count, level| {
        const layer = std.fmt.comptimePrint("bb.layers.{d}", .{level});
        inline for (0..count) |j| {
            const key = std.fmt.comptimePrint("{s}.blocks.{d}", .{ layer, j });
            out[i..][0..13].* = pair(key ++ ".attn.qkv", .{ 3 * c, c }, 3 * c) ++ pair(key ++ ".attn.proj", .{ c, c }, c) ++
                .{entry(key ++ ".attn.relative_bias", .{ @divExact(c, 32), window * window, window * window })} ++
                pair(key ++ ".mlp.fc1", .{ 4 * c, c }, 4 * c) ++ pair(key ++ ".mlp.fc2", .{ c, 4 * c }, c) ++
                pair(key ++ ".norm1", .{c}, c) ++ pair(key ++ ".norm2", .{c}, c);
            i += 13;
        }
        out[i..][0..2].* = pair(std.fmt.comptimePrint("bb.norm{d}", .{level}), .{c}, c);
        i += 2;
        if (level < 3) {
            out[i..][0..3].* = pair(layer ++ ".downsample.norm", .{4 * c}, 4 * c) ++
                .{entry(layer ++ ".downsample.reduction.weight", .{ 2 * c, 4 * c })};
            i += 3;
        }
    }
    if (i != out.len) unreachable;
    return out;
}

fn matting() [342]P {
    var out: [342]P = undefined;
    out[0..58].* = unit("squeeze_module.0", 5760, 3072);
    var i: usize = 58;
    inline for (decoder_input, decoder_output, 0..) |input, output, level| {
        const key = std.fmt.comptimePrint("decoder.decoder_block{d}", .{level + 1});
        out[i..][0..58].* = unit(key, input, output);
        i += 58;
        if (level > 0) {
            const index = std.fmt.comptimePrint("{d}", .{level + 1});
            out[i..][0..10].* = pair("decoder.gdt_convs_" ++ index ++ ".0", .{ 16, output, 3, 3 }, 16) ++
                bn("decoder.gdt_convs_" ++ index ++ ".1", 16) ++ pair("decoder.gdt_convs_attn_" ++ index ++ ".0", .{ 1, 16, 1, 1 }, 1) ++
                pair("decoder.lateral_block" ++ index ++ ".conv", .{ output, output, 1, 1 }, output);
            i += 10;
        }
    }
    inline for ([_]i64{ 3, 48, 192, 768, 3072 }, [_]i64{ 48, 96, 192, 384, 384 }, 1..) |c, output, level| {
        const key = std.fmt.comptimePrint("decoder.ipt_blk{d}", .{level});
        out[i..][0..4].* = pair(key ++ ".conv1", .{ 64, c, 3, 3 }, 64) ++ pair(key ++ ".conv_out", .{ output, 64, 3, 3 }, output);
        i += 4;
    }
    out[i..][0..2].* = pair("decoder.conv_out1.0", .{ 1, 240, 1, 1 }, 1);
    i += 2;
    if (i != out.len) unreachable;
    return out;
}

pub const schema = params: {
    @setEvalBranchQuota(500_000);
    break :params backbone() ++ matting();
};
