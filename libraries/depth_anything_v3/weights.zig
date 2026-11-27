const std = @import("std");
const rt = @import("zml_lib/runtime");
const dino = @import("zml_lib/dinov2");
const P = rt.layers.Parameter;
pub const width = 768;
pub const head_count = 12;
pub const projection_widths = [4]i64{ 96, 192, 384, 768 };
pub const backbone = "model.backbone.pretrained";
pub const decoder = "model.cam_dec";
pub const pyramid = "model.head.scratch";

fn parameter(comptime name: []const u8, shape: anytype) P {
    return .{ .name = name, .shape = .init(shape, .f32) };
}

fn pair(comptime name: []const u8, shape: anytype, channels: i64) [2]P {
    return .{ parameter(name ++ ".weight", shape), parameter(name ++ ".bias", .{channels}) };
}

fn backboneSchema() [207]P {
    var result: [207]P = undefined;
    const base = dino.schemaFor(.{ .channels = width, .heads = head_count, .layers = 12, .outputs = .{ 5, 7, 9, 11 } });
    var index: usize = 0;
    for (base) |tensor| {
        if (std.mem.eql(u8, tensor.name, "pretrained.mask_token")) continue;
        result[index] = .{
            .name = "model.backbone." ++ tensor.name,
            .shape = tensor.shape,
        };
        index += 1;
    }
    result[index] = parameter(backbone ++ ".camera_token", .{ 1, 2, width });
    index += 1;
    for (4..12) |block| {
        const name = std.fmt.comptimePrint("{s}.blocks.{d}.attn", .{ backbone, block });
        result[index..][0..4].* = pair(name ++ ".q_norm", .{64}, 64) ++ pair(name ++ ".k_norm", .{64}, 64);
        index += 4;
    }
    return result;
}

fn heads() [114]P {
    var result: [114]P = undefined;
    result[0..2].* = pair("model.head.norm", .{1536}, 1536);
    var next: usize = 2;
    for (projection_widths, 0..) |channels, i| {
        const project = std.fmt.comptimePrint("model.head.projects.{d}", .{i});
        const resizer = std.fmt.comptimePrint("model.head.resize_layers.{d}", .{i});
        result[next..][0..2].* = pair(project, .{ channels, 1536, 1, 1 }, channels);
        next += 2;
        if (i != 2) {
            const kernel: i64 = switch (i) { 0 => 4, 1 => 2, else => 3 };
            result[next..][0..2].* = pair(resizer, .{ channels, channels, kernel, kernel }, channels);
            next += 2;
        }
        const scratch = std.fmt.comptimePrint("{s}.layer{d}_rn.weight", .{ pyramid, i + 1 });
        result[next] = parameter(scratch, .{ 128, channels, 3, 3 });
        next += 1;
    }
    for ([_][]const u8{ "", "_aux" }) |suffix| {
        for (1..5) |level| {
            const scope = std.fmt.comptimePrint("{s}.refinenet{d}{s}", .{ pyramid, level, suffix });
            if (level != 4) {
                result[next..][0..4].* = pair(scope ++ ".resConfUnit1.conv1", .{ 128, 128, 3, 3 }, 128) ++
                    pair(scope ++ ".resConfUnit1.conv2", .{ 128, 128, 3, 3 }, 128);
                next += 4;
            }
            result[next..][0..6].* = pair(scope ++ ".resConfUnit2.conv1", .{ 128, 128, 3, 3 }, 128) ++
                pair(scope ++ ".resConfUnit2.conv2", .{ 128, 128, 3, 3 }, 128) ++
                pair(scope ++ ".out_conv", .{ 128, 128, 1, 1 }, 128);
            next += 6;
        }
    }
    result[next..][0..6].* = pair(pyramid ++ ".output_conv1", .{ 64, 128, 3, 3 }, 64) ++
        pair(pyramid ++ ".output_conv2.0", .{ 32, 64, 3, 3 }, 32) ++ pair(pyramid ++ ".output_conv2.2", .{ 2, 32, 1, 1 }, 2);
    next += 6;
    for (0..5) |i| {
        const scope = std.fmt.comptimePrint("{s}.output_conv1_aux.3.{d}", .{ pyramid, i });
        const out: i64 = if (i % 2 == 0) 64 else 128;
        result[next..][0..2].* = pair(scope, .{ out, 192 - out, 3, 3 }, out);
        next += 2;
    }
    result[next..][0..6].* = pair(pyramid ++ ".output_conv2_aux.3.0", .{ 32, 64, 3, 3 }, 32) ++
        pair(pyramid ++ ".output_conv2_aux.0.2", .{32}, 32) ++ pair(pyramid ++ ".output_conv2_aux.3.5", .{ 7, 32, 1, 1 }, 7);
    next += 6;
    if (next != result.len) unreachable;
    return result;
}

pub const schema = parameters: {
    @setEvalBranchQuota(250_000);
    break :parameters backboneSchema() ++ heads() ++
        pair(decoder ++ ".backbone.0", .{ 1536, 1536 }, 1536) ++ pair(decoder ++ ".backbone.2", .{ 1536, 1536 }, 1536) ++
        pair(decoder ++ ".fc_t", .{ 3, 1536 }, 3) ++ pair(decoder ++ ".fc_qvec", .{ 4, 1536 }, 4) ++
        pair(decoder ++ ".fc_fov.0", .{ 2, 1536 }, 2);
};
