const std = @import("std");
const rt = @import("zml_lib/runtime");
const dino = @import("zml_lib/dinov2");
const P = rt.layers.Parameter;

pub const backbone: dino.Architecture = arch: {
    var arch = dino.large;
    arch.outputs = .{ 5, 11, 17, 23 };
    break :arch arch;
};

pub const widths = [5]i64{ 1024, 256, 128, 64, 32 };

fn pair(comptime name: []const u8, dims: anytype, out: i64) [2]P {
    return .{
        .{ .name = name ++ ".weight", .shape = .init(dims, .f32) },
        .{ .name = name ++ ".bias", .shape = .init(.{out}, .f32) },
    };
}

fn stack(comptime name: []const u8, comptime out: i64) [if (out == 0) 48 else 38]P {
    var result: [if (out == 0) 48 else 38]P = undefined;
    var next: usize = 0;
    inline for (widths, 0..) |channels, level| {
        const index = std.fmt.comptimePrint("{d}", .{level});
        const inputs: i64 = if (out != 0) channels else if (level == 0) 1026 else 2;
        result[next..][0..2].* = pair(name ++ ".input_blocks." ++ index, .{ channels, inputs, 1, 1 }, channels);
        next += 2;
        if (level > 0 and level < 4) {
            inline for (0..if (out == 0) 2 else 1) |block| {
                const scope = std.fmt.comptimePrint("{s}.res_blocks.{d}.{d}.layers", .{ name, level, block });
                result[next..][0..2].* = pair(scope ++ ".2", .{ channels, channels, 3, 3 }, channels);
                result[next + 2 ..][0..2].* = pair(scope ++ ".5", .{ channels, channels, 3, 3 }, channels);
                next += 4;
            }
        }
        if (level < 4) {
            const successor = widths[level + 1];
            const scope = name ++ ".resamplers." ++ index;
            if (level < 3) {
                result[next..][0..2].* = pair(scope ++ ".0", .{ channels, successor, 2, 2 }, successor);
                next += 2;
            }
            result[next..][0..2].* = pair(scope ++ ".1", .{ successor, if (level < 3) successor else channels, 3, 3 }, successor);
            next += 2;
        } else if (out != 0) {
            result[next..][0..2].* = pair(name ++ ".output_blocks.4", .{ out, channels, 1, 1 }, out);
            next += 2;
        }
    }
    if (next != result.len) unreachable;
    return result;
}

fn projection() [10]P {
    var result: [10]P = undefined;
    result[0] = .{ .name = "encoder.image_mean", .shape = .init(.{ 1, 3, 1, 1 }, .f32) };
    result[1] = .{ .name = "encoder.image_std", .shape = .init(.{ 1, 3, 1, 1 }, .f32) };
    for (0..4) |index| {
        const name = std.fmt.comptimePrint("encoder.output_projections.{d}", .{index});
        result[2 + 2 * index ..][0..2].* = pair(name, .{ 1024, 1024, 1, 1 }, 1024);
    }
    return result;
}

pub const backbone_schema = dino.schemaFor(backbone);
pub fn schema(comptime normals: bool) [if (normals) 521 else 483]P {
    @setEvalBranchQuota(250_000);
    const scale = pair("scale_head.0", .{ 1024, 1024 }, 1024) ++
        pair("scale_head.2", .{ 1024, 1024 }, 1024) ++ pair("scale_head.4", .{ 1, 1024 }, 1);
    return backbone_schema ++ projection() ++ stack("neck", 0) ++ stack("points_head", 3) ++ stack("mask_head", 1) ++
        (if (normals) stack("normal_head", 3) else @as([0]P, .{})) ++ scale;
}
