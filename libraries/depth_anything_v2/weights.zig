const std = @import("std");
const Parameter = @import("zml_lib/runtime").layers.Parameter;
const dinov2 = @import("zml_lib/dinov2");

pub const channels = [_]i64{ 256, 512, 1024, 1024 };

fn parameter(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn pair(comptime name: []const u8, dims: anytype, bias: i64) [2]Parameter {
    return .{ parameter(name ++ ".weight", dims), parameter(name ++ ".bias", .{bias}) };
}

pub const schema = schemaFor("depth_head", dinov2.channels, channels, 256);

pub fn schemaFor(comptime prefix: []const u8, comptime embed: i64, comptime widths: [4]i64, comptime features: i64) [64]Parameter {
    @setEvalBranchQuota(50_000);
    var result: [64]Parameter = undefined;
    for (widths, 0..) |c, i| {
        const project = std.fmt.comptimePrint("{s}.projects.{d}", .{ prefix, i });
        result[i * 2 ..][0..2].* = pair(project, .{ c, embed, 1, 1 }, c);
        const layer = std.fmt.comptimePrint("{s}.scratch.layer{d}_rn.weight", .{ prefix, i + 1 });
        result[14 + i] = parameter(layer, .{ features, c, 3, 3 });

        const refine = std.fmt.comptimePrint("{s}.scratch.refinenet{d}", .{ prefix, i + 1 });
        const residual = .{ features, features, 3, 3 };
        result[18 + i * 10 ..][0..10].* =
            pair(refine ++ ".resConfUnit1.conv1", residual, features) ++
            pair(refine ++ ".resConfUnit1.conv2", residual, features) ++
            pair(refine ++ ".resConfUnit2.conv1", residual, features) ++
            pair(refine ++ ".resConfUnit2.conv2", residual, features) ++
            pair(refine ++ ".out_conv", .{ features, features, 1, 1 }, features);
    }
    result[8..14].* =
        pair(prefix ++ ".resize_layers.0", .{ widths[0], widths[0], 4, 4 }, widths[0]) ++
        pair(prefix ++ ".resize_layers.1", .{ widths[1], widths[1], 2, 2 }, widths[1]) ++
        pair(prefix ++ ".resize_layers.3", .{ widths[3], widths[3], 3, 3 }, widths[3]);
    result[58..64].* =
        pair(prefix ++ ".scratch.output_conv1", .{ features / 2, features, 3, 3 }, features / 2) ++
        pair(prefix ++ ".scratch.output_conv2.0", .{ 32, features / 2, 3, 3 }, 32) ++
        pair(prefix ++ ".scratch.output_conv2.2", .{ 1, 32, 1, 1 }, 1);
    return result;
}
