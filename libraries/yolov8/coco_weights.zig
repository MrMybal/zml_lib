const std = @import("std");
const l = @import("zml_lib/runtime").layers;

fn conv(comptime prefix: []const u8, comptime in: i64, comptime out: i64, comptime kernel: i64) [5]l.Parameter {
    return .{
        .{ .name = prefix ++ ".conv.weight", .shape = .init(.{ out, in, kernel, kernel }, .f32) },
        .{ .name = prefix ++ ".bn.weight", .shape = .init(.{out}, .f32) },
        .{ .name = prefix ++ ".bn.bias", .shape = .init(.{out}, .f32) },
        .{ .name = prefix ++ ".bn.running_mean", .shape = .init(.{out}, .f32) },
        .{ .name = prefix ++ ".bn.running_var", .shape = .init(.{out}, .f32) },
    };
}

fn c2f(comptime name: []const u8, comptime in: i64, comptime out: i64, comptime repeats: usize) [(2 * repeats + 2) * 5]l.Parameter {
    const middle = @divExact(out, 2);
    var blocks: [repeats * 10]l.Parameter = undefined;
    for (0..repeats) |idx| {
        const prefix = std.fmt.comptimePrint("{s}.m.{d}", .{ name, idx });
        blocks[idx * 10 ..][0..10].* = conv(prefix ++ ".cv1", middle, middle, 3) ++ conv(prefix ++ ".cv2", middle, middle, 3);
    }
    return conv(name ++ ".cv1", in, out, 1) ++ blocks ++ conv(name ++ ".cv2", middle * @as(i64, 2 + repeats), out, 1);
}

fn head(comptime branch: usize, comptime index: usize, comptime in: i64, comptime middle: i64, comptime out: i64) [12]l.Parameter {
    const name = std.fmt.comptimePrint("model.22.cv{d}.{d}", .{ branch, index });
    return conv(name ++ ".0", in, middle, 3) ++ conv(name ++ ".1", middle, middle, 3) ++ [_]l.Parameter{
        .{ .name = name ++ ".2.weight", .shape = .init(.{ out, middle, 1, 1 }, .f32) },
        .{ .name = name ++ ".2.bias", .shape = .init(.{out}, .f32) },
    };
}

pub const schema = blk: {
    @setEvalBranchQuota(100_000);
    break :blk conv("model.0", 3, 80, 3) ++ conv("model.1", 80, 160, 3) ++
        c2f("model.2", 160, 160, 3) ++ conv("model.3", 160, 320, 3) ++
        c2f("model.4", 320, 320, 6) ++ conv("model.5", 320, 640, 3) ++
        c2f("model.6", 640, 640, 6) ++ conv("model.7", 640, 640, 3) ++
        c2f("model.8", 640, 640, 3) ++ conv("model.9.cv1", 640, 320, 1) ++ conv("model.9.cv2", 1280, 640, 1) ++
        c2f("model.12", 1280, 640, 3) ++ c2f("model.15", 960, 320, 3) ++ conv("model.16", 320, 320, 3) ++
        c2f("model.18", 960, 640, 3) ++ conv("model.19", 640, 640, 3) ++ c2f("model.21", 1280, 640, 3) ++
        head(2, 0, 320, 80, 64) ++ head(2, 1, 640, 80, 64) ++ head(2, 2, 640, 80, 64) ++
        head(3, 0, 320, 320, 80) ++ head(3, 1, 640, 320, 80) ++ head(3, 2, 640, 320, 80) ++
        [_]l.Parameter{.{ .name = "model.22.dfl.conv.weight", .shape = .init(.{ 1, 16, 1, 1 }, .f32) }};
};
