const std = @import("std");
const rt = @import("zml_lib/runtime");
const P = rt.layers.Parameter;

pub const Profile = enum {
    shape_encode,
    shape_decode,
    texture_encode,
    texture_decode,

    pub fn encoding(self: Profile) bool {
        return self == .shape_encode or self == .texture_encode;
    }

    pub fn finalWidth(self: Profile) i64 {
        return if (self.encoding()) 64 else if (self == .shape_decode) 7 else 6;
    }

    pub fn channels(self: Profile, level: usize) i64 {
        const widths = [5]i64{ 1024, 512, 256, 128, 64 };
        return widths[if (self.encoding()) 4 - level else level];
    }

    pub fn blocks(self: Profile, level: usize) usize {
        const counts = [5]usize{ 4, 16, 8, 4, 0 };
        return counts[if (self.encoding()) 4 - level else level];
    }
};

pub fn resolutionScope(comptime level: usize, comptime item: usize) []const u8 {
    return std.fmt.comptimePrint("blocks.{d}.{d}", .{ level, item });
}

fn pair(comptime name: []const u8, dimensions: anytype, out: i64) [2]P {
    return .{
        .{ .name = name ++ ".weight", .shape = .init(dimensions, .f32) },
        .{ .name = name ++ ".bias", .shape = .init(.{out}, .f32) },
    };
}

pub fn phaseCount(comptime p: Profile, comptime level: usize, comptime post: bool) usize {
    if (post) return 2;
    return 8 * p.blocks(level) + 4 + @as(usize, if (p == .shape_decode and level != 4) 2 else 0) + @as(usize, if (level == 0) 2 else 0) -
        @as(usize, if (level == 4) 2 else 0);
}

pub fn phaseSchema(comptime p: Profile, comptime level: usize, comptime post: bool) [phaseCount(p, level, post)]P {
    @setEvalBranchQuota(100_000);
    const c = p.channels(level);
    const scope = resolutionScope(level, p.blocks(level));
    if (post) {
        const target = p.channels(level + 1);
        return pair(scope ++ ".conv2", .{ target, 27 * target }, target);
    }
    var out: [phaseCount(p, level, post)]P = undefined;
    var first: usize = 0;
    if (level == 0) {
        out[0..2].* = if (p.encoding()) pair("input_layer", .{ c, 6 }, c) else pair("from_latent", .{ c, 32 }, c);
        first = 2;
    }
    for (0..p.blocks(level)) |idx| {
        const key = std.fmt.comptimePrint("blocks.{d}.{d}", .{ level, idx });
        out[first..][0..8].* = pair(key ++ ".conv", .{ c, 27 * c }, c) ++ pair(key ++ ".norm", .{c}, c) ++
            pair(key ++ ".mlp.0", .{ 4 * c, c }, 4 * c) ++ pair(key ++ ".mlp.2", .{ c, 4 * c }, c);
        first += 8;
    }
    if (level == 4) {
        const o = p.finalWidth();
        out[first..][0..2].* = if (p.encoding()) pair("to_latent", .{ o, c }, o) else pair("output_layer", .{ o, c }, o);
        first += 2;
    } else {
        const target = p.channels(level + 1);
        const conv = if (p.encoding()) @divExact(target, 8) else target * 8;
        out[first..][0..4].* = pair(scope ++ ".norm1", .{c}, c) ++ pair(scope ++ ".conv1", .{ conv, 27 * c }, conv);
        first += 4;
        if (p == .shape_decode) {
            out[first..][0..2].* = pair(scope ++ ".to_subdiv", .{ 8, c }, 8);
            first += 2;
        }
    }
    if (first != out.len) unreachable;
    return out;
}
