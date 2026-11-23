const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const p = @import("sift_scale.zig");
const detect = @import("sift_detect.zig");
const desc = @import("sift_descriptor.zig");
const m = @import("post_math.zig");
const rgb_scale = @import("pose_scale.zig");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;

pub const std_options: std.Options = .{ .log_level = .info };

const capacities = [_]i64{ 3584, 1408, 640, 288, 112, 48, 32, 24 };
const feature_capacity = 4096;
const Stage = rt.Compiled;

fn check(session: *const rt.Session, buffer: *const Buffer) !void {
    rt.probe_io.placement(session, try session.verifyPlacement(buffer));
}

fn halfKeyframe(image: Tensor) Tensor {
    const reduced = rgb_scale.forward(image.reshape(.{ 1, image.dim(0), image.dim(1), 3 }));
    return reduced.reshape(.{ reduced.dim(1), reduced.dim(2), 3 });
}

const Octave = struct {
    space: Stage,
    candidates: Stage,
    orientations: Stage,
    descriptors: Stage,
    feature_shape: zml.Shape,

    fn init(session: *const rt.Session, h: i64, w: i64, cap: i64, pixel_scale: f32) !Octave {
        const base = Tensor.init(.{ h, w }, .f32);
        const space = Tensor.init(.{ p.levels, h, w }, .f32);
        const points = Tensor.init(.{ cap, detect.point_width }, .f32);
        const field = Tensor.init(.{ p.levels * h * w, 2 }, .f32);
        const oriented = Tensor.init(.{ cap, desc.oriented_width }, .f32);

        const pyramid = try Stage.init(session, p.octave, .{base});
        errdefer pyramid.deinit(session.allocator);
        const candidates = try Stage.init(session, detect.forward, .{ .{ .capacity = cap }, space });
        errdefer candidates.deinit(session.allocator);
        const orientations = try Stage.init(session, desc.orient, .{ space, points });
        errdefer orientations.deinit(session.allocator);
        const descriptors = try Stage.init(session, desc.describe, .{ .{ .pixel_scale = pixel_scale }, field, oriented, .{ .width = w, .height = h } });
        return .{
            .space = pyramid,
            .candidates = candidates,
            .orientations = orientations,
            .descriptors = descriptors,
            .feature_shape = .init(.{ cap, desc.feature_width }, .f32),
        };
    }

    fn deinit(self: *const Octave, allocator: std.mem.Allocator) void {
        self.descriptors.deinit(allocator);
        self.orientations.deinit(allocator);
        self.candidates.deinit(allocator);
        self.space.deinit(allocator);
    }

    fn run(self: *Octave, session: *const rt.Session, base: Buffer) !struct { Buffer, Buffer } {
        try self.space.run(.{base});
        var pyramid = self.space.results.get(struct { Buffer, Buffer });
        defer pyramid[0].deinit();
        errdefer pyramid[1].deinit();
        try check(session, &pyramid[0]);
        try check(session, &pyramid[1]);

        try self.candidates.run(.{pyramid[0]});
        var points = self.candidates.results.get(Buffer);
        defer points.deinit();
        try check(session, &points);

        try self.orientations.run(.{ pyramid[0], points });
        var orientation = self.orientations.results.get(struct { Buffer, Buffer });
        defer orientation[0].deinit();
        defer orientation[1].deinit();
        try check(session, &orientation[0]);
        try check(session, &orientation[1]);

        try self.descriptors.run(.{ orientation[0], orientation[1] });
        var features = self.descriptors.results.get(Buffer);
        errdefer features.deinit();
        try check(session, &features);
        return .{ pyramid[1], features };
    }
};

fn pack(octaves: [capacities.len]Tensor) Tensor {
    const features = Tensor.concatenate(&octaves, 0);
    return p.compact(features, m.component(features, 2).cmp(.GT, m.scalar(0)), feature_capacity);
}

fn extract(session: *const rt.Session, initial: *Stage, octaves: []Octave, packing: *Stage, image: Buffer, empty: Buffer) !Buffer {
    try initial.run(.{image});
    var base = initial.results.get(Buffer);
    defer base.deinit();
    try check(session, &base);
    var all: [capacities.len]Buffer = @splat(empty);
    var ready: usize = 0;
    defer for (all[0..ready]) |*buffer| buffer.deinit();
    for (octaves, 0..) |*octave, index| {
        const next, const features = try octave.run(session, base);
        base.deinit();
        base = next;
        all[index] = features;
        ready += 1;
    }
    try packing.run(.{all});
    var result = packing.results.get(Buffer);
    errdefer result.deinit();
    try check(session, &result);
    return result;
}

pub fn main(init: std.process.Init) !void {
    const command = try init.minimal.args.toSlice(init.arena.allocator());
    const halve = command.len > 1 and std.mem.eql(u8, command[1], "half");
    const args = command[1 + @as(usize, @intFromBool(halve)) ..];
    if (args.len < 7) return error.ExpectedBackendHeightWidthKeyframesInputOutputRepeats;
    var extras = args[7..];
    const reduced_path: ?[]const u8 = if (halve and extras.len >= 2 and std.mem.eql(u8, extras[0], "--reduced-rgb")) blk: {
        const path = extras[1];
        extras = extras[2..];
        break :blk path;
    } else null;
    if (extras.len > 1 or (halve and extras.len == 1 and std.mem.eql(u8, extras[0], "--reduced-rgb"))) return error.InvalidKeyframeOptions;

    const backend = std.meta.stringToEnum(rt.Backend, args[0]) orelse return error.InvalidBackend;
    const input_height = try std.fmt.parseInt(u32, args[1], 10);
    const input_width = try std.fmt.parseInt(u32, args[2], 10);
    const frames = try std.fmt.parseInt(u32, args[3], 10);
    const repeats = try std.fmt.parseInt(u16, args[6], 10);
    if (input_height == 0 or input_width == 0 or input_height > 8192 or input_width > 8192) return error.InvalidKeyframeDimensions;
    const height = if (halve) rgb_scale.halfSize(input_height) else input_height;
    const width = if (halve) rgb_scale.halfSize(input_width) else input_width;
    if (height < p.minimum_side / 2 or width < p.minimum_side / 2 or height > 4096 or width > 4096 or
        width * height > 2_097_152 or frames < 2 or frames > 514 or repeats == 0)
        return error.InvalidKeyframeDimensions;

    const allocator = std.heap.page_allocator;
    const io = init.io;
    const input = try std.Io.Dir.cwd().openFile(io, args[4], .{});
    defer input.close(io);
    const input_shape: zml.Shape = .init(.{ input_height, input_width, 3 }, .u8);
    const image_shape: zml.Shape = .init(.{ height, width, 3 }, .u8);
    const image = try allocator.alloc(u8, input_shape.count());
    defer allocator.free(image);
    if (try input.length(io) != frames * image.len) return error.InvalidKeyframeBuffer;
    const match_shape: zml.Shape = .init(.{ feature_capacity, desc.match_width }, .f32);
    const feature_shape: zml.Shape = .init(.{ feature_capacity, desc.feature_width }, .f32);
    const matches = try allocator.alloc(f32, (frames - 1) * match_shape.count());
    defer allocator.free(matches);
    const diagnostics = if (extras.len == 1) try allocator.alloc(f32, frames * feature_shape.count()) else null;
    defer if (diagnostics) |buffer| allocator.free(buffer);

    const session = try rt.Session.open(allocator, io, backend);
    defer session.close();
    var resize = if (halve) try Stage.init(&session, halfKeyframe, .{Tensor.fromShape(input_shape)}) else null;
    defer if (resize) |*model| model.deinit(allocator);
    std.log.info("Compiling SIFT base image for {d} by {d} keyframes", .{ width, height });
    var initial = try Stage.init(&session, p.first, .{Tensor.fromShape(image_shape)});
    defer initial.deinit(allocator);
    var octaves: [capacities.len]Octave = undefined;
    var used: usize = 0;
    defer for (octaves[0..used]) |*octave| octave.deinit(allocator);
    const empty_shape: zml.Shape = .init(.{ 1, desc.feature_width }, .f32);
    var signatures: [capacities.len]Tensor = undefined;
    for (&signatures) |*signature| signature.* = Tensor.fromShape(empty_shape);
    var h: i64 = 2 * @as(i64, height);
    var w: i64 = 2 * @as(i64, width);
    var scale: f32 = 0.5;
    while (used < octaves.len and @min(h, w) >= p.minimum_side) : (used += 1) {
        const cap = @min(capacities[used], @max(16, @divFloor(h * w, 32)));
        std.log.info("Compiling SIFT octave {d} at {d} by {d}", .{ used, w, h });
        octaves[used] = try Octave.init(&session, h, w, cap, scale);
        signatures[used] = Tensor.fromShape(octaves[used].feature_shape);
        h = @divFloor(h, 2);
        w = @divFloor(w, 2);
        scale *= 2;
    }
    var packing = try Stage.init(&session, pack, .{signatures});
    defer packing.deinit(allocator);
    var matching = try Stage.init(&session, desc.match, .{ Tensor.fromShape(feature_shape), Tensor.fromShape(feature_shape) });
    defer matching.deinit(allocator);
    const zeros: [desc.feature_width]f32 = @splat(0);
    var empty = try Buffer.fromBytesOpts(io, session.platform, empty_shape, std.mem.asBytes(&zeros), .{ .wait = true });
    defer empty.deinit();

    const reduced = if (reduced_path != null) try allocator.alloc(u8, image_shape.byteSize()) else null;
    defer if (reduced) |bytes| allocator.free(bytes);
    const reduced_file = if (reduced_path) |path| try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true }) else null;
    errdefer if (reduced_path) |path| std.Io.Dir.cwd().deleteFile(io, path) catch {};
    defer if (reduced_file) |file| file.close(io);
    var staging: [8192]u8 = undefined;
    var reduced_stream = if (reduced_file) |file| file.writerStreaming(io, &staging) else null;
    for (0..repeats) |pass| {
        var previous: ?Buffer = null;
        defer if (previous) |*buffer| buffer.deinit();
        for (0..frames) |frame| {
            if (try input.readPositionalAll(io, image, frame * image.len) != image.len) return error.InvalidKeyframeBuffer;
            var image_buffer = try Buffer.fromBytesOpts(io, session.platform, input_shape, image, .{ .wait = true });
            defer image_buffer.deinit();
            var reduced_value: ?Buffer = null;
            defer if (reduced_value) |*buffer| buffer.deinit();
            if (resize) |*model| {
                try model.run(.{image_buffer});
                reduced_value = model.results.get(Buffer);
                const buffer = &reduced_value.?;
                if (!buffer.shape().eql(image_shape)) return error.InvalidResizedKeyframe;
                try check(&session, buffer);
                if (pass + 1 == repeats) {
                    if (reduced) |bytes| {
                        try buffer.toSlice(io, .init(image_shape, bytes));
                        try reduced_stream.?.interface.writeAll(bytes);
                    }
                }
            }
            const fresh = try extract(&session, &initial, octaves[0..used], &packing, reduced_value orelse image_buffer, empty);
            var old = previous;
            previous = fresh;
            defer if (old) |*buffer| buffer.deinit();
            if (diagnostics) |buffer| {
                const slice = buffer[frame * feature_shape.count() ..][0..feature_shape.count()];
                try fresh.toSlice(io, .init(feature_shape, std.mem.sliceAsBytes(slice)));
            }
            if (old) |left| {
                try matching.run(.{ left, fresh });
                var result = matching.results.get(Buffer);
                defer result.deinit();
                try check(&session, &result);
                const slice = matches[(frame - 1) * match_shape.count() ..][0..match_shape.count()];
                try result.toSlice(io, .init(match_shape, std.mem.sliceAsBytes(slice)));
            }
        }
    }
    if (reduced_stream) |*writer| try writer.interface.flush();
    try rt.probe_io.write(io, args[5], matches);
    if (diagnostics) |buffer| try rt.probe_io.write(io, extras[0], buffer);
}
