const std = @import("std");
const builtin = @import("builtin");
const rt = @import("zml_lib/runtime");
const zml = @import("zml/core");
const crop = @import("pose_crop.zig");
const scale = @import("pose_scale.zig");

pub const std_options: std.Options = .{ .log_level = .info };

extern "c" fn _putenv_s(name: [*:0]const u8, value: [*:0]const u8) c_int;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn keepAffinePrecision(init: std.process.Init) !void {
    const key = "IREE_PJRT_IREE_COMPILER_OPTIONS";
    const flag = "--iree-input-demote-f64-to-f32";
    const required = flag ++ "=false";
    const existing = init.environ_map.get(key) orelse "";
    if (std.mem.indexOf(u8, existing, required) != null) return;
    if (std.mem.indexOf(u8, existing, flag) != null) {
        std.log.err("Pose cropping requires {s}", .{required});
        return error.IncompatibleIreePrecision;
    }
    // This executable's PJRT compiler must see FP64 arithmetic for OpenCV's
    // fixed-point grid. Other probes keep their own compiler settings.
    const options = try init.arena.allocator().printSentinel("{s} {s}", .{ existing, required }, 0);
    const status = if (builtin.os.tag == .windows) _putenv_s(key, options) else setenv(key, options, 1);
    if (status != 0) return error.CannotConfigureIreePrecision;
}

pub fn main(init: std.process.Init) !void {
    const command = try init.minimal.args.toSlice(init.arena.allocator());
    const halve = command.len > 1 and std.mem.eql(u8, command[1], "half");
    const argv = command[1 + @as(usize, @intFromBool(halve)) ..];
    const reference_output: ?[]const u8 = if (halve and argv.len == 10) argv[9] else null;
    if (argv.len != 9 and reference_output == null) return error.ExpectedBackendHeightWidthFramesBatchImagesBoxesOutputRepeats;
    const mode: crop.Mode = if (reference_output != null) .half_check else if (halve) .half else .crop;
    const backend = std.meta.stringToEnum(rt.Backend, argv[0]) orelse return error.InvalidBackend;
    const height = try std.fmt.parseInt(usize, argv[1], 10);
    const width = try std.fmt.parseInt(usize, argv[2], 10);
    const frames = try std.fmt.parseInt(usize, argv[3], 10);
    const batch = try std.fmt.parseInt(usize, argv[4], 10);
    const repeats = try std.fmt.parseInt(u16, argv[8], 10);
    const side_limit: usize = if (halve) 8192 else 4096;
    if (height == 0 or height > side_limit or width == 0 or width > side_limit or
        frames == 0 or frames > 4096 or batch == 0 or batch > 8 or repeats == 0)
        return error.InvalidCropDimensions;

    const allocator = std.heap.page_allocator;
    const io = init.io;
    const frame_size = height * width * 3;
    const source = try std.Io.Dir.cwd().openFile(io, argv[5], .{});
    defer source.close(io);
    if (try source.length(io) != frames * frame_size) return error.InvalidBufferSize;
    const boxes = try rt.probe_io.read(allocator, io, argv[6], frames * 3);
    defer allocator.free(boxes);
    var blurred: usize = 0;
    for (0..frames) |frame| {
        const x, const y, const size = boxes[3 * frame ..][0..3].*;
        if (!std.math.isFinite(x) or !std.math.isFinite(y) or !std.math.isFinite(size) or
            @abs(x) > 1e6 or @abs(y) > 1e6 or size <= 0 or size > 1e6)
            return error.InvalidCropBox;
        if (size / 1024 > 1.1) blurred += 1;
    }
    std.log.info("pose_crop_frames={d} antialiased={d}", .{ frames, blurred });

    const input_shape: zml.Shape = .init(.{ batch, height, width, 3 }, .u8);
    const box_shape: zml.Shape = .init(.{ batch, 3 }, .f32);
    const shape: zml.Shape = .init(.{ batch, 3, rt.pose_vit.height, rt.pose_vit.width }, .f32);
    const reduced_shape: zml.Shape = .init(.{ batch, scale.halfSize(@intCast(height)), scale.halfSize(@intCast(width)), 3 }, .u8);
    const images = try allocator.alloc(u8, input_shape.count());
    defer allocator.free(images);
    const batch_boxes = try allocator.alloc(f32, box_shape.count());
    defer allocator.free(batch_boxes);
    const values = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(values);
    const reduced = if (reference_output != null) try allocator.alloc(u8, reduced_shape.byteSize()) else null;
    defer if (reduced) |bytes| allocator.free(bytes);
    if (backend != .cpu) try keepAffinePrecision(init);
    const session = try rt.Session.open(allocator, io, backend);
    defer session.close();
    const signatures = .{ zml.Tensor.fromShape(input_shape), zml.Tensor.fromShape(box_shape) };
    var model = switch (mode) {
        inline else => |kind| if (blurred != 0)
            try rt.Compiled.init(&session, crop.Pipeline(kind, true).forward, signatures)
        else
            try rt.Compiled.init(&session, crop.Pipeline(kind, false).forward, signatures),
    };
    defer model.deinit(allocator);
    const destination = try std.Io.Dir.cwd().createFile(io, argv[7], .{ .exclusive = true });
    errdefer std.Io.Dir.cwd().deleteFile(io, argv[7]) catch {};
    defer destination.close(io);
    const reduced_file = if (reference_output) |path|
        try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true })
    else
        null;
    errdefer if (reference_output) |path| std.Io.Dir.cwd().deleteFile(io, path) catch {};
    defer if (reduced_file) |file| file.close(io);
    var staging: [8192]u8 = undefined;
    var stream = destination.writerStreaming(io, &staging);
    var reduced_staging: [8192]u8 = undefined;
    var reduced_stream = if (reduced_file) |file| file.writerStreaming(io, &reduced_staging) else null;
    for (0..repeats) |pass| {
        var first: usize = 0;
        while (first < frames) : (first += batch) {
            const used = @min(batch, frames - first);
            const loaded = images[0 .. used * frame_size];
            if (try source.readPositionalAll(io, loaded, first * frame_size) != loaded.len) return error.InvalidBufferSize;
            @memset(images[loaded.len..], 0);
            @memcpy(batch_boxes[0 .. used * 3], boxes[first * 3 ..][0 .. used * 3]);
            @memset(batch_boxes[used * 3 ..], 0);

            var rgb = try zml.Buffer.fromBytesOpts(io, session.platform, input_shape, images, .{ .wait = true });
            defer rgb.deinit();
            var coordinates = try zml.Buffer.fromBytesOpts(io, session.platform, box_shape, std.mem.sliceAsBytes(batch_boxes), .{ .wait = true });
            defer coordinates.deinit();
            try model.run(.{ rgb, coordinates });
            var reduced_value: ?zml.Buffer = null;
            var value = if (mode == .half_check) blk: {
                const crops, const half_images = model.results.get(struct { zml.Buffer, zml.Buffer });
                reduced_value = half_images;
                break :blk crops;
            } else model.results.get(zml.Buffer);
            defer value.deinit();
            defer if (reduced_value) |*buffer| buffer.deinit();
            if (!value.shape().eql(shape)) return error.UnexpectedCropOutput;
            rt.probe_io.placement(&session, try session.verifyPlacement(&value));
            if (reduced_value) |*buffer| {
                if (!buffer.shape().eql(reduced_shape)) return error.UnexpectedScaleOutput;
                rt.probe_io.placement(&session, try session.verifyPlacement(buffer));
                if (pass + 1 == repeats) {
                    const bytes = reduced.?;
                    try buffer.toSlice(io, .init(reduced_shape, bytes));
                    try reduced_stream.?.interface.writeAll(bytes[0 .. used * (reduced_shape.byteSize() / batch)]);
                }
            }
            if (pass + 1 == repeats) {
                try value.toSlice(io, .init(shape, values));
                try stream.interface.writeAll(values[0 .. used * (shape.byteSize() / batch)]);
            }
        }
    }
    try stream.interface.flush();
    if (reduced_stream) |*writer| try writer.interface.flush();
}
