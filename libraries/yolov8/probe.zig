const std = @import("std");
const runtime = @import("zml_lib/runtime");
const yolo = @import("zml_lib/yolov8");
const zml = @import("zml/core");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const command = try init.minimal.args.toSlice(allocator);
    var args = command[1..];
    if (args.len == 2 and std.mem.eql(u8, args[0], "schema")) {
        const profile = std.meta.stringToEnum(yolo.Profile, args[1]) orelse return error.InvalidProfile;
        var scratch: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (yolo.schema(profile)) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    if (args.len == 3 and std.mem.eql(u8, args[0], "letterbox-shape")) {
        const width = try std.fmt.parseInt(u32, args[1], 10);
        const height = try std.fmt.parseInt(u32, args[2], 10);
        const layout = try yolo.letterbox.layout(height, width);
        var scratch: [64]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        try stdout.interface.print("{d} {d}\n", .{ layout.canvas_width, layout.canvas_height });
        return stdout.interface.flush();
    }
    if (args.len == 0) return error.ExpectedBackendProfileWeightsWidthHeightInputOutputOptionalFramesRepeats;
    const preview = std.mem.eql(u8, args[0], "letterbox");
    const nms_source = std.mem.eql(u8, args[0], "nms-source");
    const bgr_nms_source = std.mem.eql(u8, args[0], "bgr-nms-source");
    const nms_only = nms_source or std.mem.eql(u8, args[0], "nms");
    const bgr_nms = bgr_nms_source or std.mem.eql(u8, args[0], "bgr-nms");
    const source_coordinates = nms_source or bgr_nms_source;
    const bgr = preview or bgr_nms or std.mem.eql(u8, args[0], "bgr");
    const has_weights = !preview and !nms_only;
    if (bgr or nms_only) args = args[1..];
    var raw_path: ?[]const u8 = null;
    if (bgr_nms and args.len >= 2 and std.mem.eql(u8, args[args.len - 2], "--raw-output")) {
        raw_path = args[args.len - 1];
        args = args[0 .. args.len - 2];
    }
    const first_dimension: usize = if (has_weights) 3 else 1;
    const dimensions: usize = if (nms_source) 3 else if (nms_only) 1 else 2;
    const first_path = first_dimension + dimensions;
    if (args.len != first_path + 2 and args.len != first_path + 4)
        return error.ExpectedBackendProfileWeightsWidthHeightInputOutputOptionalFramesRepeats;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const profile = if (has_weights) std.meta.stringToEnum(yolo.Profile, args[1]) orelse return error.InvalidProfile else .coco;
    if (bgr and profile != .coco) return error.BgrRequiresCoco;
    const anchors = if (nms_only) try std.fmt.parseInt(u32, args[first_dimension], 10) else 0;
    if (nms_only and (anchors == 0 or anchors > yolo.nms.max_cells)) return error.InvalidNmsAnchorCount;
    const image_dimensions = first_dimension + @as(usize, @intFromBool(nms_source));
    const has_image_dimensions = !nms_only or nms_source;
    const width = if (has_image_dimensions) try std.fmt.parseInt(u32, args[image_dimensions], 10) else 1;
    const height = if (has_image_dimensions) try std.fmt.parseInt(u32, args[image_dimensions + 1], 10) else 1;
    const files = args[first_path..];
    const frames = if (files.len == 4) try std.fmt.parseInt(usize, files[2], 10) else 1;
    const repeats = if (files.len == 4) try std.fmt.parseInt(u16, files[3], 10) else 1;
    if (frames == 0 or frames > 4096 or repeats == 0) return error.InvalidFrameCount;
    const layout = if (bgr or nms_source) try yolo.letterbox.layout(height, width) else null;
    const canvas_width = if (layout) |value| @as(u32, @intCast(value.canvas_width)) else width;
    const canvas_height = if (layout) |value| @as(u32, @intCast(value.canvas_height)) else height;
    const preview_shape: zml.Shape = .init(.{ 1, 3, canvas_height, canvas_width }, .f32);
    const raw_count = if (nms_only) @as(usize, anchors) * yolo.nms.fields else if (preview) preview_shape.count() else try yolo.outputCount(profile, canvas_width, canvas_height);
    const raw_shape: zml.Shape = .init(.{raw_count}, .f32);
    const nms_shape: zml.Shape = .init(.{ yolo.nms.limit, yolo.nms.width }, .f32);
    const values = if (nms_only or bgr_nms) nms_shape.count() else raw_count;
    const output = try allocator.alloc(f32, values);
    const bytes_per_channel: usize = if (bgr) 1 else @sizeOf(f32);
    const input = try allocator.alloc(u8, if (nms_only) raw_shape.byteSize() else @as(usize, width) * height * 3 * bytes_per_channel);
    const source = try std.Io.Dir.cwd().openFile(init.io, files[0], .{});
    defer source.close(init.io);
    if (try source.length(init.io) != frames * input.len) return error.InvalidBufferSize;
    const destination = try std.Io.Dir.cwd().createFile(init.io, files[1], .{ .exclusive = true });
    errdefer std.Io.Dir.cwd().deleteFile(init.io, files[1]) catch {};
    defer destination.close(init.io);
    var buffer: [8192]u8 = undefined;
    var stream = destination.writerStreaming(init.io, &buffer);
    const raw_file = if (raw_path) |path| try std.Io.Dir.cwd().createFile(init.io, path, .{ .exclusive = true }) else null;
    errdefer if (raw_path) |path| std.Io.Dir.cwd().deleteFile(init.io, path) catch {};
    defer if (raw_file) |file| file.close(init.io);
    const raw_bytes = if (raw_file != null) try allocator.alloc(u8, raw_shape.byteSize()) else null;
    var raw_staging: [8192]u8 = undefined;
    var raw_stream = if (raw_file) |file| file.writerStreaming(init.io, &raw_staging) else null;
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    const preview_input_shape: zml.Shape = .init(.{ 1, height, width, 3 }, .u8);
    var preprocessing = if (preview) try runtime.Compiled.init(&session, yolo.letterbox.forward, .{zml.Tensor.fromShape(preview_input_shape)}) else null;
    defer if (preprocessing) |*graph| graph.deinit(allocator);
    const raw_tensor = zml.Tensor.fromShape(raw_shape);
    var postprocessing = if (source_coordinates)
        try runtime.Compiled.init(&session, yolo.nms.forwardSource, .{ yolo.letterbox.sourceTransform(layout.?, height, width), raw_tensor })
    else if (nms_only or bgr_nms)
        try runtime.Compiled.init(&session, yolo.nms.forward, .{raw_tensor})
    else
        null;
    defer if (postprocessing) |*graph| graph.deinit(allocator);
    var model = if (has_weights) try yolo.Model.load(&session, profile, args[2]) else null;
    defer if (model) |*network| network.deinit();
    for (0..repeats) |pass| {
        for (0..frames) |frame| {
            if (try source.readPositionalAll(init.io, input, frame * input.len) != input.len)
                return error.InvalidBufferSize;
            const last_pass = pass + 1 == repeats;
            if (preprocessing) |*graph| {
                var pixels = try zml.Buffer.fromBytesOpts(init.io, session.platform, preview_input_shape, input, .{ .wait = true });
                defer pixels.deinit();
                try graph.run(.{pixels});
                var result = graph.results.get(zml.Buffer);
                defer result.deinit();
                if (!result.shape().eql(preview_shape)) return error.InvalidLetterboxOutput;
                runtime.probe_io.placement(&session, try session.verifyPlacement(&result));
                if (last_pass) try result.toSlice(init.io, .init(preview_shape, std.mem.sliceAsBytes(output)));
            } else if (postprocessing) |*graph| {
                var raw = if (model) |*network| try network.predictBgr(width, height, input) else try zml.Buffer.fromBytesOpts(init.io, session.platform, raw_shape, input, .{ .wait = true });
                defer raw.deinit();
                if (!raw.shape().eql(raw_shape)) return error.InvalidNmsInput;
                if (bgr_nms) runtime.probe_io.placement(&session, try session.verifyPlacement(&raw));
                try graph.run(.{raw});
                var result = graph.results.get(zml.Buffer);
                defer result.deinit();
                if (!result.shape().eql(nms_shape)) return error.InvalidNmsOutput;
                runtime.probe_io.placement(&session, try session.verifyPlacement(&result));
                if (last_pass) {
                    try result.toSlice(init.io, .init(nms_shape, std.mem.sliceAsBytes(output)));
                    if (output[5] == -2) return error.NonFiniteNmsPredictions;
                    if (raw_bytes) |bytes| {
                        try raw.toSlice(init.io, .init(raw_shape, bytes));
                        try raw_stream.?.interface.writeAll(bytes);
                    }
                }
            } else if (model) |*network| {
                const shards = if (bgr) try network.inferBgr(width, height, input, output) else try network.inferFp32Bytes(width, height, input, output);
                runtime.probe_io.placement(&session, shards);
            }
            if (last_pass) try stream.interface.writeAll(std.mem.sliceAsBytes(output));
        }
    }
    try stream.interface.flush();
    if (raw_stream) |*writer| try writer.interface.flush();
}
