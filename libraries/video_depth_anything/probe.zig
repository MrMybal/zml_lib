const std = @import("std");
const runtime = @import("zml_lib/runtime");
const video = @import("zml_lib/video_depth_anything");
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 6 and std.mem.eql(u8, args[1], "windows"))
        return streamWindows(init, allocator, args[2..]);
    if (args.len != 8) return error.ExpectedBackendWeightsFramesHeightWidthInputOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const t = try std.fmt.parseInt(u32, args[3], 10);
    const h = try std.fmt.parseInt(u32, args[4], 10);
    const w = try std.fmt.parseInt(u32, args[5], 10);
    const size = try video.pixelCount(t, h, w);
    const input = try runtime.probe_io.read(allocator, init.io, args[6], size * 3);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try video.Model.load(&session, args[2]);
    defer model.deinit();
    runtime.probe_io.placement(&session, try model.infer(t, h, w, input, output));
    try runtime.probe_io.write(init.io, args[7], output);
}

fn streamWindows(init: std.process.Init, allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const h = try std.fmt.parseInt(u32, args[2], 10);
    const w = try std.fmt.parseInt(u32, args[3], 10);
    const frames = 32;
    const size = try video.pixelCount(frames, h, w);
    const input = try allocator.alloc(f32, size * 3);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try video.Model.load(&session, args[1]);
    defer model.deinit();

    var input_buffer: [8192]u8 = undefined;
    var output_buffer: [8192]u8 = undefined;
    var source = std.Io.File.stdin().readerStreaming(init.io, &input_buffer);
    var destination = std.Io.File.stdout().writerStreaming(init.io, &output_buffer);
    while (true) {
        _ = source.interface.peekByte() catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        try source.interface.readSliceAll(std.mem.sliceAsBytes(input));
        runtime.probe_io.placement(&session, try model.infer(frames, h, w, input, output));
        try destination.interface.writeAll(std.mem.sliceAsBytes(output));
        try destination.interface.flush();
    }
}
