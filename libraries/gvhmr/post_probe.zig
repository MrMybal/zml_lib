const std = @import("std");
const rt = @import("zml_lib/runtime");
const post = @import("zml_lib/gvhmr/postprocess");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 8) return error.ExpectedBackendModeWeightsFramesInputOutputRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    const mode = std.meta.stringToEnum(post.Mode, args[2]) orelse return error.InvalidCameraMode;
    const frames = try std.fmt.parseInt(i64, args[4], 10);
    if (frames < 1 or frames > post.frames_max) return error.InvalidFrameCount;
    const count: usize = @intCast(frames);
    const repeats = try std.fmt.parseInt(u16, args[7], 10);
    if (repeats == 0) return error.InvalidRepeatCount;

    const allocator = std.heap.page_allocator;
    const input = try rt.probe_io.read(allocator, init.io, args[5], count * post.input_width);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, count * post.output_width);
    defer allocator.free(output);
    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try post.Model.load(&session, args[3], mode);
    defer model.deinit();
    for (0..repeats) |_| rt.probe_io.placement(&session, try model.predict(frames, input, output));
    try rt.probe_io.write(init.io, args[6], output);
}
