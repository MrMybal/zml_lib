const std = @import("std");
const runtime = @import("zml_lib/runtime");
const video = @import("zml_lib/video_depth_anything");
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
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
