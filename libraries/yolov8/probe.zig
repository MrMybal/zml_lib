const std = @import("std");
const runtime = @import("zml_lib/runtime");
const yolo = @import("zml_lib/yolov8");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 8) return error.ExpectedBackendProfileWeightsWidthHeightInputOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const profile = std.meta.stringToEnum(yolo.Profile, args[2]) orelse return error.InvalidProfile;
    const width = try std.fmt.parseInt(u32, args[4], 10);
    const height = try std.fmt.parseInt(u32, args[5], 10);
    const output = try allocator.alloc(f32, try yolo.outputCount(profile, width, height));
    const input = try runtime.probe_io.read(allocator, init.io, args[6], @as(usize, width) * height * 3);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try yolo.Model.load(&session, profile, args[3]);
    defer model.deinit();
    const shards = try model.infer(width, height, input, output);
    try runtime.probe_io.write(init.io, args[7], output);
    runtime.probe_io.placement(&session, shards);
}
