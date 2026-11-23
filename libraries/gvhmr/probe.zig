const std = @import("std");
const rt = @import("zml_lib/runtime");
const gv = @import("zml_lib/gvhmr");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var buffer: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (gv.schema) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    if (args.len != 7) return error.ExpectedBackendWeightsFramesInputOutputRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    const frames = try std.fmt.parseInt(i64, args[3], 10);
    if (frames < 1 or frames > gv.frames_max) return error.InvalidFrameCount;
    const count: usize = @intCast(frames);
    const repeats = try std.fmt.parseInt(u16, args[6], 10);
    if (repeats == 0) return error.InvalidRepeatCount;
    const allocator = std.heap.page_allocator;
    const input = try rt.probe_io.read(allocator, init.io, args[4], count * gv.input_width);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, count * gv.output_width);
    defer allocator.free(output);
    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try gv.Model.load(&session, args[2]);
    defer model.deinit();
    for (0..repeats) |_| rt.probe_io.placement(&session, try model.predict(frames, input, output));
    try rt.probe_io.write(init.io, args[5], output);
}
