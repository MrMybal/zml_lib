const std = @import("std");
const rt = @import("zml_lib/runtime");
const trellis = @import("zml_lib/trellis");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedOperationAndProfile;
    if (std.mem.eql(u8, args[2], "official")) return run(trellis.official, init, args[1], args[3..]);
    if (std.mem.eql(u8, args[2], "fixture")) return run(trellis.fixture, init, args[1], args[3..]);
    return error.InvalidProfile;
}

fn run(comptime widths: [3]i64, init: std.process.Init, operation: []const u8, args: []const [:0]const u8) !void {
    if (std.mem.eql(u8, operation, "schema")) {
        if (args.len != 0) return error.UnexpectedArguments;
        var buffer: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (comptime trellis.schema(widths)) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    if (!std.mem.eql(u8, operation, "decode") or args.len != 8)
        return error.ExpectedBackendWeightsDepthHeightWidthInputOutputRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[0]) orelse return error.InvalidBackend;
    const size = try trellis.Volume.init(
        try std.fmt.parseInt(i64, args[2], 10),
        try std.fmt.parseInt(i64, args[3], 10),
        try std.fmt.parseInt(i64, args[4], 10),
    );
    const repeats = try std.fmt.parseInt(u16, args[7], 10);
    if (repeats == 0) return error.InvalidRepeatCount;
    const allocator = std.heap.page_allocator;
    const input = try rt.probe_io.read(allocator, init.io, args[5], size.latentCount());
    defer allocator.free(input);
    const output = try allocator.alloc(f32, size.fieldCount());
    defer allocator.free(output);
    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    var decoder = try trellis.Decoder(widths).load(&session, args[1]);
    defer decoder.deinit();
    for (0..repeats) |_| rt.probe_io.placement(&session, try decoder.decode(size, input, output));
    try rt.probe_io.write(init.io, args[6], output);
}
