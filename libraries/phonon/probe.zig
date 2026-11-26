const std = @import("std");
const runtime = @import("zml_lib/runtime");
const phonon = @import("zml_lib/phonon");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var buffer: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (phonon.schema) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (args.len < 5 or (args.len - 3) % 2 != 0) return error.ExpectedBackendWeightsAndInputOutputPairs;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const allocator = init.gpa;
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var decoder = try phonon.Decoder.load(&session, args[2]);
    defer decoder.deinit();
    const output = try allocator.alloc(f32, phonon.output_count);
    defer allocator.free(output);
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        const input = try runtime.probe_io.read(allocator, init.io, args[i], phonon.input_count);
        defer allocator.free(input);
        const shards = try decoder.step(input, output);
        for (output) |value| if (!std.math.isFinite(value)) return error.NonfiniteOutput;
        runtime.probe_io.placement(&session, shards);
        try runtime.probe_io.write(init.io, args[i + 1], output);
    }
}
