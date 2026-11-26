const std = @import("std");
const runtime = @import("zml_lib/runtime");
const clip = @import("zml_lib/clip_text");
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var scratch: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (clip.schema) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (args.len < 5 or args.len > 4 + clip.context) return error.ExpectedBackendWeightsOutputTokenIds;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    var tokens: [clip.context]u32 = undefined;
    const ids = tokens[0 .. args.len - 4];
    for (args[4..], ids) |value, *id| id.* = try std.fmt.parseInt(u32, value, 10);
    const output = try allocator.alloc(f32, ids.len * clip.channels);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try clip.Model.load(&session, args[2]);
    defer model.deinit();
    runtime.probe_io.placement(&session, try model.encode(ids, output));
    try runtime.probe_io.write(init.io, args[3], output);
}
