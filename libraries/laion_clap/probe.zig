const std = @import("std");
const runtime = @import("zml_lib/runtime");
const text = @import("zml_lib/clap_text");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 6 or args.len - 4 > text.context_limit) return error.ExpectedBackendWeightsOutputAndTokenIds;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const len = args.len - 4;
    var ids: [text.context_limit]u32 = undefined;
    var mask: [text.context_limit]u8 = undefined;
    for (args[4..], ids[0..len], mask[0..len]) |arg, *id, *present| {
        id.* = try std.fmt.parseInt(u32, arg, 10);
        present.* = @intFromBool(id.* != 1);
    }
    const session = try runtime.Session.open(init.gpa, init.io, backend);
    defer session.close();
    var model = try text.Model.load(&session, args[2]);
    defer model.deinit();
    var output: [text.embedding_size]f32 = undefined;
    runtime.probe_io.placement(&session, try model.encode(ids[0..len], mask[0..len], &output));
    try runtime.probe_io.write(init.io, args[3], &output);
}
