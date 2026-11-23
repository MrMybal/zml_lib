const std = @import("std");
const runtime = @import("zml_lib/runtime");
const t5 = @import("zml_lib/t5_encoder");
pub const std_options: std.Options = .{ .log_level = .info };

fn run(
    comptime profile: t5.Profile,
    comptime storage: runtime.checkpoint.Storage,
    init: std.process.Init,
    args: []const [:0]const u8,
) !void {
    const Model = t5.EncoderWithStorage(profile, storage);
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) {
        var buffer: [4096]u8 = undefined;
        var output = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (Model.schema) |parameter| {
            try output.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try output.interface.print("\t{d}", .{dim});
            try output.interface.writeByte('\n');
        }
        return output.interface.flush();
    }
    if (args.len != 5) return error.ExpectedBackendWeightsTokenCountInputOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const count = try std.fmt.parseInt(usize, args[2], 10);
    if (count < 1 or count > t5.max_tokens) return error.InvalidTokenCount;
    const allocator = init.gpa;
    const input = try runtime.probe_io.read(allocator, init.io, args[3], 2 * count);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, count * @as(usize, @intCast(Model.width)));
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try Model.load(&session, args[1]);
    defer model.deinit();
    for (0..2) |_| runtime.probe_io.placement(&session, try model.encode(count, input, output));
    try runtime.probe_io.write(init.io, args[4], output);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.ExpectedProfile;
    const profile = std.meta.stringToEnum(t5.Profile, args[1]) orelse return error.InvalidProfile;
    const prefix = "--storage=";
    var storage: runtime.checkpoint.Storage = .f32;
    var arguments = args[2..];
    if (arguments.len != 0 and std.mem.startsWith(u8, arguments[0], prefix)) {
        storage = std.meta.stringToEnum(runtime.checkpoint.Storage, arguments[0][prefix.len..]) orelse
            return error.InvalidWeightStorage;
        arguments = arguments[1..];
    }
    switch (profile) {
        inline else => |value| switch (storage) {
            .e4m3 => return error.InvalidWeightStorage,
            inline else => |precision| return run(value, precision, init, arguments),
        },
    }
}
