const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const dino = @import("zml_lib/dinov2");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedOperationAndProfile;
    if (std.mem.eql(u8, args[2], "registered")) return launch(dino.registered, init, args);
    if (std.mem.eql(u8, args[2], "fixture")) return launch(dino.fixture, init, args);
    return error.InvalidProfile;
}

fn launch(comptime p: dino.Architecture, init: std.process.Init, args: []const [:0]const u8) !void {
    const graph = dino.Conditioner(p);
    if (args.len == 3 and std.mem.eql(u8, args[1], "schema")) {
        var scratch: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (graph.parameters) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    if (args.len != 10 or !std.mem.eql(u8, args[1], "encode")) return error.InvalidEncoderRequest;
    const backend = std.meta.stringToEnum(rt.Backend, args[3]) orelse return error.InvalidBackend;
    const h = try std.fmt.parseInt(i64, args[5], 10);
    const w = try std.fmt.parseInt(i64, args[6], 10);
    const patch = dino.patch_size;
    if (h < patch or w < patch or h > 518 or w > 518 or @mod(h, patch) != 0 or @mod(w, patch) != 0) return error.InvalidImageDimensions;
    const repeats = try std.fmt.parseInt(u16, args[9], 10);
    if (repeats == 0) return error.InvalidRepeatCount;
    const shape: zml.Shape = .init(.{ 1, 3, h, w }, .f32);
    const output_shape: zml.Shape = .init(.{ 1, 1 + p.registers + @divExact(h, patch) * @divExact(w, patch), p.channels }, .f32);
    const allocator = init.arena.allocator();
    const image = try rt.probe_io.read(allocator, init.io, args[7], shape.count());
    const output = try allocator.alloc(f32, output_shape.count());
    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    var encoder = try rt.TensorModel.load(&session, &graph.parameters, args[4]);
    defer encoder.deinit();
    for (0..repeats) |_| rt.probe_io.placement(&session, try encoder.infer(graph.forward, shape, output_shape, image, output));
    try rt.probe_io.write(init.io, args[8], output);
}
