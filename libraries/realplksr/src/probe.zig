const std = @import("std");
const runtime = @import("zml_lib/runtime");
const realplksr = @import("zml_lib/realplksr");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 7) {
        std.log.err("usage: realplksr_probe BACKEND WEIGHTS WIDTH HEIGHT INPUT_F32 OUTPUT_F32", .{});
        return error.InvalidArguments;
    }
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const width = try std.fmt.parseInt(u32, args[3], 10);
    const height = try std.fmt.parseInt(u32, args[4], 10);
    const session = try runtime.Session.open(allocator, io, backend);
    defer session.close();
    var model = try realplksr.Model.load(&session, args[2], width, height);
    defer model.deinit();
    const input = try allocator.alloc(f32, model.input_shape.count());
    const output = try allocator.alloc(f32, model.output_shape.count());
    const cwd = std.Io.Dir.cwd();
    const source = try cwd.openFile(io, args[5], .{});
    defer source.close(io);
    const input_bytes = std.mem.sliceAsBytes(input);
    if (try source.length(io) != input_bytes.len or try source.readPositionalAll(io, input_bytes, 0) != input_bytes.len)
        return error.InvalidBufferSize;
    const shards = try model.infer(input, output);
    const second = try allocator.alloc(f32, output.len);
    _ = try model.infer(input, second);
    for (output, second) |first, repeated| {
        if (!std.math.isFinite(first) or !std.math.isFinite(repeated) or
            !std.math.approxEqAbs(f32, first, repeated, 1e-5))
            return error.InconsistentOutput;
    }
    try cwd.writeFile(io, .{ .sub_path = args[6], .data = std.mem.sliceAsBytes(output), .flags = .{ .exclusive = true } });
    std.log.info("backend={s} pjrt={s} shards={d}", .{ session.target(), session.pjrtPlatform(), shards });
}
