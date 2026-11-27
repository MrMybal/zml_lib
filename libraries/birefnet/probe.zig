const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const birefnet = @import("zml_lib/birefnet");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var scratch: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (birefnet.parameters) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    const raw = args.len == 9 and std.mem.eql(u8, args[8], "--logits");
    if (args.len != 8 and !raw) return error.ExpectedBackendWeightsDimsInputOutputAndRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const height = try std.fmt.parseInt(i64, args[3], 10);
    const width = try std.fmt.parseInt(i64, args[4], 10);
    const repeats = try std.fmt.parseInt(u16, args[7], 10);
    if (height < 64 or height > 1024 or width < 64 or width > 1024 or @mod(height, 64) != 0 or @mod(width, 64) != 0 or repeats == 0)
        return error.InvalidImageDimensions;
    const input_shape: zml.Shape = .init(.{ 1, 3, height, width }, .f32);
    const output_shape: zml.Shape = .init(.{ 1, 1, height, width }, .f32);
    const input = try rt.probe_io.read(init.gpa, init.io, args[5], input_shape.count());
    defer init.gpa.free(input);
    for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteInput;
    const values = try init.gpa.alloc(f32, output_shape.count());
    defer init.gpa.free(values);
    const session = try rt.Session.open(init.gpa, init.io, backend);
    defer session.close();
    var model = try rt.TensorModel.loadCheckpoint(&session, &birefnet.parameters, args[2], .f16);
    defer model.deinit();
    for (0..repeats) |_| {
        const shards = if (raw)
            try model.infer(birefnet.logits, input_shape, output_shape, input, values)
        else
            try model.infer(birefnet.forward, input_shape, output_shape, input, values);
        rt.probe_io.placement(&session, shards);
    }
    for (values) |value| if (!std.math.isFinite(value)) return error.NonfiniteOutput;
    try rt.probe_io.write(init.io, args[6], values);
}
