const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const da3 = @import("zml_lib/depth_anything_v3");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var scratch: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (da3.parameters) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    if (args.len != 8) return error.ExpectedBackendWeightsHeightWidthInputOutputAndRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const height = try std.fmt.parseInt(i64, args[3], 10);
    const width = try std.fmt.parseInt(i64, args[4], 10);
    const repeats = try std.fmt.parseInt(u16, args[7], 10);
    if (height < 14 or height > 518 or width < 14 or width > 518 or @mod(height, 14) != 0 or @mod(width, 14) != 0 or repeats == 0)
        return error.InvalidImageDimensions;

    const input_shape: zml.Shape = .init(.{ 4, 3, height, width }, .f32);
    const output_shape: zml.Shape = .init(.{da3.elements(4, height, width)}, .f32);
    const input = try rt.probe_io.read(init.gpa, init.io, args[5], input_shape.count());
    defer init.gpa.free(input);
    for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteInput;
    const output = try init.gpa.alloc(f32, output_shape.count());
    defer init.gpa.free(output);

    const session = try rt.Session.open(init.gpa, init.io, backend);
    defer session.close();
    var model = try rt.TensorModel.load(&session, &da3.parameters, args[2]);
    defer model.deinit();
    for (0..repeats) |_|
        rt.probe_io.placement(&session, try model.infer(da3.forward, input_shape, output_shape, input, output));
    for (output) |value| if (!std.math.isFinite(value)) return error.NonfiniteOutput;
    try rt.probe_io.write(init.io, args[6], output);
}
