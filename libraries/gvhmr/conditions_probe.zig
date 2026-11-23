const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const conditions = @import("conditions.zig");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) return error.ExpectedBackendFramesInputStatsOutputRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    const frames = try std.fmt.parseInt(usize, args[2], 10);
    const repeats = try std.fmt.parseInt(u16, args[6], 10);
    if (frames == 0 or frames > 4096 or repeats == 0) return error.InvalidConditionsDimensions;

    const allocator = std.heap.page_allocator;
    const io = init.io;
    const values = try rt.probe_io.read(allocator, io, args[3], frames * conditions.input_width);
    defer allocator.free(values);
    const stats = try rt.probe_io.read(allocator, io, args[4], 12);
    defer allocator.free(stats);
    for (values) |value| if (!std.math.isFinite(value)) return error.NonFiniteConditions;
    for (stats, 0..) |value, i| if (!std.math.isFinite(value) or (i >= 6 and value <= 0)) return error.InvalidCameraStatistics;
    for (0..frames) |i| {
        const row = values[i * conditions.input_width ..][0..conditions.input_width];
        if (row[conditions.box_start + 2] <= 0 or row[conditions.camera_start] <= 0) return error.InvalidCameraCondition;
    }

    const input_shape: zml.Shape = .init(.{ frames, conditions.input_width }, .f32);
    const stats_shape: zml.Shape = .init(.{12}, .f32);
    const output_shape: zml.Shape = .init(.{ frames, conditions.output_width }, .f32);
    const output = try allocator.alloc(f32, output_shape.count());
    defer allocator.free(output);

    const session = try rt.Session.open(allocator, io, backend);
    defer session.close();
    var model = try rt.Compiled.init(&session, conditions.forward, .{ zml.Tensor.fromShape(input_shape), zml.Tensor.fromShape(stats_shape) });
    defer model.deinit(allocator);
    var input = try zml.Buffer.fromBytesOpts(io, session.platform, input_shape, std.mem.sliceAsBytes(values), .{ .wait = true });
    defer input.deinit();
    var normalization = try zml.Buffer.fromBytesOpts(io, session.platform, stats_shape, std.mem.sliceAsBytes(stats), .{ .wait = true });
    defer normalization.deinit();
    for (0..repeats) |pass| {
        try model.run(.{ input, normalization });
        var result = model.results.get(zml.Buffer);
        defer result.deinit();
        if (!result.shape().eql(output_shape)) return error.InvalidConditionsOutput;
        rt.probe_io.placement(&session, try session.verifyPlacement(&result));
        if (pass + 1 == repeats) try result.toSlice(io, .init(output_shape, std.mem.sliceAsBytes(output)));
    }
    for (output) |value| if (!std.math.isFinite(value)) return error.NonFiniteConditionsOutput;
    try rt.probe_io.write(io, args[5], output);
}
