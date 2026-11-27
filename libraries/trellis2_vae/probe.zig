const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const vae = @import("zml_lib/trellis2_vae");

pub const std_options: std.Options = .{ .log_level = .info };

fn run(comptime graph: type, init: std.process.Init, args: []const [:0]const u8) !void {
    if (std.mem.eql(u8, args[1], "schema") and args.len == 5) {
        var storage: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &storage);
        for (graph.parameters) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (args.len != 10 or !std.mem.eql(u8, args[1], "infer")) return error.ExpectedVaeStageRequest;
    const backend = std.meta.stringToEnum(rt.Backend, args[5]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const rows = try std.fmt.parseInt(i64, args[7], 10);
    // The input protocol carries integer neighbor indices as exact FP32 values.
    if (rows < 1 or rows > 1 << 24) return error.InvalidSparseSize;
    const input_shape: zml.Shape = .init(.{ rows, graph.input_width + 27 }, .f32);
    const output_shape: zml.Shape = .init(.{ rows, graph.output_width }, .f32);
    const input = try rt.probe_io.read(init.gpa, init.io, args[8], input_shape.count());
    defer init.gpa.free(input);
    const stride: usize = @intCast(graph.input_width + 27);
    for (input, 0..) |value, offset| {
        if (!std.math.isFinite(value)) return error.NonfiniteInput;
        if (offset % stride >= graph.input_width and (value != @floor(value) or value < 0 or value > @as(f32, @floatFromInt(rows))))
            return error.InvalidNeighborIndex;
    }
    const output = try init.gpa.alloc(f32, output_shape.count());
    defer init.gpa.free(output);

    const session = try rt.Session.open(init.gpa, init.io, backend);
    defer session.close();
    const weights = try rt.checkpoint.loadMixedE4M3(init.gpa, init.io, session.platform, graph.parameters.len, &graph.parameters, rt.layers.Parameter.at, args[6]);
    defer weights.deinit(init.gpa);
    var executable = try rt.Compiled.init(&session, graph.forward, .{ weights.tensors, zml.Tensor.fromShape(input_shape) });
    defer executable.deinit(init.gpa);
    executable.args.bake(weights.buffers);
    var device_input = try zml.Buffer.fromBytesOpts(init.io, session.platform, input_shape, std.mem.sliceAsBytes(input), .{ .wait = true });
    defer device_input.deinit();
    try executable.run(.{device_input});
    var result = executable.results.get(zml.Buffer);
    defer result.deinit();
    if (!result.shape().eql(output_shape)) return error.UnexpectedOutputShape;
    const shards = try session.verifyPlacement(&result);
    try result.toSlice(init.io, .init(output_shape, std.mem.sliceAsBytes(output)));
    for (output) |value| if (!std.math.isFinite(value)) return error.NonfiniteOutput;
    try rt.probe_io.write(init.io, args[9], output);
    rt.probe_io.placement(&session, shards);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) return error.ExpectedProfileLevelAndPhase;
    const profile = std.meta.stringToEnum(vae.Profile, args[2]) orelse return error.InvalidProfile;
    const level = try std.fmt.parseInt(u8, args[3], 10);
    const post = std.mem.eql(u8, args[4], "post");
    if (!post and !std.mem.eql(u8, args[4], "pre")) return error.InvalidPhase;
    if (level > 4 or (level == 4 and post)) return error.InvalidLevel;
    switch (profile) {
        inline else => |p| {
            switch (level) {
                inline 0...3 => |i| {
                    if (post) return run(vae.Stage(p, i, true), init, args);
                    return run(vae.Stage(p, i, false), init, args);
                },
                4 => return run(vae.Stage(p, 4, false), init, args),
                else => unreachable,
            }
        },
    }
}
