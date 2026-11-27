const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const flow = @import("zml_lib/trellis2");

pub const std_options: std.Options = .{ .log_level = .info };

fn read(session: *const rt.Session, path: []const u8, shape: zml.Shape) !zml.Buffer {
    const values = try rt.probe_io.read(session.allocator, session.io, path, shape.count());
    defer session.allocator.free(values);
    for (values) |value| if (!std.math.isFinite(value)) return error.NonfiniteInput;
    return zml.Buffer.fromBytesOpts(session.io, session.platform, shape, std.mem.sliceAsBytes(values), .{ .wait = true });
}

fn count(text: []const u8, limit: i64) !i64 {
    const n = try std.fmt.parseInt(i64, text, 10);
    if (n < 1 or n > limit) return error.InvalidTokenCount;
    return n;
}

fn run(comptime p: flow.Profile, init: std.process.Init, args: []const [:0]const u8) !void {
    const Model = flow.Denoiser(p);
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) {
        var storage: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &storage);
        for (Model.schema) |tensor| {
            try writer.interface.writeAll(tensor.name);
            for (tensor.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    const streaming = args.len == 5 and std.mem.eql(u8, args[4], "stream");
    if (!streaming and args.len != 11) return error.ExpectedBackendWeightsSizesInputsOutputAndRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[0]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const rows = try count(args[2], if (p == .structure or p == .structure_multiview) 4096 else 65_536);
    const context_rows = try count(args[3], 8192);
    const alloc = init.gpa;
    const session = try rt.Session.open(alloc, init.io, backend);
    defer session.close();
    const parameters = try rt.checkpoint.loadMixedE4M3(alloc, init.io, session.platform, Model.schema.len, &Model.schema, rt.layers.Parameter.at, args[1]);
    defer parameters.deinit(alloc);

    const shapes: [5]zml.Shape = .{
        .init(.{ rows, p.inputWidth() }, .f32),
        .init(.{ rows, 3 }, .f32),
        .init(.{1}, .f32),
        .init(.{ context_rows, flow.context_width }, .f32),
        .init(.{ rows, @max(1, p.projectionWidth()) }, .f32),
    };
    const out_shape: zml.Shape = .init(.{ rows, p.outputWidth() }, .f32);
    const values = try alloc.alloc(f32, out_shape.count());
    defer alloc.free(values);
    var exe = try rt.Compiled.init(&session, Model.forward, .{
        parameters.tensors,
        zml.Tensor.fromShape(shapes[0]),
        zml.Tensor.fromShape(shapes[1]),
        zml.Tensor.fromShape(shapes[2]),
        zml.Tensor.fromShape(shapes[3]),
        zml.Tensor.fromShape(shapes[4]),
    });
    defer exe.deinit(alloc);
    exe.args.bake(parameters.buffers);
    if (streaming) return stream(&session, &exe, shapes, out_shape, values);

    const repeats = try count(args[10], 64);
    var buffers: [5]zml.Buffer = undefined;
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |*buffer| buffer.deinit();
    for (&buffers, shapes, args[4..9]) |*buffer, shape, input| {
        buffer.* = try read(&session, input, shape);
        loaded += 1;
    }
    for (0..@as(usize, @intCast(repeats))) |_| {
        rt.probe_io.placement(&session, try infer(&session, &exe, buffers, out_shape, values));
    }
    try rt.probe_io.write(init.io, args[9], values);
}

fn infer(session: *const rt.Session, exe: anytype, buffers: [5]zml.Buffer, shape: zml.Shape, values: []f32) !u32 {
    try exe.run(.{ buffers[0], buffers[1], buffers[2], buffers[3], buffers[4] });
    var output = exe.results.get(zml.Buffer);
    defer output.deinit();
    if (!output.shape().eql(shape)) return error.UnexpectedOutputShape;
    const shards = try session.verifyPlacement(&output);
    try output.toSlice(session.io, .init(shape, std.mem.sliceAsBytes(values)));
    for (values) |v| if (!std.math.isFinite(v)) return error.NonfiniteOutput;
    return shards;
}

fn stream(session: *const rt.Session, exe: anytype, shapes: [5]zml.Shape, out_shape: zml.Shape, values: []f32) !void {
    var read_buffer: [8192]u8 = undefined;
    var write_buffer: [8192]u8 = undefined;
    var reader = std.Io.File.stdin().readerStreaming(session.io, &read_buffer);
    var writer = std.Io.File.stdout().writerStreaming(session.io, &write_buffer);
    var max_count: usize = 0;
    for (shapes) |shape| max_count = @max(max_count, shape.count());
    const host = try session.allocator.alloc(f32, max_count);
    defer session.allocator.free(host);

    try writer.interface.writeAll("ready\n");
    try writer.interface.flush();
    while (true) {
        const command = reader.interface.takeByte() catch |err| switch (err) {
            error.EndOfStream => return,
            else => return err,
        };
        if (command != 0) return error.InvalidFlowRequest;
        var buffers: [5]zml.Buffer = undefined;
        var loaded: usize = 0;
        defer for (buffers[0..loaded]) |*buffer| buffer.deinit();
        for (&buffers, shapes) |*buffer, shape| {
            const input = host[0..shape.count()];
            const bytes = std.mem.sliceAsBytes(input);
            try reader.interface.readSliceAll(bytes);
            for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteInput;
            buffer.* = try zml.Buffer.fromBytesOpts(session.io, session.platform, shape, bytes, .{ .wait = true });
            loaded += 1;
        }
        const shards = try infer(session, exe, buffers, out_shape, values);
        try writer.interface.print("backend={s} pjrt={s} device={s} shards={d}\n", .{
            session.target(), session.pjrtPlatform(), session.platform.devices[0].debugString(), shards,
        });
        try writer.interface.writeAll(std.mem.sliceAsBytes(values));
        try writer.interface.flush();
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedProfile;
    const profile = std.meta.stringToEnum(flow.Profile, args[1]) orelse return error.InvalidProfile;
    switch (profile) {
        inline else => |p| return run(p, init, args[2..]),
    }
}
