const std = @import("std");
const runtime = @import("session.zig");
const zml = @import("zml/core");

pub fn read(allocator: std.mem.Allocator, io: std.Io, path: []const u8, count: usize) ![]f32 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    if (try file.length(io) != try std.math.mul(usize, count, @sizeOf(f32))) return error.InvalidBufferSize;
    const values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);
    const bytes = std.mem.sliceAsBytes(values);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.InvalidBufferSize;
    return values;
}

pub fn write(io: std.Io, path: []const u8, values: []const f32) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(values), .flags = .{ .exclusive = true } });
}

pub fn placement(session: *const runtime.Session, shards: u32) void {
    std.log.info("backend={s} pjrt={s} device={s} shards={d}", .{
        session.target(), session.pjrtPlatform(), session.platform.devices[0].debugString(), shards,
    });
}

pub fn poseProbe(comptime graph: type, init: std.process.Init, args: []const [:0]const u8) !void {
    const has_weights = graph.schema.len != 0;
    if (has_weights and args.len == 1 and std.mem.eql(u8, args[0], "schema")) {
        var scratch: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (graph.schema) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    const offset = 1 + @as(usize, @intFromBool(has_weights));
    if (args.len != offset + 5) return error.ExpectedBackendOptionalWeightsFramesBatchInputOutputRepeats;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const frames = try std.fmt.parseInt(usize, args[offset], 10);
    const batch = try std.fmt.parseInt(usize, args[offset + 1], 10);
    const repeats = try std.fmt.parseInt(u16, args[offset + 4], 10);
    if (frames == 0 or frames > 4096 or batch == 0 or batch > 16 or repeats == 0)
        return error.InvalidPoseRequest;

    const in_shape: zml.Shape = .init(.{batch} ++ graph.input_dims, .f32);
    const out_shape: zml.Shape = .init(.{batch} ++ graph.output_dims, .f32);
    const frame_in = in_shape.count() / batch;
    const frame_out = out_shape.count() / batch;
    const file = try std.Io.Dir.cwd().openFile(init.io, args[offset + 2], .{});
    defer file.close(init.io);
    if (try file.length(init.io) != frames * frame_in * @sizeOf(f32)) return error.InvalidBufferSize;

    const allocator = std.heap.page_allocator;
    const input = try allocator.alloc(f32, in_shape.count());
    defer allocator.free(input);
    const output = try allocator.alloc(f32, out_shape.count());
    defer allocator.free(output);
    const all = try allocator.alloc(f32, frames * frame_out);
    defer allocator.free(all);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = if (has_weights) try runtime.TensorModel.load(&session, &graph.schema, args[1]) else runtime.TensorModel{
        .session = &session,
        .parameters = .{ .tensors = &.{}, .buffers = &.{} },
    };
    defer model.deinit();

    for (0..repeats) |_| {
        var first: usize = 0;
        while (first < frames) : (first += batch) {
            const used = @min(batch, frames - first);
            const bytes = std.mem.sliceAsBytes(input[0 .. used * frame_in]);
            if (try file.readPositionalAll(init.io, bytes, first * frame_in * @sizeOf(f32)) != bytes.len)
                return error.InvalidBufferSize;
            @memset(input[used * frame_in ..], 0);
            placement(&session, try model.infer(graph.forward, in_shape, out_shape, input, output));
            @memcpy(all[first * frame_out ..][0 .. used * frame_out], output[0 .. used * frame_out]);
        }
    }
    try write(init.io, args[offset + 3], all);
}

pub fn restorationProbe(comptime model: type, init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) return error.ExpectedBackendWeightsHeightWidthInputOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const h = try std.fmt.parseInt(u32, args[3], 10);
    const w = try std.fmt.parseInt(u32, args[4], 10);
    const count = try std.math.mul(usize, 3, try std.math.mul(usize, h, w));
    const out_count = try std.math.mul(usize, count, model.scale * model.scale);
    const input = try read(allocator, init.io, args[5], count);
    defer allocator.free(input);
    const output = try allocator.alloc(f32, out_count);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var instance = try model.Model.load(&session, args[2]);
    defer instance.deinit();
    for (0..2) |_| placement(&session, try instance.infer(h, w, input, output));
    try write(init.io, args[6], output);
}

pub fn segmentationProbe(comptime model: type, init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 7) return error.ExpectedBackendEncodeOrDecodeWeightsPointCountInputOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    if (std.mem.eql(u8, args[2], "encode")) {
        const input = try read(allocator, init.io, args[5], 3 * model.image_size * model.image_size);
        const output = try allocator.alloc(f32, model.embedding_count);
        var encoder = try model.Encoder.load(&session, args[3]);
        defer encoder.deinit();
        const shards = try encoder.encode(input, output);
        try write(init.io, args[6], output);
        placement(&session, shards);
    } else if (std.mem.eql(u8, args[2], "decode")) {
        const points = try std.fmt.parseInt(usize, args[4], 10);
        const input = try read(allocator, init.io, args[5], try model.predictInputSize(points));
        const output = try allocator.alloc(f32, model.output_count);
        var decoder = try model.Decoder.load(&session, args[3]);
        defer decoder.deinit();
        const shards = try decoder.predict(points, input, output);
        try write(init.io, args[6], output);
        placement(&session, shards);
    } else return error.InvalidOperation;
}
