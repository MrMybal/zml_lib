const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const heads = @import("zml_lib/trellis/heads");
const raw = rt.probe_io;
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) return error.ExpectedOperationProfileAndRepresentation;
    const kind = std.meta.stringToEnum(heads.Kind, args[3]) orelse return error.InvalidRepresentation;
    if (std.mem.eql(u8, args[2], "official")) return selectProfile(heads.official, kind, init, args);
    if (std.mem.eql(u8, args[2], "fixture")) return selectProfile(heads.fixture, kind, init, args);
    return error.InvalidProfile;
}

fn selectProfile(comptime p: heads.Profile, kind: heads.Kind, init: std.process.Init, args: []const [:0]const u8) !void {
    return switch (kind) {
        inline else => |comptime_kind| launch(p, comptime_kind, init, args),
    };
}

fn upload(session: *const rt.Session, shape: zml.Shape, path: []const u8) !zml.Buffer {
    const data = try raw.read(session.allocator, session.io, path, shape.count());
    defer session.allocator.free(data);
    return zml.Buffer.fromBytesOpts(session.io, session.platform, shape, std.mem.sliceAsBytes(data), .{ .wait = true });
}

fn launch(comptime p: heads.Profile, comptime kind: heads.Kind, init: std.process.Init, args: []const [:0]const u8) !void {
    const graph = heads.Decoder(p, kind);
    if (args.len == 4 and std.mem.eql(u8, args[1], "schema")) {
        var scratch: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (graph.parameters) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    if (args.len != 21 or !std.mem.eql(u8, args[1], "decode")) return error.InvalidDecoderRequest;
    const backend = std.meta.stringToEnum(rt.Backend, args[4]) orelse return error.InvalidBackend;
    const rows = try std.fmt.parseInt(i64, args[6], 10);
    const repeats = try std.fmt.parseInt(u16, args[20], 10);
    if (rows < 1 or rows > p.resolution * p.resolution * p.resolution or repeats == 0) return error.InvalidInputDimensions;

    const feat_shape: zml.Shape = .init(.{ rows, 8 }, .f32);
    const coords_shape: zml.Shape = .init(.{ rows, 3 }, .f32);
    const restore_shape: zml.Shape = .init(.{rows}, .f32);
    const out_shape: zml.Shape = .init(.{ if (kind == .mesh) 64 * rows else rows, heads.outputWidth(kind) }, .f32);
    const neighbor_shapes: [2]zml.Shape = if (kind == .mesh)
        .{ .init(.{ 8 * rows, 27 }, .f32), .init(.{ 64 * rows, 27 }, .f32) }
    else
        @splat(.init(.{ 1, 1 }, .f32));
    var window_shapes: [2]zml.Shape = undefined;
    for (&window_shapes, 0..) |*shape, i| {
        const count = try std.fmt.parseInt(i64, args[7 + i * 2], 10);
        const width = try std.fmt.parseInt(i64, args[8 + i * 2], 10);
        if (count < 1 or count > @min(rows, 729) or width < 1 or width > @min(rows, 512) or count * width < rows)
            return error.InvalidAttentionWindows;
        shape.* = .init(.{ count, width }, .f32);
    }
    const allocator = std.heap.page_allocator;
    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    var features = try upload(&session, feat_shape, args[11]);
    defer features.deinit();
    var coords = try upload(&session, coords_shape, args[12]);
    defer coords.deinit();
    var windows = try uploadPair(&session, window_shapes, args[13..15]);
    defer for (&windows) |*buffer| buffer.deinit();
    var inverse = try uploadPair(&session, @splat(restore_shape), args[15..17]);
    defer for (&inverse) |*buffer| buffer.deinit();
    var neighbors = try uploadPair(&session, neighbor_shapes, args[17..19]);
    defer for (&neighbors) |*buffer| buffer.deinit();
    const weights = try rt.checkpoint.load(allocator, session.io, session.platform, graph.parameters.len, @as([]const rt.layers.Parameter, &graph.parameters), rt.layers.Parameter.at, args[5]);
    defer weights.deinit(allocator);
    var compiled = try rt.Compiled.init(&session, graph.forward, .{
        weights.tensors,        zml.Tensor.fromShape(feat_shape), zml.Tensor.fromShape(coords_shape),
        tensors(window_shapes), tensors(@splat(restore_shape)),   tensors(neighbor_shapes),
    });
    defer compiled.deinit(allocator);
    compiled.args.bake(weights.buffers);
    const result = try allocator.alloc(f32, out_shape.count());
    defer allocator.free(result);
    for (0..repeats) |_| {
        try compiled.run(.{ features, coords, windows, inverse, neighbors });
        var output = compiled.results.get(zml.Buffer);
        defer output.deinit();
        if (!output.shape().eql(out_shape)) return error.UnexpectedDecoderOutput;
        raw.placement(&session, try session.verifyPlacement(&output));
        try output.toSlice(session.io, .init(out_shape, std.mem.sliceAsBytes(result)));
    }
    try raw.write(init.io, args[19], result);
}

fn uploadPair(session: *const rt.Session, shapes: [2]zml.Shape, paths: []const [:0]const u8) ![2]zml.Buffer {
    var first = try upload(session, shapes[0], paths[0]);
    errdefer first.deinit();
    return .{ first, try upload(session, shapes[1], paths[1]) };
}

fn tensors(shapes: [2]zml.Shape) [2]zml.Tensor {
    return .{ zml.Tensor.fromShape(shapes[0]), zml.Tensor.fromShape(shapes[1]) };
}
