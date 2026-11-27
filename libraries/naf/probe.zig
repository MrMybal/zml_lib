const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const naf = @import("zml_lib/naf");

pub const std_options: std.Options = .{ .log_level = .info };

fn positive(s: []const u8, max: i64) !i64 {
    const n = try std.fmt.parseInt(i64, s, 10);
    if (n < 1 or n > max) return error.InvalidSize;
    return n;
}

fn buffer(session: *const rt.Session, shape: zml.Shape, input: []const f32) !zml.Buffer {
    return zml.Buffer.fromBytesOpts(session.io, session.platform, shape, std.mem.sliceAsBytes(input), .{ .wait = true });
}

fn finite(values: []const f32) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonfiniteValue;
}

fn read(session: *const rt.Session, path: []const u8, shape: zml.Shape) ![]f32 {
    const input = try rt.probe_io.read(session.allocator, session.io, path, shape.count());
    errdefer session.allocator.free(input);
    try finite(input);
    return input;
}

fn encode(comptime forward: anytype, comptime stride: i64, session: *const rt.Session, args: []const [:0]const u8) !void {
    if (args.len != 6) return error.ExpectedWeightsHeightWidthInputOutputRepeats;
    const height = try positive(args[1], 1024);
    const width = try positive(args[2], 1024);
    if (@mod(height, stride) != 0 or @mod(width, stride) != 0 or @min(height, width) < 2)
        return error.InvalidImageDimensions;
    const repeats = try positive(args[5], 64);
    const input_shape: zml.Shape = .init(.{ 1, 3, height, width }, .f32);
    const out_shape: zml.Shape = .init(.{ 1, naf.channels, @divExact(height, stride), @divExact(width, stride) }, .f32);
    const alloc = session.allocator;
    const input = try read(session, args[3], input_shape);
    defer alloc.free(input);
    const output = try alloc.alloc(f32, out_shape.count());
    defer alloc.free(output);
    var model = try rt.TensorModel.load(session, &naf.schema, args[0]);
    defer model.deinit();
    for (0..@as(usize, @intCast(repeats))) |_| {
        const shards = try model.infer(forward, input_shape, out_shape, input, output);
        try finite(output);
        rt.probe_io.placement(session, shards);
    }
    try rt.probe_io.write(session.io, args[4], output);
}

fn sample(session: *const rt.Session, args: []const [:0]const u8) !void {
    if (args.len != 10) return error.ExpectedShapeAndFiles;
    const height = try positive(args[0], 1024);
    const width = try positive(args[1], 1024);
    const rows = try positive(args[2], height);
    const cols = try positive(args[3], width);
    const depth = try positive(args[4], 1024);
    const count = try positive(args[5], 65_536);
    if (@min(rows, cols) < naf.window or @mod(height, rows) != 0 or @mod(width, cols) != 0 or @mod(depth, naf.heads) != 0)
        return error.InvalidNeighborhoodShape;
    const chunk = @min(count, 128);
    const q_shape: zml.Shape = .init(.{ 1, naf.channels, height, width }, .f32);
    const v_shape: zml.Shape = .init(.{ 1, depth, rows, cols }, .f32);
    const p_shape: zml.Shape = .init(.{ chunk, 2 }, .f32);
    const out_shape: zml.Shape = .init(.{ chunk, depth }, .f32);
    const alloc = session.allocator;
    const queries = try read(session, args[6], q_shape);
    defer alloc.free(queries);
    const features = try read(session, args[7], v_shape);
    defer alloc.free(features);
    const grid = try read(session, args[8], .init(.{ count, 2 }, .f32));
    defer alloc.free(grid);
    const output = try alloc.alloc(f32, @intCast(count * depth));
    defer alloc.free(output);
    const positions = try alloc.alloc(f32, p_shape.count());
    defer alloc.free(positions);
    const result = try alloc.alloc(f32, out_shape.count());
    defer alloc.free(result);
    var compiled = try rt.Compiled.init(session, naf.sample, .{
        zml.Tensor.fromShape(q_shape), zml.Tensor.fromShape(v_shape), zml.Tensor.fromShape(p_shape),
    });
    defer compiled.deinit(alloc);
    var q_buffer = try buffer(session, q_shape, queries);
    defer q_buffer.deinit();
    var v_buffer = try buffer(session, v_shape, features);
    defer v_buffer.deinit();

    var begin: i64 = 0;
    while (begin < count) : (begin += chunk) {
        const length = @min(chunk, count - begin);
        const first: usize = @intCast(begin * 2);
        const samples: usize = @intCast(length * 2);
        @memcpy(positions[0..samples], grid[first..][0..samples]);
        @memset(positions[samples..], 0);
        var p_buffer = try buffer(session, p_shape, positions);
        defer p_buffer.deinit();
        try compiled.run(.{ q_buffer, v_buffer, p_buffer });
        var y = compiled.results.get(zml.Buffer);
        defer y.deinit();
        if (!y.shape().eql(out_shape)) return error.UnexpectedOutputShape;
        const shards = try session.verifyPlacement(&y);
        try y.toSlice(session.io, .init(out_shape, std.mem.sliceAsBytes(result)));
        try finite(result);
        const elements: usize = @intCast(length * depth);
        @memcpy(output[@intCast(begin * depth)..][0..elements], result[0..elements]);
        rt.probe_io.placement(session, shards);
    }
    try rt.probe_io.write(session.io, args[9], output);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var storage: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &storage);
        for (naf.schema) |p| {
            try writer.interface.writeAll(p.name);
            for (p.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (args.len < 3) return error.ExpectedOperationAndBackend;
    const backend = std.meta.stringToEnum(rt.Backend, args[2]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const session = try rt.Session.open(init.gpa, init.io, backend);
    defer session.close();
    if (std.mem.eql(u8, args[1], "encode")) return encode(naf.encodeSame, 1, &session, args[3..]);
    if (std.mem.eql(u8, args[1], "encode-half")) return encode(naf.encodeHalf, 2, &session, args[3..]);
    if (std.mem.eql(u8, args[1], "sample")) return sample(&session, args[3..]);
    return error.InvalidOperation;
}
