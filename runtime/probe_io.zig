const std = @import("std");
const runtime = @import("session.zig");

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
