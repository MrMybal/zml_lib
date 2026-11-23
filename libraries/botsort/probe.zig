const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const cost = @import("cost.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const max_tracks = 16384;
const max_detections = 512;
const fuse_shape: zml.Shape = .init(.{1}, .i32);

const Matrix = struct {
    tracks: []f32,
    detections: []f32,
    values: []f32,
    track_shape: zml.Shape,
    detection_shape: zml.Shape,
    output_shape: zml.Shape,
    graph: rt.Compiled,

    fn init(session: *const rt.Session, rows: usize, columns: usize) !Matrix {
        const allocator = session.allocator;
        const track_shape: zml.Shape = .init(.{ rows, cost.track_fields }, .f32);
        const detection_shape: zml.Shape = .init(.{ columns, cost.detection_fields }, .f32);
        const output_shape: zml.Shape = .init(.{ rows, columns }, .f32);
        const tracks = try allocator.alloc(f32, track_shape.count());
        errdefer allocator.free(tracks);
        const detections = try allocator.alloc(f32, detection_shape.count());
        errdefer allocator.free(detections);
        const values = try allocator.alloc(f32, output_shape.count());
        errdefer allocator.free(values);
        const graph = try rt.Compiled.init(session, cost.forward, .{
            zml.Tensor.fromShape(track_shape), zml.Tensor.fromShape(detection_shape), zml.Tensor.fromShape(fuse_shape),
        });
        std.log.info("botsort_cost_capacity={d}x{d}", .{ rows, columns });
        return .{ .tracks = tracks, .detections = detections, .values = values, .track_shape = track_shape, .detection_shape = detection_shape, .output_shape = output_shape, .graph = graph };
    }

    fn deinit(self: *const Matrix, allocator: std.mem.Allocator) void {
        self.graph.deinit(allocator);
        allocator.free(self.values);
        allocator.free(self.detections);
        allocator.free(self.tracks);
    }

    fn evaluate(self: *Matrix, session: *const rt.Session, fuse: i32) !void {
        const io = session.io;
        var tracks = try zml.Buffer.fromBytesOpts(io, session.platform, self.track_shape, std.mem.sliceAsBytes(self.tracks), .{ .wait = true });
        defer tracks.deinit();
        var detections = try zml.Buffer.fromBytesOpts(io, session.platform, self.detection_shape, std.mem.sliceAsBytes(self.detections), .{ .wait = true });
        defer detections.deinit();
        var mode = try zml.Buffer.fromBytesOpts(io, session.platform, fuse_shape, std.mem.asBytes(&fuse), .{ .wait = true });
        defer mode.deinit();
        try self.graph.run(.{ tracks, detections, mode });
        var result = self.graph.results.get(zml.Buffer);
        defer result.deinit();
        if (!result.shape().eql(self.output_shape)) return error.InvalidCostOutput;
        rt.probe_io.placement(session, try session.verifyPlacement(&result));
        try result.toSlice(io, .init(self.output_shape, std.mem.sliceAsBytes(self.values)));
    }
};

fn readRows(reader: *std.Io.Reader, store: []f32, count: usize) !void {
    const used = store[0..count];
    try reader.readSliceAll(std.mem.sliceAsBytes(used));
    for (used) |value| if (!std.math.isFinite(value)) return error.NonFiniteAssociationInput;
    @memset(store[count..], 0);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedBackend;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    const allocator = std.heap.page_allocator;
    const io = init.io;
    const session = try rt.Session.open(allocator, io, backend);
    defer session.close();
    var workspace: ?Matrix = null;
    defer if (workspace) |*matrix| matrix.deinit(allocator);
    var row_capacity: usize = 0;
    var column_capacity: usize = 0;
    var in_buffer: [8192]u8 = undefined;
    var out_buffer: [8192]u8 = undefined;
    var input = std.Io.File.stdin().readerStreaming(io, &in_buffer);
    var output = std.Io.File.stdout().writerStreaming(io, &out_buffer);
    while (true) {
        _ = input.interface.peekByte() catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        var header: [12]u8 = undefined;
        try input.interface.readSliceAll(&header);
        const rows: usize = std.mem.readInt(u32, header[0..4], .little);
        const columns: usize = std.mem.readInt(u32, header[4..8], .little);
        const fuse = std.mem.readInt(i32, header[8..12], .little);
        if (rows == 0 or rows > max_tracks or columns == 0 or columns > max_detections or (fuse != 0 and fuse != 1))
            return error.InvalidAssociationRequest;
        if (rows > row_capacity or columns > column_capacity) {
            row_capacity = @max(row_capacity, try std.math.ceilPowerOfTwo(usize, @max(8, rows)));
            column_capacity = @max(column_capacity, try std.math.ceilPowerOfTwo(usize, @max(8, columns)));
            const replacement = try Matrix.init(&session, row_capacity, column_capacity);
            if (workspace) |*previous| previous.deinit(allocator);
            workspace = replacement;
        }
        const matrix = &workspace.?;
        try readRows(&input.interface, matrix.tracks, rows * cost.track_fields);
        try readRows(&input.interface, matrix.detections, columns * cost.detection_fields);
        try matrix.evaluate(&session, fuse);
        for (0..rows) |row| {
            const active = matrix.values[row * column_capacity ..][0..columns];
            for (active) |value| if (!std.math.isFinite(value)) return error.NonFiniteAssociationCost;
            try output.interface.writeAll(std.mem.sliceAsBytes(active));
        }
        try output.interface.flush();
    }
}
