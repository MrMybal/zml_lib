const std = @import("std");
const runtime = @import("zml_lib/runtime");
const tripo = @import("zml_lib/triposr");
const raw = runtime.probe_io;

const Operation = enum { schema, encode, points, grid };
const ProfileName = enum { official, fixture };
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedOperationAndProfile;
    const operation = std.meta.stringToEnum(Operation, args[1]) orelse return error.InvalidOperation;
    const profile = std.meta.stringToEnum(ProfileName, args[2]) orelse return error.InvalidProfile;
    switch (profile) {
        .official => try run(tripo.official, operation, init, args[3..]),
        .fixture => try run(tripo.fixture, operation, init, args[3..]),
    }
}

fn writeSchema(comptime p: tripo.Profile, io: std.Io) !void {
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buffer);
    const output = &stdout.interface;
    const parts = .{ .{ "encoder", tripo.encoderSchema(p) }, .{ "field", tripo.fieldSchema(p) } };
    inline for (parts) |part| {
        for (part[1]) |parameter| {
            try output.print("{s}\t{s}", .{ part[0], parameter.name });
            for (parameter.shape.dims()) |dimension| try output.print("\t{d}", .{dimension});
            try output.writeByte('\n');
        }
    }
    try output.flush();
}

fn run(comptime p: tripo.Profile, operation: Operation, init: std.process.Init, args: []const [:0]const u8) !void {
    if (operation == .schema) {
        if (args.len != 0) return error.UnexpectedArguments;
        return writeSchema(p, init.io);
    }
    const expected: usize = switch (operation) {
        .encode, .grid => 5,
        .points => 6,
        .schema => unreachable,
    };
    if (args.len != expected) return error.UnexpectedArguments;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const allocator = std.heap.page_allocator;
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    if (operation == .encode) {
        const repeat = try std.fmt.parseInt(u16, args[4], 10);
        if (repeat == 0) return error.InvalidRepetitionCount;
        const input = try raw.read(allocator, init.io, args[2], tripo.Encoder(p).input_count);
        defer allocator.free(input);
        const planes = try allocator.alloc(f32, p.planeCount());
        defer allocator.free(planes);
        var encoder = try tripo.Encoder(p).load(&session, args[1]);
        defer encoder.deinit();
        for (0..repeat) |_| raw.placement(&session, try encoder.encode(input, planes));
        try raw.write(init.io, args[3], planes);
        return;
    }
    const size = try std.fmt.parseInt(usize, args[3], 10);
    if (size == 0 or (operation == .grid and (size < 2 or size > 256))) return error.InvalidQuerySize;
    const count = if (operation == .grid) size * size * size else size;
    const points = if (operation == .points) try raw.read(allocator, init.io, args[4], try std.math.mul(usize, count, 3)) else null;
    defer if (points) |values| allocator.free(values);
    const planes = try raw.read(allocator, init.io, args[2], p.planeCount());
    defer allocator.free(planes);
    var field = try tripo.Field(p).load(&session, args[1]);
    defer field.deinit();
    const result_width: usize = if (operation == .grid) 1 else tripo.field_channels;
    const output = try allocator.alloc(f32, try std.math.mul(usize, count, result_width));
    defer allocator.free(output);
    try query(p, &session, &field, planes, points, size, count, output);
    try raw.write(init.io, args[args.len - 1], output);
}

fn gridPoint(index: usize, size: usize) [3]f32 {
    const scale = (2 * tripo.radius) / @as(f32, @floatFromInt(size - 1));
    return .{
        @as(f32, @floatFromInt(index / (size * size))) * scale - tripo.radius,
        @as(f32, @floatFromInt((index / size) % size)) * scale - tripo.radius,
        @as(f32, @floatFromInt(index % size)) * scale - tripo.radius,
    };
}

fn query(
    comptime p: tripo.Profile,
    session: *const runtime.Session,
    field: *tripo.Field(p),
    planes: []const f32,
    points: ?[]const f32,
    size: usize,
    count: usize,
    output: []f32,
) !void {
    // Zig can narrow @min with a comptime bound. The allocation uses products of this count.
    const batch: usize = @min(count, tripo.query_batch);
    const coordinates = try session.allocator.alloc(f32, 3 * batch);
    defer session.allocator.free(coordinates);
    const values = try session.allocator.alloc(f32, batch * tripo.field_channels);
    defer session.allocator.free(values);
    var offset: usize = 0;
    while (offset < count) : (offset += batch) {
        const length = @min(batch, count - offset);
        if (points) |positions| {
            @memcpy(coordinates[0 .. 3 * length], positions[3 * offset ..][0 .. 3 * length]);
        } else {
            for (0..length) |i| coordinates[3 * i ..][0..3].* = gridPoint(offset + i, size);
        }
        @memset(coordinates[3 * length ..], 0);
        const shards = try field.query(planes, coordinates, values);
        if (offset == 0) raw.placement(session, shards);
        if (points == null) {
            for (output[offset..][0..length], 0..) |*dst, i| dst.* = values[i * tripo.field_channels];
        } else {
            @memcpy(output[offset * tripo.field_channels ..][0 .. length * tripo.field_channels], values[0 .. length * tripo.field_channels]);
        }
    }
}
