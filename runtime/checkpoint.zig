const std = @import("std");
const zml = @import("zml/core");

const Region = struct {
    start: u64,
    end: u64,

    fn lessThan(_: void, a: Region, b: Region) bool {
        return a.start < b.start;
    }
};

pub const Storage = enum { f32, f16, e4m3 };

fn checkTensor(value: std.json.Value, shape: zml.Shape, storage: Storage) !Region {
    if (value != .object) return error.InvalidCheckpoint;
    const dtype = value.object.get("dtype") orelse return error.InvalidCheckpoint;
    const dimensions = value.object.get("shape") orelse return error.InvalidCheckpoint;
    const offsets = value.object.get("data_offsets") orelse return error.InvalidCheckpoint;
    const expected_dtype = switch (storage) {
        .f32 => "F32",
        .f16 => "F16",
        .e4m3 => "F8_E4M3",
    };
    if (dtype != .string or !std.mem.eql(u8, dtype.string, expected_dtype)) return error.CheckpointDtypeMismatch;
    if (dimensions != .array or dimensions.array.items.len != shape.rank()) return error.CheckpointShapeMismatch;
    for (dimensions.array.items, shape.dims()) |dim, expected| {
        if (dim != .integer or dim.integer != expected) return error.CheckpointShapeMismatch;
    }
    if (offsets != .array or offsets.array.items.len != 2) return error.InvalidCheckpoint;
    const start, const end = offsets.array.items[0..2].*;
    if (start != .integer or end != .integer or start.integer < 0 or end.integer < start.integer)
        return error.InvalidCheckpoint;
    const bytes_per_value: usize = switch (storage) {
        .f32 => @sizeOf(f32),
        .f16 => @sizeOf(f16),
        .e4m3 => @sizeOf(u8),
    };
    const expected_bytes = shape.count() * bytes_per_value;
    if (end.integer - start.integer != expected_bytes) return error.InvalidCheckpoint;
    return .{ .start = @intCast(start.integer), .end = @intCast(end.integer) };
}

pub const Parameters = struct {
    tensors: []const zml.Tensor,
    buffers: []zml.Buffer,

    pub fn deinit(self: Parameters, allocator: std.mem.Allocator) void {
        for (self.buffers) |*buffer| buffer.deinit();
        allocator.free(self.buffers);
        allocator.free(self.tensors);
    }
};

pub fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    count: usize,
    spec_context: anytype,
    comptime specAt: anytype,
    path: []const u8,
) !Parameters {
    return loadWithStorage(allocator, io, platform, count, spec_context, specAt, path, .f32);
}

pub fn loadWithStorage(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    count: usize,
    spec_context: anytype,
    comptime specAt: anytype,
    path: []const u8,
    storage: Storage,
) !Parameters {
    return loadParameters(allocator, io, platform, count, spec_context, specAt, path, storage);
}

/// Load FP32 tensors with an E4M3FN representation allowed for linear weights.
pub fn loadMixedE4M3(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    count: usize,
    spec_context: anytype,
    comptime specAt: anytype,
    path: []const u8,
) !Parameters {
    return loadParameters(allocator, io, platform, count, spec_context, specAt, path, null);
}

fn loadParameters(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    count: usize,
    spec_context: anytype,
    comptime specAt: anytype,
    path: []const u8,
    storage: ?Storage,
) !Parameters {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const file_size = try file.length(io);
    var prefix: [8]u8 = undefined;
    if (file_size < prefix.len) return error.InvalidCheckpoint;
    if (try file.readPositionalAll(io, &prefix, 0) != prefix.len) return error.InvalidCheckpoint;
    const header_len = std.mem.readInt(u64, &prefix, .little);
    if (header_len == 0 or header_len > 1024 * 1024 or header_len > file_size - prefix.len)
        return error.InvalidCheckpoint;
    const data_start = prefix.len + header_len;
    const header = try allocator.alloc(u8, @intCast(header_len));
    defer allocator.free(header);
    if (try file.readPositionalAll(io, header, prefix.len) != header.len) return error.InvalidCheckpoint;
    if (header[0] != '{') return error.InvalidCheckpoint;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidCheckpoint,
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidCheckpoint;
    const object = parsed.value.object;
    const metadata = object.get("__metadata__");
    if (metadata) |value| {
        if (value != .object) return error.InvalidCheckpoint;
        for (value.object.values()) |entry| {
            if (entry != .string) return error.InvalidCheckpoint;
        }
    }
    if (object.count() != count + @intFromBool(metadata != null)) return error.CheckpointTensorCountMismatch;

    const tensors = try allocator.alloc(zml.Tensor, count);
    errdefer allocator.free(tensors);
    const regions = try allocator.alloc(Region, count);
    defer allocator.free(regions);
    var max_size: usize = 0;
    var max_widen_count: usize = 0;
    for (tensors, regions, 0..) |*tensor, *region, index| {
        var name_buffer: [64]u8 = undefined;
        const spec = specAt(spec_context, index, &name_buffer);
        const entry = object.get(spec.name) orelse {
            std.log.err("Missing tensor {s}", .{spec.name});
            return error.CheckpointTensorMissing;
        };
        var shape = spec.shape;
        const format = storage orelse blk: {
            if (entry != .object or shape.dtype() != .f32) return error.CheckpointDtypeMismatch;
            const dtype = entry.object.get("dtype") orelse return error.CheckpointDtypeMismatch;
            if (dtype != .string) return error.CheckpointDtypeMismatch;
            if (std.mem.eql(u8, dtype.string, "F32")) break :blk Storage.f32;
            if (!std.mem.eql(u8, dtype.string, "F8_E4M3") or shape.rank() != 2 or
                !std.mem.endsWith(u8, spec.name, ".weight")) return error.CheckpointDtypeMismatch;
            shape = shape.withDtype(.u8);
            break :blk Storage.e4m3;
        };
        const dtype = shape.dtype();
        const compatible = switch (format) {
            .f32 => dtype == .f32,
            .f16 => dtype == .f16 or dtype == .f32,
            .e4m3 => dtype == .u8,
        };
        if (!compatible) return error.CheckpointDtypeMismatch;
        region.* = checkTensor(entry, shape, format) catch |err| {
            std.log.err("{s}: {s}", .{ spec.name, @errorName(err) });
            return err;
        };
        max_size = @max(max_size, shape.byteSize());
        if (format == .f16 and dtype == .f32)
            max_widen_count = @max(max_widen_count, shape.count());
        tensor.* = .fromShape(shape);
    }

    const sorted = try allocator.dupe(Region, regions);
    defer allocator.free(sorted);
    std.mem.sort(Region, sorted, {}, Region.lessThan);
    var next_offset: u64 = 0;
    for (sorted) |region| {
        if (region.start != next_offset) return error.InvalidCheckpoint;
        next_offset = region.end;
    }
    if (next_offset != file_size - data_start) return error.InvalidCheckpoint;

    const scratch = try allocator.alignedAlloc(u8, .@"64", max_size);
    defer allocator.free(scratch);
    const half = try allocator.alloc(f16, max_widen_count);
    defer allocator.free(half);
    const buffers = try allocator.alloc(zml.Buffer, count);
    errdefer allocator.free(buffers);
    var uploaded: usize = 0;
    errdefer for (buffers[0..uploaded]) |*buffer| buffer.deinit();
    for (tensors, regions, buffers) |tensor, region, *buffer| {
        const data = scratch[0..tensor.shape().byteSize()];
        const widen = storage == .f16 and tensor.shape().dtype() == .f32;
        const saved = if (widen) std.mem.sliceAsBytes(half[0..tensor.shape().count()]) else data;
        if (try file.readPositionalAll(io, saved, data_start + region.start) != saved.len)
            return error.InvalidCheckpoint;
        if (widen) {
            for (std.mem.bytesAsSlice(f32, data), half[0..tensor.shape().count()]) |*out, value|
                out.* = @floatCast(value);
        }
        buffer.* = try .fromBytesOpts(io, platform, tensor.shape(), data, .{ .wait = true });
        uploaded += 1;
    }
    return .{ .tensors = tensors, .buffers = buffers };
}
