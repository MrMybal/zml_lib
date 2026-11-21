const std = @import("std");
const runtime = @import("zml_lib/runtime");
const sam2 = @import("zml_lib/sam2");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 7) return error.ExpectedOperationAndFiles;
    const memory_encode = std.mem.eql(u8, args[2], "memory_encode");
    const memory_attention = std.mem.eql(u8, args[2], "memory_attention");
    if (!memory_encode and !memory_attention) return runtime.probe_io.segmentationProbe(sam2, init);
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    const count = if (memory_encode) sam2.memory.memoryInputSize else try std.fmt.parseInt(usize, args[4], 10);
    if (count > sam2.memory.attentionOutputSize + 7 * (sam2.memory.spatialMemory + 1) + 16 * sam2.memory.pointerSize)
        return error.InvalidMemorySize;
    const input = try runtime.probe_io.read(allocator, init.io, args[5], count);
    const output = try allocator.alloc(f32, if (memory_encode) sam2.memory.memoryOutputSize else sam2.memory.attentionOutputSize);
    const shards = if (memory_encode) blk: {
        var encoder = try sam2.memory.Encoder.load(&session, args[3]);
        defer encoder.deinit();
        break :blk try encoder.run(input, output);
    } else blk: {
        var attention = try sam2.memory.Attention.load(&session, args[3]);
        defer attention.deinit();
        break :blk try attention.run(input, output);
    };
    try runtime.probe_io.write(init.io, args[6], output);
    runtime.probe_io.placement(&session, shards);
}
