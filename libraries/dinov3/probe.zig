const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const dino = @import("zml_lib/dinov3");

pub const std_options: std.Options = .{ .log_level = .info };

fn side(text: []const u8) !i64 {
    const n = try std.fmt.parseInt(i64, text, 10);
    if (n < dino.patch_size or n > 1024 or @mod(n, dino.patch_size) != 0)
        return error.InvalidImageDimensions;
    return n;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var storage: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &storage);
        for (dino.schema) |p| {
            try writer.interface.writeAll(p.name);
            for (p.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (args.len < 7 or @mod(args.len - 5, 2) != 0) return error.ExpectedBackendWeightsShapeAndInputOutputPairs;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const height = try side(args[3]);
    const width = try side(args[4]);
    const tokens = dino.prefix_tokens + @divExact(height * width, dino.patch_size * dino.patch_size);
    const input_shape: zml.Shape = .init(.{ 1, 3, height, width }, .f32);
    const out_shape: zml.Shape = .init(.{ 1, tokens, dino.channels }, .f32);
    const alloc = init.gpa;
    const session = try rt.Session.open(alloc, init.io, backend);
    defer session.close();
    var model = try rt.TensorModel.load(&session, &dino.schema, args[2]);
    defer model.deinit();
    const output = try alloc.alloc(f32, out_shape.count());
    defer alloc.free(output);
    var i: usize = 5;
    while (i < args.len) : (i += 2) {
        const input = try rt.probe_io.read(alloc, init.io, args[i], input_shape.count());
        defer alloc.free(input);
        for (input) |v| if (!std.math.isFinite(v)) return error.InvalidInput;
        const shards = try model.infer(dino.forward, input_shape, out_shape, input, output);
        for (output) |v| if (!std.math.isFinite(v)) return error.NonfiniteFeature;
        rt.probe_io.placement(&session, shards);
        try rt.probe_io.write(init.io, args[i + 1], output);
    }
}
