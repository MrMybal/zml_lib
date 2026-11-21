const std = @import("std");
const model = @import("zml_lib/depth_anything_v2");
const runtime = @import("zml_lib/runtime");

pub const std_options: std.Options = .{ .log_level = .info };
const Mode = enum { trace, depth };

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 9) return error.ExpectedBackendWeightsModeHeightWidthRepeatInputOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const mode = std.meta.stringToEnum(Mode, args[3]) orelse return error.InvalidMode;
    const image = try model.Image.init(try std.fmt.parseInt(i64, args[4], 10), try std.fmt.parseInt(i64, args[5], 10));
    const repeat = try std.fmt.parseInt(u8, args[6], 10);
    if (repeat == 0) return error.InvalidRepeatCount;
    const input = try runtime.probe_io.read(allocator, init.io, args[7], 3 * image.pixels);
    defer allocator.free(input);
    const count = if (mode == .trace) image.feature_count + image.pixels else image.pixels;
    const output = try allocator.alloc(f32, count);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    switch (mode) {
        .trace => {
            var graph = try runtime.TensorModel.loadCheckpoint(&session, &model.schema, args[2], .f16);
            defer graph.deinit();
            const height, const width = .{ image.height, image.width };
            for (0..repeat) |_| {
                runtime.probe_io.placement(&session, try graph.infer(model.trace, .init(.{ 1, 3, height, width }, .f32), .init(.{@as(i64, @intCast(count))}, .f32), input, output));
            }
        },
        .depth => {
            var loaded = try model.Model.load(&session, args[2]);
            defer loaded.deinit();
            for (0..repeat) |_| {
                runtime.probe_io.placement(&session, try loaded.predict(image.height, image.width, input, output));
            }
        },
    }
    try runtime.probe_io.write(init.io, args[8], output);
}
