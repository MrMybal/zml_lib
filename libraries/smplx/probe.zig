const std = @import("std");
const rt = @import("zml_lib/runtime");
const smplx = @import("zml_lib/smplx");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) return error.ExpectedBackendWeightsFramesInputOutputRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    const frames = try std.fmt.parseInt(usize, args[3], 10);
    if (frames == 0 or frames > 8192) return error.InvalidFrameCount;
    const repeats = try std.fmt.parseInt(u16, args[6], 10);
    if (repeats == 0) return error.InvalidRepeatCount;

    const allocator = std.heap.page_allocator;
    const input = try rt.probe_io.read(allocator, init.io, args[4], frames * smplx.input_width);
    defer allocator.free(input);
    const output_width = smplx.output_points * 3;
    const output = try allocator.alloc(f32, frames * output_width);
    defer allocator.free(output);
    const batch = try allocator.alloc(f32, smplx.batch_size * smplx.input_width);
    defer allocator.free(batch);
    const points = try allocator.alloc(f32, smplx.batch_size * output_width);
    defer allocator.free(points);

    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try smplx.Model.load(&session, args[2]);
    defer model.deinit();
    for (0..repeats) |_| {
        var first: usize = 0;
        while (first < frames) {
            const count: usize = @min(frames - first, smplx.batch_size);
            const values = count * smplx.input_width;
            const start = first * smplx.input_width;
            @memcpy(batch[0..values], input[start..][0..values]);
            @memset(batch[values..], 0);
            rt.probe_io.placement(&session, try model.predict(smplx.batch_size, batch, points));
            @memcpy(output[first * output_width ..][0 .. count * output_width], points[0 .. count * output_width]);
            first += count;
        }
    }
    try rt.probe_io.write(init.io, args[5], output);
}
