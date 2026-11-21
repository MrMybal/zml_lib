const std = @import("std");
const runtime = @import("zml_lib/runtime");
const vocoder = @import("zml_lib/hifigan");
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 6) return error.ExpectedBackendWeightsFramesMelWaveform;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const frames = try std.fmt.parseInt(u32, args[3], 10);
    const output = try allocator.alloc(f32, try vocoder.sampleCount(frames));
    defer allocator.free(output);
    const mel = try runtime.probe_io.read(allocator, init.io, args[4], @as(usize, frames) * vocoder.mel_bands);
    defer allocator.free(mel);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try vocoder.Model.load(&session, args[2]);
    defer model.deinit();
    runtime.probe_io.placement(&session, try model.synthesize(frames, mel, output));
    try runtime.probe_io.write(init.io, args[5], output);
}
