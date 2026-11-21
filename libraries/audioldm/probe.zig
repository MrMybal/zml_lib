const std = @import("std");
const runtime = @import("zml_lib/runtime");
const audio = @import("zml_lib/audioldm");
const hifigan = @import("zml_lib/hifigan");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 8) return error.ExpectedBackendDecoderVocoderFrameCountLatentMelWaveform;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const latent_frames = try std.fmt.parseInt(u32, args[4], 10);
    const frames = try audio.melFrames(latent_frames);
    const latent_count = try std.math.mul(usize, latent_frames, audio.latent_channels * audio.latent_bands);
    const mel_count = try std.math.mul(usize, frames, audio.mel_bands);
    const latent = try runtime.probe_io.read(allocator, init.io, args[5], latent_count);
    defer allocator.free(latent);
    const mel = try allocator.alloc(f32, mel_count);
    defer allocator.free(mel);
    const waveform = try allocator.alloc(f32, try hifigan.sampleCount(frames));
    defer allocator.free(waveform);

    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var decoder = try audio.Decoder.load(&session, args[2]);
    defer decoder.deinit();
    var vocoder = try hifigan.Model.load(&session, args[3]);
    defer vocoder.deinit();
    runtime.probe_io.placement(&session, try decoder.decode(latent_frames, latent, mel));
    runtime.probe_io.placement(&session, try vocoder.synthesize(frames, mel, waveform));
    try runtime.probe_io.write(init.io, args[6], mel);
    try runtime.probe_io.write(init.io, args[7], waveform);
}
