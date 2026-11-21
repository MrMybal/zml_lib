const std = @import("std");
const runtime = @import("zml_lib/runtime");
const audio = @import("zml_lib/audioldm");
const hifigan = @import("zml_lib/hifigan");
const clap = @import("zml_lib/clap_text");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 14 or args.len - 12 > clap.context_limit) {
        std.log.err("Expected BACKEND CLAP UNET VAE VOCODER FRAMES STEPS GUIDANCE INITIAL DENOISED AUDIO TOKEN_IDS...", .{});
        return error.InvalidArguments;
    }
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const latent_frames = try std.fmt.parseInt(u32, args[6], 10);
    const steps = try std.fmt.parseInt(usize, args[7], 10);
    const guidance = try std.fmt.parseFloat(f32, args[8]);
    const frames = try audio.melFrames(latent_frames);
    const latent_count = try std.math.mul(usize, latent_frames, audio.latent_channels * audio.latent_bands);
    const latent = try runtime.probe_io.read(allocator, init.io, args[9], latent_count);
    defer allocator.free(latent);
    const mel = try allocator.alloc(f32, try std.math.mul(usize, frames, audio.mel_bands));
    defer allocator.free(mel);
    const waveform = try allocator.alloc(f32, try hifigan.sampleCount(frames));
    defer allocator.free(waveform);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();

    var positive: [clap.embedding_size]f32 = undefined;
    var negative: [clap.embedding_size]f32 = undefined;
    {
        var encoder = try clap.Model.load(&session, args[2]);
        defer encoder.deinit();
        const count = args.len - 12;
        var ids: [clap.context_limit]u32 = undefined;
        var mask: [clap.context_limit]u8 = undefined;
        for (args[12..], ids[0..count], mask[0..count]) |arg, *id, *present| {
            id.* = try std.fmt.parseInt(u32, arg, 10);
            present.* = @intFromBool(id.* != 1);
        }
        runtime.probe_io.placement(&session, try encoder.encode(ids[0..count], mask[0..count], &positive));
        if (guidance > 1) {
            @memset(ids[0..count], 1);
            @memset(mask[0..count], 0);
            ids[0..2].* = .{ 0, 2 };
            mask[0..2].* = @splat(1);
            runtime.probe_io.placement(&session, try encoder.encode(ids[0..count], mask[0..count], &negative));
        }
    }
    {
        var denoiser = try audio.Denoiser.load(&session, args[3]);
        defer denoiser.deinit();
        runtime.probe_io.placement(&session, try denoiser.generateLatent(latent_frames, steps, latent, &positive, if (guidance > 1) &negative else null, guidance));
    }
    {
        var decoder = try audio.Decoder.load(&session, args[4]);
        defer decoder.deinit();
        runtime.probe_io.placement(&session, try decoder.decode(latent_frames, latent, mel));
    }
    var vocoder = try hifigan.Model.load(&session, args[5]);
    defer vocoder.deinit();
    runtime.probe_io.placement(&session, try vocoder.synthesize(frames, mel, waveform));
    try runtime.probe_io.write(init.io, args[10], latent);
    try runtime.probe_io.write(init.io, args[11], waveform);
}
