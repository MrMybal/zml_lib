const std = @import("std");
const runtime = @import("zml_lib/runtime");
const vae = @import("zml_lib/taef1");
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) return error.ExpectedBackendWeightsHeightWidthLatentRGB;
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    const height = try std.fmt.parseInt(u32, args[3], 10);
    const width = try std.fmt.parseInt(u32, args[4], 10);
    const area = try std.math.mul(usize, height, width);
    const latent = try runtime.probe_io.read(allocator, init.io, args[5], try std.math.mul(usize, area, vae.latent_channels));
    defer allocator.free(latent);
    const rgb = try allocator.alloc(f32, try std.math.mul(usize, area, 3 * vae.factor * vae.factor));
    defer allocator.free(rgb);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try vae.Decoder.load(&session, args[2]);
    defer model.deinit();
    runtime.probe_io.placement(&session, try model.decode(height, width, latent, rgb));
    try runtime.probe_io.write(init.io, args[6], rgb);
}
