//! Binary FP32 interface for SD 1.5 component validation and image generation.
const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const unet = @import("unet.zig");
const ddim = @import("ddim.zig");
const vae = runtime.kl_decoder.Decoder(.{ .channels = 4, .scaling = 0.18215, .post_quant = true });

pub const std_options: std.Options = .{ .log_level = .info };

fn schema(init: std.process.Init, parameters: []const runtime.layers.Parameter) !void {
    var scratch: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writerStreaming(init.io, &scratch);
    for (parameters) |param| {
        try writer.interface.writeAll(param.name);
        for (param.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
        try writer.interface.writeByte('\n');
    }
    try writer.interface.flush();
}

fn checkFinite(values: []const f32) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonfiniteBuffer;
}

fn read(init: std.process.Init, source: []const u8, count: usize) ![]f32 {
    const values = try runtime.probe_io.read(init.gpa, init.io, source, count);
    errdefer init.gpa.free(values);
    try checkFinite(values);
    return values;
}

fn decode(init: std.process.Init, session: *const runtime.Session, path: []const u8, height: usize, width: usize, input: []const u8, output: []const u8) !void {
    const area = height * width;
    const latent = try read(init, input, area * vae.channels);
    defer init.gpa.free(latent);
    const rgb = try init.gpa.alloc(f32, area * 3 * vae.factor * vae.factor);
    defer init.gpa.free(rgb);
    var model = try runtime.TensorModel.load(session, &vae.schema, path);
    defer model.deinit();
    runtime.probe_io.placement(session, try vae.decode(&model, @intCast(height), @intCast(width), latent, rgb));
    try checkFinite(rgb);
    try runtime.probe_io.write(init.io, output, rgb);
}

fn denoise(init: std.process.Init, session: *const runtime.Session, path: []const u8, height: usize, width: usize, rest: []const [:0]const u8, sample: bool) !void {
    const count = height * width * unet.channels;
    const embedding_len = unet.context_length * unet.context_width;
    const step = try std.fmt.parseInt(usize, rest[0], 10);
    const guidance = if (sample) try std.fmt.parseFloat(f32, rest[1]) else 1;
    if (!std.math.isFinite(guidance) or guidance < 1 or guidance > 20) return error.InvalidGuidance;
    const schedule = if (sample) try ddim.Schedule.init(step) else null;
    if (!sample and step >= 1000) return error.InvalidTimestep;
    const input = try read(init, rest[rest.len - 2], count + embedding_len * (1 + @as(usize, @intFromBool(guidance > 1))));
    defer init.gpa.free(input);
    const latent = input[0..count];
    const epsilon = try init.gpa.alloc(f32, count);
    defer init.gpa.free(epsilon);
    var model = try unet.Model.load(session, path, @intCast(height), @intCast(width));
    defer model.deinit();
    var positive = try model.condition(input[count..][0..embedding_len]);
    defer positive.deinit();
    var negative: ?zml.Buffer = if (guidance > 1) try model.condition(input[count + embedding_len ..]) else null;
    defer if (negative) |*buffer| buffer.deinit();
    const unconditioned = try init.gpa.alloc(f32, if (negative != null) count else 0);
    defer init.gpa.free(unconditioned);

    if (schedule) |plan| {
        for (0..step) |i| {
            const t = plan.timestep(i);
            runtime.probe_io.placement(session, try model.predict(t, latent, positive, epsilon));
            if (negative) |condition| {
                runtime.probe_io.placement(session, try model.predict(t, latent, condition, unconditioned));
                for (epsilon, unconditioned) |*cond, none| cond.* = none + guidance * (cond.* - none);
            }
            try checkFinite(epsilon);
            plan.step(i, latent, epsilon);
            try checkFinite(latent);
        }
    } else {
        runtime.probe_io.placement(session, try model.predict(step, latent, positive, epsilon));
        try checkFinite(epsilon);
    }
    try runtime.probe_io.write(init.io, rest[rest.len - 1], if (sample) latent else epsilon);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedMode;
    const Mode = enum { unet, sample, vae };
    const mode = std.meta.stringToEnum(Mode, args[1]) orelse return error.InvalidMode;
    if (args.len == 3 and std.mem.eql(u8, args[2], "schema"))
        return schema(init, if (mode == .vae) &vae.schema else &unet.schema);
    const expected: usize = switch (mode) {
        .unet => 9,
        .sample => 10,
        .vae => 8,
    };
    if (args.len != expected) return error.InvalidArguments;
    const backend = std.meta.stringToEnum(runtime.Backend, args[2]) orelse return error.InvalidBackend;
    const height = try std.fmt.parseInt(usize, args[4], 10);
    const width = try std.fmt.parseInt(usize, args[5], 10);
    if (height < 8 or height > 64 or height % 8 != 0 or width < 8 or width > 64 or width % 8 != 0)
        return error.InvalidLatentShape;
    const session = try runtime.Session.open(init.gpa, init.io, backend);
    defer session.close();
    switch (mode) {
        .vae => return decode(init, &session, args[3], height, width, args[6], args[7]),
        .unet, .sample => return denoise(init, &session, args[3], height, width, args[6..], mode == .sample),
    }
}
