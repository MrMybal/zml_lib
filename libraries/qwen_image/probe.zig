const std = @import("std");
const builtin = @import("builtin");
const runtime = @import("zml_lib/runtime");
const model = @import("zml_lib/qwen_image");
const vae = model.vae;
const diffusion = model.denoiser;
pub const std_options: std.Options = .{ .log_level = .info };

fn writeSchema(init: std.process.Init, schema: []const runtime.layers.Parameter) !void {
    var scratch: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writerStreaming(init.io, &scratch);
    for (schema) |param| {
        try writer.interface.writeAll(param.name);
        for (param.shape.dims()) |d| try writer.interface.print("\t{d}", .{d});
        try writer.interface.writeByte('\n');
    }
    try writer.interface.flush();
}

fn backend(value: []const u8) !runtime.Backend {
    const selected = std.meta.stringToEnum(runtime.Backend, value) orelse return error.InvalidBackend;
    if (selected == .auto) return error.ExplicitBackendRequired;
    return selected;
}

fn checkFinite(values: []const f32) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonfiniteTensor;
}

fn predictVelocity(instance: anytype, session: *const runtime.Session, input: []const f32, output: []f32) !void {
    const shards = try instance.predict(input, output);
    try checkFinite(output);
    runtime.probe_io.placement(session, shards);
}

fn autoencode(comptime mode: vae.Mode, init: std.process.Init, args: []const [:0]const u8) !void {
    const Vae = vae.Autoencoder(mode);
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) return writeSchema(init, &Vae.schema);
    if (args.len != 6) return error.ExpectedBackendWeightsLatentHeightLatentWidthInputOutput;
    const device = try backend(args[0]);
    const height = try std.fmt.parseInt(usize, args[2], 10);
    const width = try std.fmt.parseInt(usize, args[3], 10);
    if (height == 0 or width == 0 or height > vae.max_latent_size or width > vae.max_latent_size) return error.InvalidSize;
    const rgb = height * width * 3 * vae.factor * vae.factor;
    const latent = height * width * vae.channels;
    const input = try runtime.probe_io.read(init.gpa, init.io, args[4], if (mode == .encode) rgb else latent);
    defer init.gpa.free(input);
    try checkFinite(input);
    const output = try init.gpa.alloc(f32, if (mode == .encode) latent else rgb);
    defer init.gpa.free(output);
    const session = try runtime.Session.open(init.gpa, init.io, device);
    defer session.close();
    var instance = try Vae.load(&session, args[1]);
    defer instance.deinit();
    const shards = try instance.infer(height, width, input, output);
    try checkFinite(output);
    runtime.probe_io.placement(&session, shards);
    try runtime.probe_io.write(init.io, args[5], output);
}

fn openDenoiserSession(init: std.process.Init, device: runtime.Backend, low_memory: bool) !runtime.Session {
    if (!low_memory) return runtime.Session.open(init.gpa, init.io, device);
    if (builtin.os.tag != .linux or device != .cuda) return error.LowMemoryRequiresLinuxCuda;
    const session = try runtime.Session.openWithOptions(init.gpa, init.io, device, .{
        .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.98 } } },
    });
    std.log.info("cuda_allocator=bfc", .{});
    return session;
}

fn denoise(comptime profile: diffusion.Profile, init: std.process.Init, args: []const [:0]const u8, low_memory: bool) !void {
    const Model = diffusion.Model(profile);
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) return writeSchema(init, &Model.schema);
    const trace = args.len == 9 and std.mem.eql(u8, args[8], "--reference-trajectory");
    const blocks = args.len == 9 and std.mem.eql(u8, args[8], "--reference-blocks");
    const feedback_sweep = args.len == 9 and std.mem.eql(u8, args[8], "--reference-feedback");
    const feedback = feedback_sweep or (args.len == 9 and std.mem.eql(u8, args[8], "--reference-feedback-endpoints"));
    const prefix = args.len == 9 and std.mem.startsWith(u8, args[8], "--reference-prefix=");
    if (args.len != 8 and !trace and !blocks and !feedback and !prefix) return error.ExpectedBackendWeightsTargetTotalTextStepsInputOutput;
    if (low_memory and (blocks or prefix)) return error.BlockTraceRequiresWholeGraph;
    const device = try backend(args[0]);
    const geometry: diffusion.Geometry = .{
        .target = try std.fmt.parseInt(usize, args[2], 10),
        .image = try std.fmt.parseInt(usize, args[3], 10),
        .text = try std.fmt.parseInt(usize, args[4], 10),
    };
    if (!geometry.valid()) return error.InvalidGeometry;
    const steps = try std.fmt.parseInt(usize, args[5], 10);
    if (steps > 60 or (trace and steps == 0) or ((blocks or prefix) and steps != 0) or (feedback and steps != 2)) return error.InvalidStepCount;
    const split = if (prefix) try std.fmt.parseInt(usize, args[8]["--reference-prefix=".len..], 10) else 0;
    if (prefix and (split == 0 or split > Model.blocks)) return error.InvalidReferencePrefix;
    const input_count = Model.inputCount(geometry);
    const target_count = geometry.target * diffusion.channels;
    const input_end = input_count + steps;
    const reference_count = if (trace or feedback) (steps - 1) * target_count else 0;
    const total = if (blocks or prefix) Model.traceInputCount(geometry) else input_end + reference_count;
    const values = try runtime.probe_io.read(init.gpa, init.io, args[6], total);
    defer init.gpa.free(values);
    try checkFinite(values);
    const times = values[input_count - 1 .. input_end];
    for (times, 0..) |time, i| {
        if (time < 0 or time > 1 or (i > 0 and time >= times[i - 1])) return error.InvalidSchedule;
    }
    for (values[input_count - 1 - geometry.text .. input_count - 1]) |mask| {
        if (mask != 0 and mask != 1) return error.InvalidTextMask;
    }
    if (blocks or prefix) {
        const output = try init.gpa.alloc(f32, Model.traceOutputCount(geometry));
        defer init.gpa.free(output);
        const session = try runtime.Session.open(init.gpa, init.io, device);
        defer session.close();
        var instance = try Model.loadTrace(&session, args[1], geometry);
        defer instance.deinit();
        if (prefix) {
            values[input_count] = @floatFromInt(split + 2);
            try predictVelocity(&instance, &session, values, output);
            return runtime.probe_io.write(init.io, args[7], output[0..target_count]);
        }
        values[input_count] = 0;
        try predictVelocity(&instance, &session, values, output);
        try runtime.probe_io.write(init.io, args[7], output);
        const forced = try std.mem.concat(init.gpa, u8, &.{ args[7], ".forced" });
        defer init.gpa.free(forced);
        values[input_count] = 1;
        try predictVelocity(&instance, &session, values, output);
        try runtime.probe_io.write(init.io, forced, output);
        return;
    }
    const fractions: []const f32 = if (feedback_sweep) &.{ 0.25, 0.5, 0.75 } else &.{};
    const output_rows = if (feedback) 5 + 2 * fractions.len else if (trace) steps + 1 else 1;
    const output = try init.gpa.alloc(f32, target_count * output_rows);
    defer init.gpa.free(output);
    const current = output[0..target_count];
    const target = values[0..target_count];
    const session = try openDenoiserSession(init, device, low_memory);
    defer session.close();
    var math_library: ?model.cuda_math.Library = null;
    defer if (math_library) |*library| library.close();
    if (low_memory and profile == .edit_2511) {
        if (std.c.getenv("ZML_QWEN_CUDA_MATH")) |path| {
            math_library = try model.cuda_math.Library.open(session.platform, std.mem.span(path));
            std.log.info("qwen_cuda_math=reference", .{});
        }
    }
    var instance = if (low_memory)
        try Model.loadBlockwise(&session, args[1], geometry)
    else
        try Model.load(&session, args[1], geometry);
    defer instance.deinit();
    for (0..@max(steps, 1)) |i| {
        const time = times[i];
        values[input_count - 1] = time;
        if (feedback and i == 1) @memcpy(output[2 * target_count ..][0..target_count], target);
        try predictVelocity(&instance, &session, values[0..input_count], current);
        if ((trace or feedback) and i == 0) @memcpy(output[target_count..][0..target_count], current);
        if (feedback and i == 1) @memcpy(output[3 * target_count ..][0..target_count], current);
        if (steps != 0) {
            const dt = times[i + 1] - time;
            for (target, current) |*latent, velocity| latent.* += dt * velocity;
        }
    }
    if (steps != 0) @memcpy(current, target);
    if (trace) {
        for (1..steps) |i| {
            const reference_start = input_end + (i - 1) * target_count;
            @memcpy(target, values[reference_start..][0..target_count]);
            values[input_count - 1] = times[i];
            const velocity = output[(i + 1) * target_count ..][0..target_count];
            try predictVelocity(&instance, &session, values[0..input_count], velocity);
        }
    }
    if (feedback) {
        const reference = values[input_end..];
        const fed_back = output[2 * target_count ..][0..target_count];
        @memcpy(target, reference);
        values[input_count - 1] = times[1];
        try predictVelocity(&instance, &session, values[0..input_count], output[4 * target_count ..][0..target_count]);
        for (fractions, 0..) |fraction, i| {
            const sample = output[(5 + 2 * i) * target_count ..][0..target_count];
            for (sample, reference, fed_back) |*value, from, to| value.* = from + fraction * (to - from);
            @memcpy(target, sample);
            const velocity = output[(6 + 2 * i) * target_count ..][0..target_count];
            try predictVelocity(&instance, &session, values[0..input_count], velocity);
        }
    }
    try checkFinite(output);
    try runtime.probe_io.write(init.io, args[7], output);
}

fn projectQK(init: std.process.Init, args: []const [:0]const u8, split: bool) !void {
    const Probe = diffusion.ProjectionProbe;
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) return writeSchema(init, &Probe.schema);
    if (args.len != 6) return error.ExpectedBackendWeightsImageTextInputOutput;
    const g: diffusion.Geometry = .{
        .target = 1,
        .image = try std.fmt.parseInt(usize, args[2], 10),
        .text = try std.fmt.parseInt(usize, args[3], 10),
    };
    if (!g.valid()) return error.InvalidGeometry;
    const input = try runtime.probe_io.read(init.gpa, init.io, args[4], Probe.inputCount(g));
    defer init.gpa.free(input);
    try checkFinite(input);
    const output = try init.gpa.alloc(f32, Probe.outputCount(g));
    defer init.gpa.free(output);
    const session = try runtime.Session.open(init.gpa, init.io, try backend(args[0]));
    defer session.close();
    const shards = try Probe.run(&session, args[1], g, split, input, output);
    try checkFinite(output);
    runtime.probe_io.placement(&session, shards);
    try runtime.probe_io.write(init.io, args[5], output);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.ExpectedProfile;
    const low_memory = std.mem.eql(u8, args[1], "--cuda-low-memory");
    const command = args[1 + @as(usize, @intFromBool(low_memory)) ..];
    if (command.len == 0) return error.ExpectedProfile;
    if (!low_memory) {
        if (std.mem.eql(u8, command[0], "vae_encode")) return autoencode(.encode, init, command[1..]);
        if (std.mem.eql(u8, command[0], "vae_decode")) return autoencode(.decode, init, command[1..]);
        if (std.mem.eql(u8, command[0], "project_qk")) return projectQK(init, command[1..], false);
        if (std.mem.eql(u8, command[0], "project_qk_split")) return projectQK(init, command[1..], true);
    }
    const profile = std.meta.stringToEnum(diffusion.Profile, command[0]) orelse return error.InvalidProfile;
    switch (profile) {
        inline else => |p| return denoise(p, init, command[1..], low_memory),
    }
}
