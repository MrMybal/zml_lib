//! Binary FP32 interface for SD 1.5 component validation and image generation.
const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const unet = @import("unet.zig");
const ddim = @import("ddim.zig");
const vae_scaling: f32 = 0.18215;
const vae = runtime.kl_decoder.Decoder(.{ .channels = unet.channels, .scaling = vae_scaling, .post_quant = true });
const encoder = runtime.kl_decoder.Encoder(unet.channels, true, true);

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

fn encode(init: std.process.Init, session: *const runtime.Session, path: []const u8, height: usize, width: usize, rest: []const [:0]const u8) !void {
    const Encode = enum { masked, both };
    const mode = std.meta.stringToEnum(Encode, rest[0]) orelse return error.InvalidEncodeMode;
    const images: usize = if (mode == .both) 2 else 1;
    const count = height * width * unet.channels;
    const image_area = height * width * encoder.factor * encoder.factor;
    const packed_len = 4 * image_area;
    const input = try read(init, rest[1], packed_len + images * count);
    defer init.gpa.free(input);
    const moments = try init.gpa.alloc(f32, 2 * count);
    defer init.gpa.free(moments);
    const output = try init.gpa.alloc(f32, images * count);
    defer init.gpa.free(output);
    var model = try runtime.TensorModel.load(session, &encoder.schema, path);
    defer model.deinit();

    for (0..images) |index| {
        // The second pass encodes the original image with the same compiled graph.
        if (index == 1) @memset(input[3 * image_area .. packed_len], 0);
        runtime.probe_io.placement(session, try encoder.encodeMoments(
            &model,
            @intCast(height * encoder.factor),
            @intCast(width * encoder.factor),
            input[0..packed_len],
            moments,
        ));
        try checkFinite(moments);
        const noise = input[packed_len + index * count ..][0..count];
        for (output[index * count ..][0..count], moments[0..count], moments[count..], noise) |*value, mean, log_var, gaussian| {
            value.* = (mean + @exp(0.5 * std.math.clamp(log_var, -30, 20)) * gaussian) * vae_scaling;
        }
    }
    try checkFinite(output);
    try runtime.probe_io.write(init.io, rest[2], output);
}

fn denoise(init: std.process.Init, session: *const runtime.Session, path: []const u8, height: usize, width: usize, rest: []const [:0]const u8, sample: bool, inpaint: bool) !void {
    const area = height * width;
    const count = area * unet.channels;
    const embedding_len = unet.context_length * unet.context_width;
    const step = try std.fmt.parseInt(usize, rest[0], 10);
    const guidance = if (sample) try std.fmt.parseFloat(f32, rest[1]) else 1;
    if (!std.math.isFinite(guidance) or guidance < 1 or guidance > 20) return error.InvalidGuidance;
    const schedule = if (sample) try ddim.Schedule.init(step) else null;
    if (!sample and step >= 1000) return error.InvalidTimestep;
    const start = if (inpaint and sample) try std.fmt.parseInt(usize, rest[2], 10) else 0;
    if (sample and start >= step) return error.InvalidDDIMStart;
    const context_count = if (inpaint) area * unet.image_context_channels else 0;
    const text_offset = count + context_count + if (start > 0) count else @as(usize, 0);
    const input = try read(init, rest[rest.len - 2], text_offset + embedding_len * (1 + @as(usize, @intFromBool(guidance > 1))));
    defer init.gpa.free(input);
    const latent = input[0..count];
    const epsilon = try init.gpa.alloc(f32, count);
    defer init.gpa.free(epsilon);
    var model = try unet.Model.load(session, path, @intCast(height), @intCast(width), inpaint);
    defer model.deinit();
    var image_context = if (inpaint) try model.imageCondition(input[count..][0..context_count]) else null;
    defer if (image_context) |*buffer| buffer.deinit();
    var positive = try model.condition(input[text_offset..][0..embedding_len]);
    defer positive.deinit();
    var negative: ?zml.Buffer = if (guidance > 1) try model.condition(input[text_offset + embedding_len ..]) else null;
    defer if (negative) |*buffer| buffer.deinit();
    const unconditioned = try init.gpa.alloc(f32, if (negative != null) count else 0);
    defer init.gpa.free(unconditioned);

    if (schedule) |plan| {
        if (start > 0) plan.addNoise(start, input[text_offset - count .. text_offset], latent);
        for (start..step) |i| {
            const t = plan.timestep(i);
            runtime.probe_io.placement(session, try model.predict(t, latent, positive, image_context, epsilon));
            if (negative) |condition| {
                runtime.probe_io.placement(session, try model.predict(t, latent, condition, image_context, unconditioned));
                for (epsilon, unconditioned) |*cond, none| cond.* = none + guidance * (cond.* - none);
            }
            try checkFinite(epsilon);
            plan.step(i, latent, epsilon);
            try checkFinite(latent);
        }
    } else {
        runtime.probe_io.placement(session, try model.predict(step, latent, positive, image_context, epsilon));
        try checkFinite(epsilon);
    }
    try runtime.probe_io.write(init.io, rest[rest.len - 1], if (sample) latent else epsilon);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedMode;
    const Mode = enum { unet, sample, vae, inpaint_unet, inpaint_sample, vae_encode };
    const mode = std.meta.stringToEnum(Mode, args[1]) orelse return error.InvalidMode;
    if (args.len == 3 and std.mem.eql(u8, args[2], "schema"))
        return schema(init, switch (mode) {
            .vae => &vae.schema,
            .vae_encode => &encoder.schema,
            .unet, .sample => &unet.schema,
            .inpaint_unet, .inpaint_sample => &unet.inpaint_schema,
        });
    const expected: usize = switch (mode) {
        .unet, .inpaint_unet, .vae_encode => 9,
        .sample => 10,
        .inpaint_sample => 11,
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
        .vae_encode => return encode(init, &session, args[3], height, width, args[6..]),
        .unet, .sample, .inpaint_unet, .inpaint_sample => return denoise(
            init,
            &session,
            args[3],
            height,
            width,
            args[6..],
            mode == .sample or mode == .inpaint_sample,
            mode == .inpaint_unet or mode == .inpaint_sample,
        ),
    }
}
