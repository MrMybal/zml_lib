const std = @import("std");
const runtime = @import("zml_lib/runtime");
const flux = @import("zml_lib/flux1");
const vae = @import("vae.zig");
pub const std_options: std.Options = .{ .log_level = .info };

fn writeSchema(init: std.process.Init, schema: []const runtime.layers.Parameter) !void {
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    for (schema) |parameter| {
        try writer.interface.writeAll(parameter.name);
        for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
        try writer.interface.writeByte('\n');
    }
    return writer.interface.flush();
}

fn autoencode(comptime encode: bool, init: std.process.Init, args: []const [:0]const u8) !void {
    const schema: []const runtime.layers.Parameter = if (encode) &vae.Encoder.schema else &vae.schema;
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) return writeSchema(init, schema);
    if (args.len != 6) return error.ExpectedBackendWeightsLatentHeightLatentWidthInputOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const height = try std.fmt.parseInt(usize, args[2], 10);
    const width = try std.fmt.parseInt(usize, args[3], 10);
    if (height == 0 or width == 0 or height > 128 or width > 128) return error.InvalidDimensions;
    const area = height * width;
    const rgb_count = area * 3 * vae.factor * vae.factor;
    const input = try runtime.probe_io.read(init.gpa, init.io, args[4], if (encode) rgb_count else area * vae.channels);
    defer init.gpa.free(input);
    for (input) |value| if (!std.math.isFinite(value)) return error.InvalidInputValues;
    const output = try init.gpa.alloc(f32, if (encode) area * 2 * vae.channels else rgb_count);
    defer init.gpa.free(output);
    const session = try runtime.Session.open(init.gpa, init.io, backend);
    defer session.close();
    var model = try runtime.TensorModel.load(&session, schema, args[1]);
    defer model.deinit();
    const shards = if (encode)
        try vae.Encoder.encodeMoments(&model, @intCast(height * vae.factor), @intCast(width * vae.factor), input, output)
    else
        try vae.decode(&model, @intCast(height), @intCast(width), input, output);
    runtime.probe_io.placement(&session, shards);
    return runtime.probe_io.write(init.io, args[5], output);
}

fn modelInput(noise: []const f32, control: []const f32, combined: []f32) []const f32 {
    const channels = flux.image_channels;
    for (0..noise.len / channels) |i| {
        @memcpy(combined[i * 2 * channels ..][0..channels], noise[i * channels ..][0..channels]);
        @memcpy(combined[i * 2 * channels + channels ..][0..channels], control[i * channels ..][0..channels]);
    }
    return combined;
}

fn run(comptime profile: flux.Profile, comptime storage: flux.Storage, init: std.process.Init, args: []const [:0]const u8) !void {
    const Model = flux.Model(profile, storage);
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) return writeSchema(init, &Model.schema);
    if (args.len != 8 and args.len != 10) return error.ExpectedBackendWeightsRowsColsTextTokensStepsInputOutputAndOptionalLoRA;
    const adapters: ?flux.LoRAInput = if (args.len == 10) .{ .manifest = args[8], .weights = args[9] } else null;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const rows = try std.fmt.parseInt(usize, args[2], 10);
    const cols = try std.fmt.parseInt(usize, args[3], 10);
    const tokens = try std.fmt.parseInt(usize, args[4], 10);
    const steps = try std.fmt.parseInt(usize, args[5], 10);
    if (rows < 1 or rows > 64 or cols < 1 or cols > 64 or tokens < 1 or tokens > 512 or steps > 50)
        return error.InvalidInputDimensions;
    const image_values = rows * cols * flux.image_channels;
    const text_end = image_values + tokens * @as(usize, @intCast(Model.text_channels));
    const pooled_end = text_end + flux.pooled_channels;
    const times_end = pooled_end + steps + 1;
    const input = try runtime.probe_io.read(init.gpa, init.io, args[6], times_end +
        (if (Model.has_control) image_values + 1 else 0));
    defer init.gpa.free(input);
    for (input) |value| if (!std.math.isFinite(value)) return error.InvalidInputValues;
    const image = input[0..image_values];
    const times = input[pooled_end..times_end];
    const control = if (Model.has_control) input[times_end..][0..image_values] else &.{};
    const guidance = if (Model.has_control) input[times_end + image_values] else 0;
    const combined = try init.gpa.alloc(f32, if (Model.has_control) 2 * image_values else 0);
    defer init.gpa.free(combined);
    for (times, 0..) |time, i| {
        if (!std.math.isFinite(time) or time < 0 or time > 1 or (i != 0 and time >= times[i - 1]))
            return error.InvalidTimesteps;
    }
    const output = try init.gpa.alloc(f32, image_values);
    defer init.gpa.free(output);
    const session = try runtime.Session.open(init.gpa, init.io, backend);
    defer session.close();
    var model = try Model.load(&session, args[1], @intCast(rows), @intCast(cols), @intCast(tokens), input[image_values..text_end], input[text_end..pooled_end], adapters, guidance);
    defer model.deinit();
    if (steps == 0) {
        const values = if (Model.has_control) modelInput(image, control, combined) else image;
        runtime.probe_io.placement(&session, try model.predict(values, times[0], output));
    } else {
        for (0..steps) |i| {
            const values = if (Model.has_control) modelInput(image, control, combined) else image;
            runtime.probe_io.placement(&session, try model.predict(values, times[i], output));
            const delta = times[i + 1] - times[i];
            for (image, output) |*value, velocity| value.* += delta * velocity;
        }
    }
    try runtime.probe_io.write(init.io, args[7], if (steps == 0) output else image);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 2 and std.mem.eql(u8, args[1], "vae")) return autoencode(false, init, args[2..]);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "vae_encode")) return autoencode(true, init, args[2..]);
    if (args.len < 4) return error.ExpectedProfileAndStorage;
    const profile = std.meta.stringToEnum(flux.Profile, args[1]) orelse return error.InvalidProfile;
    const storage = std.meta.stringToEnum(flux.Storage, args[2]) orelse return error.InvalidStorage;
    switch (profile) {
        inline else => |p| switch (storage) {
            inline else => |s| return run(p, s, init, args[3..]),
        },
    }
}
