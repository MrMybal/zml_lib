const std = @import("std");
const runtime = @import("zml_lib/runtime");
const tts = @import("zml_lib/qwen3_tts");
const audio_encoder = @import("zml_lib/qwen3_tts/audio_encoder");
const speaker = @import("zml_lib/qwen3_tts/speaker");
const waveform = @import("zml_lib/qwen3_tts/waveform");

pub const std_options: std.Options = .{ .log_level = .info };

fn infer(comptime Graph: type, session: *const runtime.Session, model: *runtime.TensorModel, count: usize, input: []const f32, output: []f32) !void {
    try Graph.check(input, count);
    const shards = try model.infer(
        Graph.forward,
        .init(.{ count, Graph.input_width }, .f32),
        .init(.{ Graph.outputRows(count), Graph.output_width }, .f32),
        input,
        output,
    );
    for (output) |value| if (!std.math.isFinite(value)) return error.NonfiniteOutput;
    runtime.probe_io.placement(session, shards);
}

fn run(comptime Graph: type, comptime storage: runtime.checkpoint.Storage, comptime limit: usize, init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) {
        var buffer: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (Graph.schema) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    const streaming = args.len == 4 and std.mem.eql(u8, args[3], "stdin");
    if (!streaming and (args.len < 5 or (args.len - 3) % 2 != 0))
        return error.ExpectedBackendWeightsRowCountAndInputOutputPairs;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const count = try std.fmt.parseInt(usize, args[2], 10);
    if (count == 0 or count > limit) return error.InvalidTokenCount;

    const allocator = init.gpa;
    const output_rows = Graph.outputRows(count);
    const output = try allocator.alloc(f32, output_rows * Graph.output_width);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try runtime.TensorModel.loadCheckpoint(&session, &Graph.schema, args[1], storage);
    defer model.deinit();
    if (streaming) {
        const input = try allocator.alloc(f32, count * Graph.input_width);
        defer allocator.free(input);
        var read_buffer: [8192]u8 = undefined;
        var write_buffer: [8192]u8 = undefined;
        var reader = std.Io.File.stdin().readerStreaming(init.io, &read_buffer);
        var writer = std.Io.File.stdout().writerStreaming(init.io, &write_buffer);
        while (true) {
            const command = reader.interface.takeByte() catch |err| switch (err) {
                error.EndOfStream => return,
                else => return err,
            };
            if (command != 0) return error.InvalidVoiceRequest;
            try reader.interface.readSliceAll(std.mem.sliceAsBytes(input));
            try infer(Graph, &session, &model, count, input, output);
            try writer.interface.writeAll(std.mem.sliceAsBytes(output));
            try writer.interface.flush();
        }
    }
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        const input = try runtime.probe_io.read(allocator, init.io, args[i], count * Graph.input_width);
        defer allocator.free(input);
        try infer(Graph, &session, &model, count, input, output);
        try runtime.probe_io.write(init.io, args[i + 1], output);
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 3 and std.mem.eql(u8, args[1], "waveform"))
        return run(waveform, .f32, waveform.max_frames, init, args[2..]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "audio_encoder"))
        return run(audio_encoder, .f32, audio_encoder.max_samples, init, args[2..]);
    if (args.len < 4) return error.ExpectedProfileAndStage;
    const profile = std.meta.stringToEnum(tts.Profile, args[1]) orelse return error.InvalidProfile;
    if (std.mem.eql(u8, args[2], "speaker")) return switch (profile) {
        .qwen3_tts_0b6 => run(speaker.Graph(1024), .f32, speaker.max_frames, init, args[3..]),
        .qwen3_tts_1b7 => run(speaker.Graph(2048), .f32, speaker.max_frames, init, args[3..]),
        .fixture => error.InvalidProfile,
    };
    const stage = std.meta.stringToEnum(tts.Stage, args[2]) orelse return error.InvalidStage;
    switch (profile) {
        inline else => |p| switch (stage) {
            inline else => |s| return run(tts.Graph(p, s), .f16, 1024, init, args[3..]),
        },
    }
}
