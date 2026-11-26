const std = @import("std");
const runtime = @import("zml_lib/runtime");
const encoder = @import("zml_lib/phonon/encoder");
const tdt = @import("zml_lib/phonon");

pub const std_options: std.Options = .{ .log_level = .info };

const Mode = enum { encode, @"encode-live", decode, transcribe };
const Event = struct { token: usize, frame: usize, duration: usize };

fn encodeWithModel(session: *const runtime.Session, model: *encoder.Encoder, frames: usize, input: []const f32, output: []f32, padded_input: ?[]f32) !void {
    const shards = if (padded_input) |buffer| try model.encodeLive(frames, input, buffer, output) else try model.encode(frames, input, output);
    for (output) |value| if (!std.math.isFinite(value)) return error.NonfiniteEncoderOutput;
    runtime.probe_io.placement(session, shards);
}

fn encode(session: *const runtime.Session, weights: []const u8, frames: usize, input: []const f32, live: bool) ![]f32 {
    var model = try encoder.Encoder.load(session, weights);
    defer model.deinit();
    const padded_input = if (live) try session.allocator.alloc(f32, encoder.live_frames * (encoder.mels + 1)) else null;
    defer if (padded_input) |buffer| session.allocator.free(buffer);
    const output_frames = encoder.outputFrames(if (live) encoder.live_frames else frames);
    const output = try session.allocator.alloc(f32, output_frames * encoder.width);
    errdefer session.allocator.free(output);
    try encodeWithModel(session, &model, frames, input, output, padded_input);
    return output;
}

fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |value, i| {
        if (value > values[best]) best = i;
    }
    return best;
}

fn decodeWithModel(session: *const runtime.Session, model: *tdt.Decoder, features: []const f32, writer: *std.Io.Writer) !void {
    const frames = features.len / tdt.encoder_width;
    const input = try session.allocator.alloc(f32, tdt.input_count);
    defer session.allocator.free(input);
    const output = try session.allocator.alloc(f32, tdt.output_count);
    defer session.allocator.free(output);
    const states = input[1 + tdt.encoder_width ..];
    input[0] = tdt.blank_token;
    @memset(states, 0);
    var events: std.ArrayList(Event) = .empty;
    defer events.deinit(session.allocator);
    var frame: usize = 0;
    var shards: u32 = 0;
    const duration_end = tdt.token_count + tdt.duration_count;
    while (frame < frames) {
        if (events.items.len >= 10 * frames) return error.DecoderDidNotAdvance;
        @memcpy(input[1..][0..tdt.encoder_width], features[frame * tdt.encoder_width ..][0..tdt.encoder_width]);
        shards = try model.step(input, output);
        for (output) |value| if (!std.math.isFinite(value)) return error.NonfiniteDecoderOutput;
        const token = argmax(output[0..tdt.token_count]);
        var duration = argmax(output[tdt.token_count..duration_end]);
        if (token == tdt.blank_token) {
            duration = @max(1, duration);
        } else {
            input[0] = @floatFromInt(token);
            @memcpy(states, output[duration_end..]);
        }
        try events.append(session.allocator, .{ .token = token, .frame = frame, .duration = duration });
        frame += duration;
    }
    runtime.probe_io.placement(session, shards);
    try std.json.Stringify.value(events.items, .{}, writer);
    try writer.writeByte('\n');
    try writer.flush();
}

fn decode(session: *const runtime.Session, weights: []const u8, features: []const f32, path: []const u8) !void {
    var model = try tdt.Decoder.load(session, weights);
    defer model.deinit();
    const directory = std.Io.Dir.cwd();
    const file = try directory.createFile(session.io, path, .{ .exclusive = true });
    errdefer directory.deleteFile(session.io, path) catch {};
    defer file.close(session.io);
    var buffer: [8192]u8 = undefined;
    var writer = file.writerStreaming(session.io, &buffer);
    return decodeWithModel(session, &model, features, &writer.interface);
}

fn stream(init: std.process.Init, args: []const [:0]const u8, live: bool) !void {
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const allocator = init.gpa;
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var enc = try encoder.Encoder.load(&session, args[1]);
    defer enc.deinit();
    var dec = try tdt.Decoder.load(&session, args[2]);
    defer dec.deinit();
    std.log.info("phonon_stream_models_loaded", .{});
    const limit: usize = if (live) encoder.live_frames else encoder.max_frames;
    const input = try allocator.alloc(f32, limit * encoder.mels);
    defer allocator.free(input);
    const features = try allocator.alloc(f32, encoder.outputFrames(limit) * encoder.width);
    defer allocator.free(features);
    const padded_input = if (live) try allocator.alloc(f32, encoder.live_frames * (encoder.mels + 1)) else null;
    defer if (padded_input) |buffer| allocator.free(buffer);

    var read_buffer: [8192]u8 = undefined;
    var write_buffer: [8192]u8 = undefined;
    var reader = std.Io.File.stdin().readerStreaming(init.io, &read_buffer);
    var writer = std.Io.File.stdout().writerStreaming(init.io, &write_buffer);
    while (true) {
        var header: [4]u8 = undefined;
        header[0] = reader.interface.takeByte() catch |err| switch (err) {
            error.EndOfStream => return,
            else => return err,
        };
        try reader.interface.readSliceAll(header[1..]);
        const frames: usize = std.mem.readInt(u32, &header, .little);
        if (frames == 0 or frames > limit) return error.InvalidMelFrameCount;
        const mels = input[0 .. frames * encoder.mels];
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(mels));
        const encoded = features[0 .. encoder.outputFrames(frames) * encoder.width];
        try encodeWithModel(&session, &enc, frames, mels, if (live) features else encoded, padded_input);
        try decodeWithModel(&session, &dec, encoded, &writer.interface);
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 5 and std.mem.eql(u8, args[1], "stream")) return stream(init, args[2..], false);
    if (args.len == 5 and std.mem.eql(u8, args[1], "stream-live")) return stream(init, args[2..], true);
    if (args.len == 2 and std.mem.eql(u8, args[1], "schema")) {
        var buffer: [8192]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (encoder.schema) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (args.len < 7) return error.ExpectedModeBackendWeightsFrameCountInputAndOutput;
    const mode = std.meta.stringToEnum(Mode, args[1]) orelse return error.InvalidOperation;
    const offset: usize = if (mode == .transcribe) 1 else 0;
    if (args.len != 7 + offset) return error.ExpectedModeBackendWeightsFrameCountInputAndOutput;
    const backend = std.meta.stringToEnum(runtime.Backend, args[2]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const frames = try std.fmt.parseInt(usize, args[4 + offset], 10);
    const limit: usize = switch (mode) {
        .decode => encoder.outputFrames(encoder.max_frames),
        .@"encode-live" => encoder.live_frames,
        else => encoder.max_frames,
    };
    if (frames == 0 or frames > limit) return error.InvalidFrameCount;
    const columns: usize = if (mode == .decode) tdt.encoder_width else encoder.mels;
    const input = try runtime.probe_io.read(init.gpa, init.io, args[5 + offset], frames * columns);
    defer init.gpa.free(input);
    const output = args[6 + offset];
    const session = try runtime.Session.open(init.gpa, init.io, backend);
    defer session.close();
    if (mode == .decode) return decode(&session, args[3], input, output);
    const features = try encode(&session, args[3], frames, input, mode == .@"encode-live");
    defer init.gpa.free(features);
    if (mode == .encode or mode == .@"encode-live") {
        return runtime.probe_io.write(init.io, output, features[0 .. encoder.outputFrames(frames) * encoder.width]);
    }
    return decode(&session, args[4], features, output);
}
