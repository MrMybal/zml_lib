const std = @import("std");
const runtime = @import("zml_lib/runtime");
const whisper = @import("zml_lib/whisper");
const zml = @import("zml/core");

pub fn run(comptime profile: whisper.Profile, init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len != 4) return error.ExpectedBackendEncoderWeightsDecoderWeightsAndCapacity;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const Encoder = whisper.Graph(profile, .encoder);
    const Decoder = whisper.Graph(profile, .decoder);
    const cfg = Decoder.config;
    const capacity = try std.fmt.parseInt(usize, args[3], 10);
    if (capacity < 1 or capacity > cfg.target) return error.InvalidTokenCapacity;

    const io = init.io;
    const allocator = init.gpa;
    const session = try runtime.Session.open(allocator, io, backend);
    defer session.close();
    var encoder = try runtime.TensorModel.load(&session, &Encoder.schema, args[1]);
    defer encoder.deinit();
    var decoder = try runtime.TensorModel.load(&session, &Decoder.schema, args[2]);
    defer decoder.deinit();

    const mel_shape: zml.Shape = .init(.{2 * cfg.mels * cfg.source}, .f32);
    const context_shape: zml.Shape = .init(.{ 1, cfg.source, cfg.width }, .f32);
    const cross_shape: zml.Shape = .init(.{ cfg.decoder_layers * 2, cfg.heads, cfg.source, @divExact(cfg.width, @as(usize, @intCast(cfg.heads))) }, .f32);
    const token_shape: zml.Shape = .init(.{capacity}, .i32);
    const index_shape: zml.Shape = .init(.{1}, .i32);
    const logit_shape: zml.Shape = .init(.{ 1, cfg.vocab }, .f32);
    var cross = try runtime.Compiled.init(&session, Decoder.projectContext, .{
        decoder.parameters.tensors, zml.Tensor.fromShape(context_shape),
    });
    defer cross.deinit(allocator);
    cross.args.bake(decoder.parameters.buffers);
    var scorer = try runtime.Compiled.init(&session, Decoder.forwardAt, .{
        decoder.parameters.tensors,
        zml.Tensor.fromShape(cross_shape),
        zml.Tensor.fromShape(token_shape),
        zml.Tensor.fromShape(index_shape),
    });
    defer scorer.deinit(allocator);
    scorer.args.bake(decoder.parameters.buffers);

    const mels = try allocator.alloc(f32, mel_shape.count());
    defer allocator.free(mels);
    const tokens = try allocator.alloc(u32, capacity);
    defer allocator.free(tokens);
    const logits = try allocator.alloc(f32, cfg.vocab);
    defer allocator.free(logits);
    var context_cache: ?zml.Buffer = null;
    defer if (context_cache) |*previous| previous.deinit();
    var read_buffer: [8192]u8 = undefined;
    var write_buffer: [8192]u8 = undefined;
    var input = std.Io.File.stdin().readerStreaming(io, &read_buffer);
    var output = std.Io.File.stdout().writerStreaming(io, &write_buffer);

    while (true) {
        const request = input.interface.takeByte() catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        switch (request) {
            0 => {
                try input.interface.readSliceAll(std.mem.sliceAsBytes(mels));
                for (mels) |value| if (!std.math.isFinite(value)) return error.InvalidLogMel;
                var encoded = try encoder.inferDevice(Encoder.forward, mel_shape, std.mem.sliceAsBytes(mels));
                defer encoded.deinit();
                if (!encoded.shape().eql(context_shape)) return error.InvalidEncoderOutput;
                runtime.probe_io.placement(&session, try session.verifyPlacement(&encoded));
                try cross.run(.{encoded});
                var replacement = cross.results.get(zml.Buffer);
                errdefer replacement.deinit();
                if (!replacement.shape().eql(cross_shape)) return error.InvalidCrossAttentionCache;
                runtime.probe_io.placement(&session, try session.verifyPlacement(&replacement));
                try output.interface.writeByte(0);
                if (context_cache) |*previous| previous.deinit();
                context_cache = replacement;
            },
            1 => {
                const cache = context_cache orelse return error.MissingWhisperContext;
                var index: u32 = undefined;
                try input.interface.readSliceAll(std.mem.asBytes(&index));
                if (index >= capacity) return error.InvalidTokenPosition;
                try input.interface.readSliceAll(std.mem.sliceAsBytes(tokens));
                for (tokens) |token| if (token >= cfg.vocab) return error.InvalidTokenId;
                var prefix = try zml.Buffer.fromBytesOpts(io, session.platform, token_shape, std.mem.sliceAsBytes(tokens), .{ .wait = true });
                defer prefix.deinit();
                var position = try zml.Buffer.fromBytesOpts(io, session.platform, index_shape, std.mem.asBytes(&index), .{ .wait = true });
                defer position.deinit();
                try scorer.run(.{ cache, prefix, position });
                var result = scorer.results.get(zml.Buffer);
                defer result.deinit();
                if (!result.shape().eql(logit_shape)) return error.InvalidLogitShape;
                runtime.probe_io.placement(&session, try session.verifyPlacement(&result));
                try result.toSlice(io, .init(logit_shape, std.mem.sliceAsBytes(logits)));
                try output.interface.writeAll(std.mem.sliceAsBytes(logits));
            },
            else => return error.InvalidWhisperRequest,
        }
        try output.interface.flush();
    }
}
