const std = @import("std");
const models = @import("zml_llm_models");
const runtime = @import("zml_lib/runtime");
const zml = @import("zml");

pub const Profile = enum { llama, qwen35 };

pub fn run(comptime profile: Profile, init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.log.err("Usage: {s} cpu|cuda|rocm|vulkan MODEL_DIR SEQUENCE_LENGTH PROMPT", .{args[0]});
        return error.InvalidArguments;
    }
    const backend = std.meta.stringToEnum(runtime.Backend, args[1]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const limit = try std.fmt.parseInt(u32, args[3], 10);
    if (limit < 2) return error.InvalidSequenceLength;
    const dir = if (std.fs.path.isAbsolute(args[2]))
        try std.Io.Dir.openDirAbsolute(init.io, args[2], .{ .iterate = true })
    else
        try std.Io.Dir.cwd().openDir(init.io, args[2], .{ .iterate = true });
    defer dir.close(init.io);
    const kind = try models.detectModelType(allocator, init.io, dir);
    const supported = switch (profile) {
        .llama => kind == .llama,
        .qwen35 => kind == .qwen3_5 or kind == .qwen3_5_moe,
    };
    if (!supported) return error.UnsupportedModelType;

    var registry = try zml.safetensors.TensorRegistry.fromRepo(allocator, init.io, dir);
    defer registry.deinit();
    const backend_session = try runtime.Session.open(allocator, init.io, backend);
    defer backend_session.close();
    const platform = backend_session.platform;
    const meshes: models.Meshes = try .init(platform);
    var store = zml.io.TensorStore.fromRegistry(allocator, &registry, &meshes.all());
    defer store.deinit();
    const generation: models.GenerationOptions = .{ .sampling_strategy = .{ .topk = 1 } };
    var model = try models.LoadedModel.load(allocator, init.io, dir, store.view(), generation);
    defer model.deinit(allocator);
    var tokenizer = try loadTokenizer(allocator, init.io, dir);
    defer tokenizer.deinit();
    var progress = std.Progress.start(init.io, .{ .root_name = "Model" });
    errdefer progress.end();
    const compiled = try allocator.create(models.CompiledModel);
    defer allocator.destroy(compiled);
    compiled.* = try model.compile(allocator, init.io, platform, .{ .stablehlo = {} }, meshes, limit, &progress);
    defer compiled.deinit();
    var buffers = try model.loadBuffers(allocator, init.io, platform, &store, &progress);
    defer model.unloadBuffers(&buffers, allocator);
    progress.end();
    progress = .none;

    var session = try compiled.newSession(allocator, init.io, platform, &buffers, tokenizer);
    defer session.deinit();
    const prompt = try session.tokenizePrompt(allocator, args[4]);
    defer allocator.free(prompt);
    if (prompt.len == 0 or prompt.len >= limit) return error.InvalidPromptLength;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(allocator);
    try tokens.appendSlice(allocator, prompt);
    std.log.info("backend={s} pjrt={s} device={s}", .{ backend_session.target(), backend_session.pjrtPlatform(), platform.devices[0].debugString() });
    try session.runPrefill(tokens.items);
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    try session.runDecode(&tokens, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);
    return .fromBytes(allocator, bytes);
}
