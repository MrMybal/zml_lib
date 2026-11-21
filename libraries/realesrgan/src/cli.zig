const std = @import("std");
const builtin = @import("builtin");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const model_lib = @import("zml_lib/realesrgan");
const image = @import("image.zig");

pub const std_options: std.Options = .{ .log_level = .info };

extern fn zml_lib_cli_reserve_stdout() isize;

const Request = struct {
    model: model_lib.Variant,
    weights_path: []const u8,
    image_path: []const u8,
    output_dir: []const u8,
    backend: runtime.Backend = .auto,
    trace: bool = false,

    fn string(value: std.json.Value) ![]const u8 {
        return if (value == .string) value.string else error.InvalidRequestField;
    }

    fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Request {
        const value = try std.json.parseFromSliceLeaky(std.json.Value, allocator, bytes, .{});
        if (value != .object) return error.InvalidRequest;
        const fields = value.object;
        const model = try string(fields.get("model") orelse return error.MissingModel);
        const backend = if (fields.get("backend")) |entry| try string(entry) else "auto";
        const trace = fields.get("trace");
        if (trace) |entry| if (entry != .bool) return error.InvalidRequestField;
        for (fields.keys()) |key| {
            if (std.meta.stringToEnum(std.meta.FieldEnum(Request), key) == null)
                return error.UnknownRequestField;
        }
        return .{
            .model = std.meta.stringToEnum(model_lib.Variant, model) orelse return error.InvalidVariant,
            .backend = std.meta.stringToEnum(runtime.Backend, backend) orelse return error.InvalidBackend,
            .weights_path = try string(fields.get("weights_path") orelse return error.MissingWeightsPath),
            .image_path = try string(fields.get("image_path") orelse return error.MissingImagePath),
            .output_dir = try string(fields.get("output_dir") orelse return error.MissingOutputDir),
            .trace = if (trace) |entry| entry.bool else false,
        };
    }
};

const Response = struct {
    ok: bool = true,
    output_path: []const u8,
    width: u32,
    height: u32,
    backend: []const u8,
    pjrt_platform: []const u8,
};

const TraceFiles = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,

    fn write(context: *anyopaque, name: []const u8, buffer: zml.Buffer) !void {
        const self: *TraceFiles = @ptrCast(@alignCast(context));
        const slice = try buffer.toSliceAlloc(self.allocator, self.io);
        defer slice.free(self.allocator);
        try self.writeTensor(name, slice.constData());
    }

    fn writeTensor(self: TraceFiles, name: []const u8, bytes: []const u8) !void {
        var path: [64]u8 = undefined;
        const filename = try std.fmt.bufPrint(&path, "{s}.f32le", .{name});
        try self.dir.writeFile(self.io, .{ .sub_path = filename, .data = bytes, .flags = .{ .exclusive = true } });
    }
};

fn validPath(path: []const u8) bool {
    return path.len != 0 and std.unicode.utf8ValidateSlice(path) and std.mem.indexOfScalar(u8, path, 0) == null;
}

fn handle(allocator: std.mem.Allocator, io: std.Io) !Response {
    var input_buffer: [4096]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(io, &input_buffer);
    const bytes = try stdin.interface.allocRemaining(allocator, .limited(64 * 1024));
    const request = try Request.parse(allocator, bytes);
    if (!validPath(request.weights_path) or !validPath(request.image_path) or !validPath(request.output_dir))
        return error.InvalidPath;

    const source = try image.read(allocator, io, request.image_path);
    defer allocator.free(source.pixels);
    const size = try source.size.scaled(request.model.scale());
    const pixels = try allocator.alloc(f32, try size.elements());
    defer allocator.free(pixels);

    const cwd = std.Io.Dir.cwd();
    const directory = try cwd.createDirPathOpen(io, request.output_dir, .{});
    defer directory.close(io);
    const filename = if (request.model.scale() == 2) "realesrgan_x2.png" else "realesrgan_x4.png";
    const output_path = try std.fs.path.join(allocator, &.{ request.output_dir, filename });

    const session = try runtime.Session.open(allocator, io, request.backend);
    defer session.close();
    const pjrt_platform = try allocator.dupe(u8, session.pjrtPlatform());
    var model = try model_lib.Model.load(&session, request.model, request.weights_path);
    defer model.deinit();
    var trace_files: TraceFiles = .{ .allocator = allocator, .io = io, .dir = directory };
    const trace: ?model_lib.TraceWriter = if (request.trace) .{ .context = &trace_files, .write = TraceFiles.write } else null;
    _ = try model.infer(source.size, source.pixels, pixels, trace);
    if (request.trace) {
        try trace_files.writeTensor("input", std.mem.sliceAsBytes(source.pixels));
        try trace_files.writeTensor("output", std.mem.sliceAsBytes(pixels));
    }

    var atomic = try directory.createFileAtomic(io, filename, .{});
    defer atomic.deinit(io);
    var write_buffer: [16 * 1024]u8 = undefined;
    var output = atomic.file.writer(io, &write_buffer);
    try image.writePng(allocator, size, pixels, &output.interface);
    try output.interface.flush();
    try atomic.link(io);
    return .{
        .output_path = output_path,
        .width = size.width,
        .height = size.height,
        .backend = session.target(),
        .pjrt_platform = pjrt_platform,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const reserved = zml_lib_cli_reserve_stdout();
    if (reserved == -1) return error.ProtocolOutputUnavailable;
    const protocol: std.Io.File = .{
        .handle = if (builtin.os.tag == .windows) @ptrFromInt(@as(usize, @bitCast(reserved))) else @intCast(reserved),
        .flags = .{ .nonblocking = false },
    };
    defer protocol.close(init.io);
    var buffer: [4096]u8 = undefined;
    var stdout = protocol.writerStreaming(init.io, &buffer);
    const response = handle(allocator, init.io) catch |err| {
        std.log.err("Real-ESRGAN: {s}", .{@errorName(err)});
        try std.json.Stringify.value(.{ .ok = false, .@"error" = @errorName(err) }, .{}, &stdout.interface);
        try stdout.interface.writeByte('\n');
        try stdout.interface.flush();
        std.process.exit(1);
    };
    try std.json.Stringify.value(response, .{}, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}
