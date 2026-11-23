const std = @import("std");
const runtime = @import("zml_lib/runtime");
const zml = @import("zml_lib/whisper");

pub const std_options: std.Options = .{ .log_level = .info };

fn run(comptime profile: zml.Profile, comptime stage: zml.Stage, init: std.process.Init, args: []const [:0]const u8) !void {
    const Graph = zml.Graph(profile, stage);
    const cfg = Graph.config;
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) {
        var buffer: [4096]u8 = undefined;
        var output = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (Graph.schema) |parameter| {
            try output.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try output.interface.print("\t{d}", .{dim});
            try output.interface.writeByte('\n');
        }
        return output.interface.flush();
    }
    if (args.len != (if (stage == .encoder) @as(usize, 4) else 5))
        return error.ExpectedBackendWeightsInputOutputAndDecoderTokenCount;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    const count = if (stage == .encoder) 0 else try std.fmt.parseInt(usize, args[4], 10);
    if (stage == .decoder and (count == 0 or count > cfg.target)) return error.InvalidTokenCount;
    const context_cells = cfg.width * cfg.source;
    const count_in = if (stage == .encoder) 2 * cfg.mels * cfg.source else context_cells + count;
    const count_out = if (stage == .encoder) context_cells else cfg.vocab;
    const allocator = init.gpa;
    const input = try runtime.probe_io.read(allocator, init.io, args[2], count_in);
    defer allocator.free(input);
    if (stage == .decoder) {
        for (input[context_cells..]) |id| {
            if (!std.math.isFinite(id) or id < 0 or @trunc(id) != id or id >= @as(f32, @floatFromInt(cfg.vocab)))
                return error.InvalidTokenId;
        }
    }
    const output = try allocator.alloc(f32, count_out);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try runtime.TensorModel.load(&session, &Graph.schema, args[1]);
    defer model.deinit();
    const shapes = if (stage == .encoder) .{ 1, cfg.source, cfg.width } else .{ 1, cfg.vocab };
    runtime.probe_io.placement(&session, try model.infer(Graph.forward, .init(.{count_in}, .f32), .init(shapes, .f32), input, output));
    try runtime.probe_io.write(init.io, args[3], output);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedProfileAndStage;
    const profile = std.meta.stringToEnum(zml.Profile, args[1]) orelse return error.InvalidProfile;
    if (std.mem.eql(u8, args[2], "transcribe")) {
        switch (profile) {
            inline else => |p| return @import("worker.zig").run(p, init, args[3..]),
        }
    }
    const stage = std.meta.stringToEnum(zml.Stage, args[2]) orelse return error.InvalidStage;
    switch (profile) {
        inline else => |p| switch (stage) {
            inline else => |s| return run(p, s, init, args[3..]),
        },
    }
}
