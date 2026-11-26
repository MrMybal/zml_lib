const std = @import("std");
const runtime = @import("zml_lib/runtime");
const qwen = @import("zml_lib/qwen25_vl");
pub const std_options: std.Options = .{ .log_level = .info };

const Layout = struct { fields: usize, limit: usize, output: usize, group: usize = 1 };

fn run(comptime Model: type, comptime schema: anytype, layout: Layout, init: std.process.Init, args: []const [:0]const u8) !void {
    if (args.len == 1 and std.mem.eql(u8, args[0], "schema")) {
        var scratch: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &scratch);
        for (schema) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (args.len < 5 or (args.len - 3) % 2 != 0)
        return error.ExpectedBackendWeightsTokenCountInputOutputPairs;
    const backend = std.meta.stringToEnum(runtime.Backend, args[0]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const count = try std.fmt.parseInt(usize, args[2], 10);
    if (count == 0 or count > layout.limit or count % layout.group != 0) return error.InvalidInputCount;
    const allocator = init.gpa;
    const output = try allocator.alloc(f32, count / layout.group * layout.output);
    defer allocator.free(output);
    const session = try runtime.Session.open(allocator, init.io, backend);
    defer session.close();
    var model = try Model.load(&session, args[1]);
    defer model.deinit();
    var i: usize = 3;
    while (i < args.len) : (i += 2) {
        const input = try runtime.probe_io.read(allocator, init.io, args[i], count * layout.fields);
        defer allocator.free(input);
        const shards = try model.encode(count, input, output);
        for (output) |value| if (!std.math.isFinite(value)) return error.NonfiniteOutput;
        runtime.probe_io.placement(&session, shards);
        try runtime.probe_io.write(init.io, args[i + 1], output);
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.ExpectedProfile;
    if (std.mem.eql(u8, args[1], "vision")) {
        const vision = qwen.vision;
        return run(vision.Encoder, vision.schema, .{
            .fields = vision.input_width,
            .limit = vision.max_patches,
            .output = vision.output_width,
            .group = vision.merge_size,
        }, init, args[2..]);
    }
    const profile = std.meta.stringToEnum(qwen.Profile, args[1]) orelse return error.InvalidProfile;
    switch (profile) {
        inline else => |value| {
            const Model = qwen.Encoder(value);
            return run(Model, Model.schema, .{
                .fields = Model.input_width,
                .limit = qwen.max_tokens,
                .output = Model.width,
            }, init, args[2..]);
        },
    }
}
