const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const model = @import("zml_lib/moge2");

pub const std_options: std.Options = .{ .log_level = .info };

fn matches(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn run(comptime graph: type, init: std.process.Init, args: []const [:0]const u8) !void {
    if (matches(args[1], "schema") and args.len == 3) {
        var buffer: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (graph.parameters) |parameter| {
            try writer.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try writer.interface.print("\t{d}", .{dim});
            try writer.interface.writeByte('\n');
        }
        return writer.interface.flush();
    }
    if (!matches(args[1], "infer") or args.len < 12 or @mod(args.len - 10, 2) != 0)
        return error.ExpectedBackendWeightsImageSizeTokensAndInputOutputPairs;
    const backend = std.meta.stringToEnum(rt.Backend, args[3]) orelse return error.InvalidBackend;
    if (backend == .auto) return error.ExplicitBackendRequired;
    const h = try std.fmt.parseInt(i64, args[5], 10);
    const w = try std.fmt.parseInt(i64, args[6], 10);
    const rows = try std.fmt.parseInt(i64, args[7], 10);
    const cols = try std.fmt.parseInt(i64, args[8], 10);
    const repeats = try std.fmt.parseInt(u16, args[9], 10);
    if (h < 1 or h > 4096 or w < 1 or w > 4096 or rows < 1 or rows > 256 or cols < 1 or cols > 256 or rows * cols > 4096 or repeats == 0)
        return error.InvalidGraphDimensions;

    const session = try rt.Session.open(init.gpa, init.io, backend);
    defer session.close();
    var weights = try rt.TensorModel.load(&session, &graph.parameters, args[4]);
    defer weights.deinit();
    const in_shape: zml.Shape = .init(.{ 1, 3, h, w }, .f32);
    const out_shape: zml.Shape = .init(.{graph.elements(h, w)}, .f32);
    var exe = try rt.Compiled.init(&session, graph.forward, .{
        weights.parameters.tensors, zml.Tensor.fromShape(in_shape), model.Grid{ .rows = rows, .cols = cols },
    });
    defer exe.deinit(init.gpa);
    exe.args.bake(weights.parameters.buffers);

    const values = try init.gpa.alloc(f32, out_shape.count());
    defer init.gpa.free(values);
    var index: usize = 10;
    while (index < args.len) : (index += 2) {
        const data = try rt.probe_io.read(init.gpa, init.io, args[index], in_shape.count());
        defer init.gpa.free(data);
        for (data) |value| if (!std.math.isFinite(value)) return error.NonfiniteInput;
        var input = try zml.Buffer.fromBytesOpts(init.io, session.platform, in_shape, std.mem.sliceAsBytes(data), .{ .wait = true });
        defer input.deinit();
        for (0..repeats) |_| {
            try exe.run(.{input});
            var out = exe.results.get(zml.Buffer);
            defer out.deinit();
            if (!out.shape().eql(out_shape)) return error.UnexpectedOutputShape;
            const shards = try session.verifyPlacement(&out);
            try out.toSlice(init.io, .init(out_shape, std.mem.sliceAsBytes(values)));
            rt.probe_io.placement(&session, shards);
        }
        for (values) |value| if (!std.math.isFinite(value)) return error.NonfiniteOutput;
        try rt.probe_io.write(init.io, args[index + 1], values);
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedOperationAndProfile;
    if (matches(args[2], "camera")) return run(model.Graph(false, false), init, args);
    if (matches(args[2], "normal")) return run(model.Graph(true, true), init, args);
    if (matches(args[2], "normal-full")) return run(model.Graph(true, false), init, args);
    return error.InvalidProfile;
}
