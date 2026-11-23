const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const flow = @import("zml_lib/trellis/flow");
const raw = rt.probe_io;
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) return error.ExpectedOperationProfileAndStage;
    const sparse = std.mem.eql(u8, args[3], "slat");
    if (!sparse and !std.mem.eql(u8, args[3], "structure")) return error.InvalidFlowStage;
    if (std.mem.eql(u8, args[2], "official")) {
        return if (sparse) launch(flow.official, true, init, args) else launch(flow.official, false, init, args);
    }
    if (std.mem.eql(u8, args[2], "fixture")) {
        return if (sparse) launch(flow.fixture, true, init, args) else launch(flow.fixture, false, init, args);
    }
    if (std.mem.eql(u8, args[2], "pipeline-fixture")) {
        return if (sparse) launch(flow.pipeline_fixture, true, init, args) else launch(flow.pipeline_fixture, false, init, args);
    }
    return error.InvalidProfile;
}

fn launch(comptime p: flow.Profile, comptime sparse: bool, init: std.process.Init, args: []const [:0]const u8) !void {
    const graph = flow.Network(p, sparse);
    if (args.len == 4 and std.mem.eql(u8, args[1], "schema")) {
        var buffer: [4096]u8 = undefined;
        var stdout = std.Io.File.stdout().writerStreaming(init.io, &buffer);
        for (graph.parameters) |parameter| {
            try stdout.interface.writeAll(parameter.name);
            for (parameter.shape.dims()) |dim| try stdout.interface.print("\t{d}", .{dim});
            try stdout.interface.writeByte('\n');
        }
        return stdout.interface.flush();
    }
    if (args.len != 18 or !std.mem.eql(u8, args[1], "sample")) return error.InvalidFlowRequest;
    const backend = std.meta.stringToEnum(rt.Backend, args[4]) orelse return error.InvalidBackend;
    const rows = try std.fmt.parseInt(i64, args[6], 10);
    const conditions = try std.fmt.parseInt(i64, args[7], 10);
    const groups = try std.fmt.parseInt(i64, args[8], 10);
    const steps = try std.fmt.parseInt(u32, args[9], 10);
    const cfg = try std.fmt.parseFloat(f32, args[10]);
    const rescale = try std.fmt.parseFloat(f64, args[11]);
    if (conditions < 1 or conditions > 8192 or steps == 0 or steps > 1000 or
        !std.math.isFinite(cfg) or cfg < 0 or !std.math.isFinite(rescale) or rescale <= 0)
        return error.InvalidFlowSettings;
    if (sparse) {
        if (rows < 1 or rows > 64 * 64 * 64 or groups < 1 or groups > 32 * 32 * 32 or groups > rows)
            return error.InvalidSparseGeometry;
    } else if (rows != p.grid * p.grid * p.grid or groups != 1) return error.InvalidStructureShape;

    const allocator = std.heap.page_allocator;
    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    const state_shape: zml.Shape = .init(.{ rows, flow.latent_width }, .f32);
    const cond_shape: zml.Shape = .init(.{ conditions, p.context }, .f32);
    const scalar: zml.Shape = .init(.{1}, .f32);
    const fine_shape: zml.Shape = .init(.{ if (sparse) rows else 1, if (sparse) flow.fine_width else 1 }, .f32);
    const coarse_shape: zml.Shape = .init(.{ groups, if (sparse) flow.coarse_width else 1 }, .f32);
    const state = try raw.read(allocator, init.io, args[12], state_shape.count());
    defer allocator.free(state);
    const predicted = try allocator.alloc(f32, state.len);
    defer allocator.free(predicted);
    const negative = try allocator.alloc(f32, state.len);
    defer allocator.free(negative);

    var context = try readBuffer(&session, cond_shape, args[13]);
    defer context.deinit();
    var negative_context = try readBuffer(&session, cond_shape, args[14]);
    defer negative_context.deinit();
    var fine = try readBuffer(&session, fine_shape, args[15]);
    defer fine.deinit();
    var coarse = try readBuffer(&session, coarse_shape, args[16]);
    defer coarse.deinit();
    const parameters = try rt.checkpoint.load(allocator, init.io, session.platform, graph.parameters.len, @as([]const rt.layers.Parameter, &graph.parameters), rt.layers.Parameter.at, args[5]);
    defer parameters.deinit(allocator);
    var compiled = try rt.Compiled.init(&session, graph.forward, .{
        parameters.tensors,           zml.Tensor.fromShape(state_shape), zml.Tensor.fromShape(cond_shape),
        zml.Tensor.fromShape(scalar), zml.Tensor.fromShape(fine_shape),  zml.Tensor.fromShape(coarse_shape),
    });
    defer compiled.deinit(allocator);
    compiled.args.bake(parameters.buffers);
    for (0..@as(usize, steps)) |i| {
        const now = schedule(i, steps, rescale);
        const next = schedule(i + 1, steps, rescale);
        var sample = try upload(&session, state_shape, state);
        defer sample.deinit();
        var time = try upload(&session, scalar, &.{@floatCast(1000 * now)});
        defer time.deinit();
        try velocity(&session, &compiled, .{ sample, context, time, fine, coarse }, state_shape, predicted);
        if (cfg != 0 and now >= 0.5) {
            try velocity(&session, &compiled, .{ sample, negative_context, time, fine, coarse }, state_shape, negative);
            for (predicted, negative) |*yes, no| yes.* = (1 + cfg) * yes.* - cfg * no;
        }
        const delta: f32 = @floatCast(now - next);
        for (state, predicted) |*x, value| x.* -= delta * value;
    }
    try raw.write(init.io, args[17], state);
}

fn schedule(step: usize, steps: u32, factor: f64) f64 {
    const fraction = 1 - @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(steps));
    return factor * fraction / (1 + (factor - 1) * fraction);
}

fn upload(session: *const rt.Session, shape: zml.Shape, data: []const f32) !zml.Buffer {
    return zml.Buffer.fromBytesOpts(session.io, session.platform, shape, std.mem.sliceAsBytes(data), .{ .wait = true });
}

fn readBuffer(session: *const rt.Session, shape: zml.Shape, path: []const u8) !zml.Buffer {
    const bytes = try raw.read(session.allocator, session.io, path, shape.count());
    defer session.allocator.free(bytes);
    return upload(session, shape, bytes);
}

fn velocity(session: *const rt.Session, compiled: *rt.Compiled, inputs: anytype, shape: zml.Shape, result: []f32) !void {
    try compiled.run(inputs);
    var output = compiled.results.get(zml.Buffer);
    defer output.deinit();
    if (!output.shape().eql(shape)) return error.UnexpectedVelocityShape;
    raw.placement(session, try session.verifyPlacement(&output));
    try output.toSlice(session.io, .init(shape, std.mem.sliceAsBytes(result)));
}
