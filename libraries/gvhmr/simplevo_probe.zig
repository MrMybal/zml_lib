const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const solve = @import("vo_pose.zig");
const trajectory = @import("vo_trajectory.zig");

pub const std_options: std.Options = .{ .log_level = .info };

fn source(session: *const rt.Session, shape: zml.Shape, values: []const f32) !zml.Buffer {
    return zml.Buffer.fromBytesOpts(session.io, session.platform, shape, std.mem.sliceAsBytes(values), .{ .wait = true });
}

fn readResult(session: *const rt.Session, graph: *rt.Compiled, shape: zml.Shape, out: []f32) !void {
    var result = graph.results.get(zml.Buffer);
    defer result.deinit();
    if (!result.shape().eql(shape)) return error.UnexpectedOutputShape;
    rt.probe_io.placement(session, try session.verifyPlacement(&result));
    try result.toSlice(session.io, .init(shape, std.mem.sliceAsBytes(out)));
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 9) return error.ExpectedBackendFramesPairsMaxMatchesMatchesPlanOutputRepeats;
    const backend = std.meta.stringToEnum(rt.Backend, args[1]) orelse return error.InvalidBackend;
    const frames = try std.fmt.parseInt(usize, args[2], 10);
    const pairs = try std.fmt.parseInt(usize, args[3], 10);
    const matches = try std.fmt.parseInt(usize, args[4], 10);
    const repeats = try std.fmt.parseInt(u16, args[8], 10);
    if (frames < 2 or frames > 4096 or pairs == 0 or pairs >= frames or matches < 10 or matches > 8192 or repeats == 0)
        return error.InvalidOdometryShape;

    const allocator = std.heap.page_allocator;
    const match_shape: zml.Shape = .init(.{ matches, 5 }, .f32);
    const sample_shape: zml.Shape = .init(.{ solve.homography_trials, solve.sample_size, 4 }, .f32);
    const cutoff_shape: zml.Shape = .init(.{ 1, 4 }, .f32);
    const pair_shape: zml.Shape = .init(.{ 1, solve.output_width }, .f32);
    const deltas_shape: zml.Shape = .init(.{ pairs + 1, trajectory.pose_width }, .f32);
    const plan_shape: zml.Shape = .init(.{ frames, 3 }, .f32);
    const output_shape: zml.Shape = .init(.{ frames, trajectory.output_width }, .f32);
    const match_len = match_shape.count();
    const sample_end = match_len + sample_shape.count();
    const pair_width = sample_end + cutoff_shape.count();

    const data = try rt.probe_io.read(allocator, init.io, args[5], pairs * pair_width);
    defer allocator.free(data);
    const plan = try rt.probe_io.read(allocator, init.io, args[6], plan_shape.count());
    defer allocator.free(plan);
    const output = try allocator.alloc(f32, pairs * solve.output_width + output_shape.count());
    defer allocator.free(output);
    const deltas = try allocator.alloc(f32, deltas_shape.count());
    defer allocator.free(deltas);
    @memset(deltas[0..trajectory.pose_width], 0);
    deltas[0] = 1;
    deltas[4] = 1;
    deltas[8] = 1;

    const session = try rt.Session.open(allocator, init.io, backend);
    defer session.close();
    var two_view = try rt.Compiled.init(&session, solve.forward, .{
        zml.Tensor.fromShape(match_shape), zml.Tensor.fromShape(sample_shape), zml.Tensor.fromShape(cutoff_shape),
    });
    defer two_view.deinit(allocator);
    var accumulate = try rt.Compiled.init(&session, trajectory.forward, .{ zml.Tensor.fromShape(deltas_shape), zml.Tensor.fromShape(plan_shape) });
    defer accumulate.deinit(allocator);
    var schedule = try source(&session, plan_shape, plan);
    defer schedule.deinit();

    for (0..repeats) |_| {
        for (0..pairs) |pair| {
            const values = data[pair * pair_width ..][0..pair_width];
            var points = try source(&session, match_shape, values[0..match_len]);
            defer points.deinit();
            var hypotheses = try source(&session, sample_shape, values[match_len..sample_end]);
            defer hypotheses.deinit();
            var cutoff = try source(&session, cutoff_shape, values[sample_end..]);
            defer cutoff.deinit();
            try two_view.run(.{ points, hypotheses, cutoff });
            const out = output[pair * solve.output_width ..][0..solve.output_width];
            try readResult(&session, &two_view, pair_shape, out);
            if (out[13] == 0) {
                std.log.err("no two-view geometry for pair {d}, with {d} inliers", .{ pair + 1, out[12] });
                return error.UnreliableRelativePose;
            }
            @memcpy(deltas[(pair + 1) * trajectory.pose_width ..][0..trajectory.pose_width], out[0..trajectory.pose_width]);
        }
        var relative = try source(&session, deltas_shape, deltas);
        defer relative.deinit();
        try accumulate.run(.{ relative, schedule });
        try readResult(&session, &accumulate, output_shape, output[pairs * solve.output_width ..]);
    }
    try rt.probe_io.write(init.io, args[7], output);
}
