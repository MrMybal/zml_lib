const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const m = @import("post_math.zig");
const Tensor = zml.Tensor;

pub const frames_max = 4096;
pub const Mode = enum { moving, fixed };
const joint_count = 22;
const pose_width = (joint_count - 1) * 3;
const parents = [_]usize{ 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19 };
const contact_joints = [_]usize{ 7, 10, 8, 11, 20, 21 };
const translation_start = 10 + 3 + pose_width;
const contact_start = translation_start + 3;
const camera_rotation_start = contact_start + contact_joints.len;
const camera_translation_start = camera_rotation_start + 3;
pub const input_width = camera_translation_start + 3;
pub const output_width = 3 + pose_width;

const schema = [_]rt.layers.Parameter{
    .{ .name = "J_template", .shape = .init(.{joint_count * 3}, .f32) },
    .{ .name = "J_shapedirs", .shape = .init(.{ 10, joint_count * 3 }, .f32) },
};

const Joint = struct {
    rotation: Tensor,
    position: Tensor,

    fn child(parent: Joint, local: Joint) Joint {
        return .{
            .rotation = m.normalizeColumns(m.product(parent.rotation, local.rotation)),
            .position = parent.position.add(m.apply(parent.rotation, local.position)),
        };
    }
};

fn forwardKinematics(local: [joint_count]Joint) [joint_count]Joint {
    var world: [joint_count]Joint = undefined;
    world[0] = local[0];
    for (1..joint_count) |j| world[j] = world[parents[j]].child(local[j]);
    return world;
}

fn contacts(world: [joint_count]Joint) Tensor {
    var points: [contact_joints.len]Tensor = undefined;
    for (&points, contact_joints) |*point, j| point.* = world[j].position;
    return Tensor.stack(&points, 1, .contact);
}

fn fromFirst(x: Tensor, distance: i64) Tensor {
    return x.slice(0, .{ .end = x.dim(0) - distance });
}

fn differences(x: Tensor) Tensor {
    return x.slice(0, .{ .start = 1 }).sub(fromFirst(x, 1));
}

fn prepend(x: Tensor, first: Tensor) Tensor {
    return Tensor.concatenate(&.{ first, x }, 0);
}

fn ground(world: [joint_count]Joint, original: Tensor, corrected: Tensor) Tensor {
    var heights: [joint_count]Tensor = undefined;
    for (&heights, world) |*height, joint| height.* = m.component(joint.position, 1);
    const dy = m.component(corrected, 1).sub(m.component(original, 1));
    const ys = m.join(&heights);
    const min_y = ys.add(dy.broad(ys.shape())).reshape(.{ys.count()}).min(0).reshape(.{});
    return m.join(&.{ m.component(corrected, 0), m.component(corrected, 1).sub(min_y), m.component(corrected, 2) });
}

fn correctMoving(world: [joint_count]Joint, original: Tensor, logits: Tensor) Tensor {
    const frames = original.dim(0);
    var corrected = original;
    if (frames > 1) {
        const active = fromFirst(logits, 1);
        const mask = active.cmp(.GT, m.scalar(0));
        const is_static = mask.convert(.f32);
        const masked = active.sub(is_static.negate().addConstant(@as(f32, 1)).scale(@as(f32, 1e6)));
        const present = is_static.sum(1).cmp(.GT, m.scalar(0)).convert(.f32);
        const all = differences(contacts(world));
        const weights = masked.softmax(1).mul(present.broad(active.shape())).broadcast(all.shape(), &.{ 0, 1 });
        const contact_delta = all.mul(weights).sum(1).reshape(.{ frames - 1, 3 });
        corrected = m.prefixSum(prepend(differences(original).sub(contact_delta), original.slice(0, .{ .end = 1 })));
    }
    const horizontal = m.smooth(m.join(&.{ m.component(corrected, 0), m.component(corrected, 2) }), 3);
    return ground(world, original, m.join(&.{ m.component(horizontal, 0), m.component(corrected, 1), m.component(horizontal, 1) }));
}

const StaticRoot = struct {
    differences: Tensor,

    pub const State = struct { step: Tensor, total: Tensor, corrections: Tensor };

    pub fn cond(self: StaticRoot, state: State) Tensor {
        return state.step.cmp(.LT, Tensor.scalar(self.differences.dim(0), .i32));
    }

    pub fn body(self: StaticRoot, state: State) State {
        const difference = self.differences.slice(0, .dyn(state.step, 1)).sub(state.total);
        const needed = difference.abs().cmp(.GE, m.scalar(0.25));
        const next = state.total.add(needed.select(difference.clamp(m.scalar(-0.02), m.scalar(0.02)), m.scalar(0)));
        return .{
            .step = state.step.addConstant(@as(i32, 1)),
            .total = next,
            .corrections = state.corrections.dynamicUpdateSlice1d(next, 0, state.step),
        };
    }
};

fn correctFixed(world: [joint_count]Joint, original: Tensor, logits: Tensor, camera: Tensor, raw_root_rotation: Tensor, rest_root: Tensor) Tensor {
    const frames = original.dim(0);
    if (frames == 1) return ground(world, original, original);
    const camera_rotation = m.fromAxisAngle(camera.slice(1, .{ .end = 3 }).slice(0, .{ .end = 1 }));
    const world_rotation = raw_root_rotation.slice(0, .{ .end = 1 });
    const c2w = m.product(world_rotation, camera_rotation.transpose(.{ 0, 2, 1 }));
    const camera_root = rest_root.add(m.smooth(camera.slice(1, .{ .start = 3 }), 5));
    const rotated = m.apply(c2w.broadcast(world[0].rotation.shape(), &.{ 0, 1, 2 }), camera_root);
    const offset = world[0].position.slice(0, .{ .end = 1 }).sub(rotated.slice(0, .{ .end = 1 }));
    const root = StaticRoot{ .differences = world[0].position.sub(rotated.add(offset.broad(rotated.shape()))) };
    const drift = zml.ops.@"while"(StaticRoot, root, .{
        .step = .scalar(1, .i32),
        .total = m.scalar(0).broad(offset.shape()),
        .corrections = m.scalar(0).broad(original.shape()),
    }).corrections;

    const points = contacts(world);
    const displacement = differences(points.sub(drift.broadcast(points.shape(), &.{ 0, 2 })));
    const active = fromFirst(logits, 1).sigmoid().cmp(.GT, m.scalar(0.8)).convert(.f32);
    const count = active.sum(1).maximum(m.scalar(1));
    const mean = displacement.mul(active.broadcast(displacement.shape(), &.{ 0, 1 })).sum(1)
        .reshape(.{ frames - 1, 3 }).div(count.broad(.init(.{ frames - 1, 3 }, .f32)));
    const horizontal = m.join(&.{ m.component(mean, 0), m.scalar(0).broad(count.shape()), m.component(mean, 2) });
    const correction = m.prefixSum(prepend(horizontal, m.scalar(0).broad(offset.shape())));
    return ground(world, original, original.sub(drift).sub(correction));
}

fn contactTargets(world: [joint_count]Joint, logits: Tensor) Tensor {
    const position = contacts(world);
    const frames = position.dim(0);
    if (frames == 1) return position;
    const confidence = fromFirst(logits, 1).sigmoid().reshape(.{ frames - 1, contact_joints.len, 1 });
    var a = prepend(confidence, m.scalar(0).broad(confidence.shape().set(0, 1)));
    const points = position.slice(0, .{ .start = 1 });
    const mixed = points.mul(confidence.negate().addConstant(@as(f32, 1)).broad(points.shape()));
    var b = prepend(mixed, position.slice(0, .{ .end = 1 }));
    // Compose affine recurrence steps in parallel, from t-1 toward t.
    var stride: i64 = 1;
    while (stride < frames) : (stride *= 2) {
        b = b.add(a.broad(b.shape()).mul(m.shift(b, stride, 0)));
        a = a.mul(m.shift(a, stride, 1));
    }
    return b;
}

const Target = struct { index: usize, slot: i64 };

fn solveChain(local: *[joint_count]Joint, world: [joint_count]Joint, targets: Tensor, chain: [5]usize, goal: []const Target) void {
    var current: [chain.len]Joint = undefined;
    current[0] = world[chain[0]];
    for (1..chain.len) |j| current[j] = current[j - 1].child(local[chain[j]]);
    for (0..2) |_| {
        for (1..chain.len - 1) |j| {
            const pivot = current[j];
            var x = m.scalar(0).broad(pivot.position.shape());
            var y = x;
            var count: f32 = 0;
            for (goal) |tip| {
                if (tip.index <= j) continue;
                const desired = targets.slice(1, .single(tip.slot));
                const weight = @as(f32, @floatFromInt(j + 1)) / chain.len;
                const aligned = m.turnTowards(pivot.rotation, current[tip.index].position.sub(pivot.position), desired.sub(pivot.position), weight);
                x = x.add(m.column(aligned, 0));
                y = y.add(m.column(aligned, 1));
                count += 1;
            }
            const x_axis = m.normalize(x.scale(1 / count), 1e-9);
            const y_axis = m.normalize(y.scale(1 / count), 1e-9);
            const solved = Tensor.stack(&.{ x_axis, y_axis, m.cross(x_axis, y_axis) }, .last, .column);
            // CCD averages the x and y axes independently. The reference
            // renormalizes columns at each local/world transform composition.
            local[chain[j]].rotation = m.normalizeColumns(m.product(m.inverseNormalized(current[j - 1].rotation), solved));
            for (j..chain.len) |k| current[k] = current[k - 1].child(local[chain[k]]);
        }
    }
}

fn forward(mode: Mode, parameters: []const Tensor, input: Tensor) Tensor {
    const frames = input.dim(0);
    const offsets = input.slice(1, .{ .end = 10 }).dotGeneral(parameters[1], &.{.{ 1, 0 }}, &.{});
    const rest = offsets.add(parameters[0].broadcast(offsets.shape(), &.{1})).reshape(.{ frames, joint_count, 3 });
    const original = input.slice(1, .{ .start = translation_start, .end = contact_start });
    const logits = input.slice(1, .{ .start = contact_start, .end = camera_rotation_start });
    const angles = input.slice(1, .{ .start = 10, .end = translation_start }).reshape(.{ frames * joint_count, 3 });
    const rotation = m.fromAxisAngle(angles).reshape(.{ frames, joint_count, 3, 3 });
    const rest_root = rest.slice(1, .single(0));
    var local: [joint_count]Joint = undefined;
    const raw_root_rotation = rotation.slice(1, .single(0));
    local[0] = .{ .rotation = m.normalizeColumns(raw_root_rotation), .position = rest_root.add(original) };
    for (1..joint_count) |j| {
        local[j] = .{
            .rotation = m.normalizeColumns(rotation.slice(1, .single(@intCast(j)))),
            .position = rest.slice(1, .single(@intCast(j))).sub(rest.slice(1, .single(@intCast(parents[j])))),
        };
    }
    const initial_world = forwardKinematics(local);
    const corrected = switch (mode) {
        .moving => correctMoving(initial_world, original, logits),
        .fixed => correctFixed(initial_world, original, logits, input.slice(1, .{ .start = camera_rotation_start }), raw_root_rotation, rest_root),
    };
    local[0].position = rest_root.add(corrected);
    const world = forwardKinematics(local);
    const targets = contactTargets(world, logits);
    solveChain(&local, world, targets, .{ 0, 1, 4, 7, 10 }, &.{ .{ .index = 3, .slot = 0 }, .{ .index = 4, .slot = 1 } });
    solveChain(&local, world, targets, .{ 0, 2, 5, 8, 11 }, &.{ .{ .index = 3, .slot = 2 }, .{ .index = 4, .slot = 3 } });
    solveChain(&local, world, targets, .{ 9, 13, 16, 18, 20 }, &.{.{ .index = 4, .slot = 4 }});
    solveChain(&local, world, targets, .{ 9, 14, 17, 19, 21 }, &.{.{ .index = 4, .slot = 5 }});
    var rotations: [joint_count - 1]Tensor = undefined;
    const matrix_shape = zml.Shape.init(.{ frames, 3, 3 }, .f32);
    for (&rotations, local[1..]) |*r, j| r.* = j.rotation.withTags(matrix_shape);
    const matrices = Tensor.stack(&rotations, 1, .joint).reshape(.{ frames * (joint_count - 1), 3, 3 });
    return m.join(&.{ corrected, m.toAxisAngle(matrices).reshape(.{ frames, pose_width }) });
}

fn moving(params: []const Tensor, input: Tensor) Tensor {
    return forward(.moving, params, input);
}

fn fixed(params: []const Tensor, input: Tensor) Tensor {
    return forward(.fixed, params, input);
}

pub const Model = struct {
    graph: rt.TensorModel,
    mode: Mode,

    pub fn load(session: *const rt.Session, path: []const u8, mode: Mode) !Model {
        return .{ .graph = try .load(session, &schema, path), .mode = mode };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn predict(self: *Model, frames: i64, values: []const f32, output: []f32) !u32 {
        if (frames < 1 or frames > frames_max) return error.InvalidFrameCount;
        const in = zml.Shape.init(.{ frames, input_width }, .f32);
        const out = zml.Shape.init(.{ frames, output_width }, .f32);
        return switch (self.mode) {
            .moving => self.graph.infer(moving, in, out, values, output),
            .fixed => self.graph.infer(fixed, in, out, values, output),
        };
    }
};
