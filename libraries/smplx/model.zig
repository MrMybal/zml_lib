const rt = @import("zml_lib/runtime");
const Tensor = rt.layers.Tensor;
const Parameter = rt.layers.Parameter;

pub const vertices = 10475;
pub const joints = 55;
pub const batch_size = 16;
pub const output_points = vertices + joints;
const shape_width = 20;
const main_pose_width = 75;
const hand_pca_width = 12;
const left_start = shape_width + main_pose_width;
const right_start = left_start + hand_pca_width;
const translation_start = right_start + hand_pca_width;
pub const input_width = translation_start + 3;
const pose_features = (joints - 1) * 9;
const coordinates = vertices * 3;

const parents = [_]usize{
    0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
};

pub const schema = [_]Parameter{
    .{ .name = "v_template", .shape = .init(.{coordinates}, .f32) },
    .{ .name = "shapedirs", .shape = .init(.{ shape_width, coordinates }, .f32) },
    .{ .name = "posedirs", .shape = .init(.{ pose_features, coordinates }, .f32) },
    .{ .name = "J_regressor", .shape = .init(.{ joints, vertices }, .f32) },
    .{ .name = "lbs_weights", .shape = .init(.{ vertices, joints }, .f32) },
    .{ .name = "left_hand_components", .shape = .init(.{ hand_pca_width, 45 }, .f32) },
    .{ .name = "right_hand_components", .shape = .init(.{ hand_pca_width, 45 }, .f32) },
    .{ .name = "pose_mean", .shape = .init(.{joints * 3}, .f32) },
};

fn rotations(pose: Tensor, identity: Tensor) Tensor {
    const shifted = pose.addConstant(@as(f32, 1e-8));
    const angle = shifted.mul(shifted).sum(2).sqrt().maximum(.scalar(@as(f32, 1e-12), .f32));
    const unit = pose.div(angle.broad(pose.shape()));
    const x = unit.slice(2, .{ .end = 1 });
    const y = unit.slice(2, .{ .start = 1, .end = 2 });
    const z = unit.slice(2, .{ .start = 2 });
    const zero = Tensor.scalar(@as(f32, 0), .f32).broad(x.shape());
    const skew = Tensor.concatenate(&.{ zero, z.negate(), y, z, zero, x.negate(), y.negate(), x, zero }, 2)
        .reshape(.{ pose.dim(0), joints, 3, 3 });
    const squared = skew.dotGeneral(skew, &.{.{ 3, 2 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
    const sine = angle.sin().broadcast(skew.shape(), &.{ 0, 1, 2 });
    const one_minus_cos = angle.cos().negate().addConstant(@as(f32, 1)).broadcast(skew.shape(), &.{ 0, 1, 2 });
    return identity.broadcast(skew.shape(), &.{ 2, 3 }).add(sine.mul(skew)).add(one_minus_cos.mul(squared));
}

const Kinematics = struct { affine: Tensor, posed_joints: Tensor };

fn chain(rest_joints: Tensor, local_rot: Tensor) Kinematics {
    const frames = local_rot.dim(0);
    var global_rot: [joints]Tensor = undefined;
    var global_pos: [joints]Tensor = undefined;
    for (parents, 0..) |parent, joint| {
        const r = local_rot.slice(1, .single(@intCast(joint))).reshape(.{ frames, 3, 3 });
        const rest = rest_joints.slice(1, .single(@intCast(joint))).reshape(.{ frames, 3, 1 });
        if (joint == 0) {
            global_rot[joint] = r;
            global_pos[joint] = rest;
        } else {
            const delta = rest.sub(rest_joints.slice(1, .single(@intCast(parent))).reshape(rest.shape()));
            global_rot[joint] = global_rot[parent].dotGeneral(r, &.{.{ 2, 1 }}, &.{.{ 0, 0 }});
            global_pos[joint] = global_pos[parent].add(global_rot[parent].dotGeneral(delta, &.{.{ 2, 1 }}, &.{.{ 0, 0 }}));
        }
    }
    const rotation = Tensor.stack(&global_rot, 1, .joint);
    const position = Tensor.stack(&global_pos, 1, .joint);
    // Subtract the transformed rest joint before blending the affine transforms.
    const rotated_rest = rotation.dotGeneral(rest_joints.reshape(.{ frames, joints, 3, 1 }), &.{.{ 3, 2 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
    return .{
        .affine = Tensor.concatenate(&.{ rotation, position.sub(rotated_rest) }, 3),
        .posed_joints = position.reshape(.{ frames, joints, 3 }),
    };
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: rt.layers.Weights = .{ .schema = &schema, .tensors = parameters };
    const frames = input.dim(0);
    const shape = input.slice(1, .{ .end = shape_width });
    const main_pose = input.slice(1, .{ .start = shape_width, .end = left_start });
    const left = input.slice(1, .{ .start = left_start, .end = right_start }).dotGeneral(w.get("left_hand_components", ""), &.{.{ 1, 0 }}, &.{});
    const right = input.slice(1, .{ .start = right_start, .end = translation_start }).dotGeneral(w.get("right_hand_components", ""), &.{.{ 1, 0 }}, &.{});
    const translation = input.slice(1, .{ .start = translation_start });
    const full_pose = Tensor.concatenate(&.{ main_pose, left, right }, 1);
    const pose = full_pose.add(w.get("pose_mean", "").broadcast(full_pose.shape(), &.{1})).reshape(.{ frames, joints, 3 });

    const offsets = shape.dotGeneral(w.get("shapedirs", ""), &.{.{ 1, 0 }}, &.{});
    const shaped = offsets.add(w.get("v_template", "").broadcast(offsets.shape(), &.{1})).reshape(.{ frames, vertices, 3 });
    const rest = shaped.dotGeneral(w.get("J_regressor", ""), &.{.{ 1, 1 }}, &.{}).transpose(.{ 0, 2, 1 });

    const row = Tensor.iota(.init(.{ 3, 3 }, .i32), 0);
    const col = Tensor.iota(row.shape(), 1);
    const identity = row.cmp(.EQ, col).convert(.f32);
    const local_rot = rotations(pose, identity);
    const nonroot = local_rot.slice(1, .{ .start = 1 });
    const features = nonroot.sub(identity.broadcast(nonroot.shape(), &.{ 2, 3 })).reshape(.{ frames, pose_features });
    const corrective = features.dotGeneral(w.get("posedirs", ""), &.{.{ 1, 0 }}, &.{}).reshape(shaped.shape());
    const posed = shaped.add(corrective);

    const kinematics = chain(rest, local_rot);
    const blended = kinematics.affine.reshape(.{ frames, joints, 12 })
        .dotGeneral(w.get("lbs_weights", ""), &.{.{ 1, 1 }}, &.{}).transpose(.{ 0, 2, 1 })
        .reshape(.{ frames, vertices, 3, 4 });
    const ones = Tensor.scalar(@as(f32, 1), .f32).broad(.init(.{ frames, vertices, 1 }, .f32));
    const homogeneous = Tensor.concatenate(&.{ posed, ones }, 2);
    const mesh = blended.dotGeneral(homogeneous, &.{.{ 3, 2 }}, &.{ .{ 0, 0 }, .{ 1, 1 } });
    const points = Tensor.concatenate(&.{ mesh, kinematics.posed_joints }, 1);
    return points.add(translation.broadcast(points.shape(), &.{ 0, 2 }));
}

pub const Model = struct {
    graph: rt.TensorModel,

    pub fn load(session: *const rt.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn predict(self: *Model, frames: i64, input: []const f32, points: []f32) !u32 {
        if (frames < 1 or frames > batch_size) return error.InvalidFrameCount;
        return self.graph.infer(forward, .init(.{ frames, input_width }, .f32), .init(.{ frames, output_points, 3 }, .f32), input, points);
    }
};
