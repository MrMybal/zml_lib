const std = @import("std");
const rt = @import("zml_lib/runtime");
const Parameter = rt.layers.Parameter;

pub const width = 512;
pub const blocks = 12;
pub const keypoints = 17;
pub const position_width = 32;
pub const image_features = 1024;
pub const motion_width = 151;
pub const contact_width = 6;
pub const camera_width = 3;
pub const motion_features = 6;

fn parameter(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn norm(comptime name: []const u8, size: i64) [2]Parameter {
    return .{ parameter(name ++ ".weight", .{size}), parameter(name ++ ".bias", .{size}) };
}

fn linear(comptime name: []const u8, input: i64, output: i64) [2]Parameter {
    return .{ parameter(name ++ ".weight", .{ output, input }), parameter(name ++ ".bias", .{output}) };
}

fn mlp(comptime name: []const u8, input: i64, inner: i64, output: i64) [4]Parameter {
    return linear(name ++ ".fc1", input, inner) ++ linear(name ++ ".fc2", inner, output);
}

const input_parameters = [_]Parameter{parameter("learned_pos_params", .{ keypoints, position_width })} ++
    linear("learned_pos_linear", 2, position_width) ++
    mlp("embed_noisyobs", keypoints * position_width, width * 2, width) ++
    linear("cliffcam_embedder.0", camera_width, width) ++ linear("cliffcam_embedder.3", width, width) ++
    linear("cam_angvel_embedder.0", motion_features, width) ++ linear("cam_angvel_embedder.3", width, width) ++
    norm("imgseq_embedder.0", image_features) ++ linear("imgseq_embedder.1", image_features, width);

const output_parameters = mlp("final_layer", width, width, motion_width) ++
    mlp("pred_cam_head", width, width, camera_width) ++
    mlp("static_conf_head", width, width, contact_width);

pub const schema = makeSchema();

fn makeSchema() [input_parameters.len + 18 * blocks + output_parameters.len]Parameter {
    @setEvalBranchQuota(150_000);
    var result: [input_parameters.len + 18 * blocks + output_parameters.len]Parameter = undefined;
    result[0..input_parameters.len].* = input_parameters;
    for (0..blocks) |i| {
        const name = std.fmt.comptimePrint("blocks.{d}", .{i});
        result[input_parameters.len + i * 18 ..][0..18].* = [_]Parameter{
            parameter(name ++ ".gate_msa", .{ 1, 1, width }),
            parameter(name ++ ".gate_mlp", .{ 1, 1, width }),
        } ++ norm(name ++ ".norm1", width) ++
            linear(name ++ ".attn.query", width, width) ++ linear(name ++ ".attn.key", width, width) ++
            linear(name ++ ".attn.value", width, width) ++ linear(name ++ ".attn.proj", width, width) ++
            norm(name ++ ".norm2", width) ++ mlp(name ++ ".mlp", width, width * 4, width);
    }
    result[result.len - output_parameters.len ..].* = output_parameters;
    return result;
}
