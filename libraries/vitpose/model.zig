const rt = @import("zml_lib/runtime");
const l = rt.layers;
const vit = rt.pose_vit;
const udp = @import("udp.zig");
const Tensor = l.Tensor;
const parameter = vit.parameter;

pub const input_dims = .{ 3, vit.height, vit.width };
pub const output_dims = .{ udp.joints, udp.height, udp.width };
pub const schema = vit.schema ++ [_]l.Parameter{
    parameter("keypoint_head.deconv_layers.0.weight", .{ vit.channels, 256, 4, 4 }),
    parameter("keypoint_head.deconv_layers.1.weight", .{256}),
    parameter("keypoint_head.deconv_layers.1.bias", .{256}),
    parameter("keypoint_head.deconv_layers.1.running_mean", .{256}),
    parameter("keypoint_head.deconv_layers.1.running_var", .{256}),
    parameter("keypoint_head.deconv_layers.3.weight", .{ 256, 256, 4, 4 }),
    parameter("keypoint_head.deconv_layers.4.weight", .{256}),
    parameter("keypoint_head.deconv_layers.4.bias", .{256}),
    parameter("keypoint_head.deconv_layers.4.running_mean", .{256}),
    parameter("keypoint_head.deconv_layers.4.running_var", .{256}),
    parameter("keypoint_head.final_layer.weight", .{ 17, 256, 1, 1 }),
    parameter("keypoint_head.final_layer.bias", .{17}),
};

fn upsample(w: l.Weights, input: Tensor, comptime conv: []const u8, comptime norm: []const u8) Tensor {
    // ConvTranspose2d with kernel 4, stride 2, padding 1 and no output padding.
    const kernel = w.get(conv, ".weight").transpose(.{ 1, 0, 2, 3 }).reverse(.{ 2, 3 });
    const x = input.conv2d(kernel, .{ .lhs_dilation = &.{ 2, 2 }, .padding = &.{ 2, 2, 2, 2 } });
    return l.batchNorm(w, norm, x).maximum(.scalar(@as(f32, 0), .f32));
}

pub fn forward(params: []const Tensor, image: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = params };
    const n = image.dim(0);
    const inputs = Tensor.concatenate(&.{ image, image.reverse(.{3}) }, 0);
    const tokens = vit.encode(w, inputs);
    const features = tokens.transpose(.{ 0, 2, 1 }).reshape(.{ 2 * n, vit.channels, vit.grid_height, vit.grid_width });
    const first = upsample(w, features, "keypoint_head.deconv_layers.0", "keypoint_head.deconv_layers.1");
    const second = upsample(w, first, "keypoint_head.deconv_layers.3", "keypoint_head.deconv_layers.4");
    const heatmaps = l.conv(w, "keypoint_head.final_layer", second, 1, @splat(0), true);
    const direct = heatmaps.slice(0, .{ .end = n });
    const flipped = heatmaps.slice(0, .{ .start = n }).reverse(.{3});
    const pairs = flipped.slice(1, .{ .start = 1 }).reshape(.{ n, 8, 2, 64, 48 })
        .reverse(.{2}).reshape(.{ n, 16, 64, 48 });
    const restored = Tensor.concatenate(&.{ flipped.slice(1, .{ .end = 1 }), pairs }, 1);
    return direct.add(restored).scale(@as(f32, 0.5));
}

const crop_values = 3 * vit.height * vit.width;
const heatmap_values = udp.joints * udp.height * udp.width;
const point_values = udp.joints * 3;
const pose_parameters = schema;

fn unpack(rows: Tensor, comptime values: i64, comptime dims: anytype) struct { Tensor, Tensor } {
    return .{
        rows.slice(1, .{ .end = values }).reshape(.{rows.dim(0)} ++ dims),
        rows.slice(1, .{ .start = values }),
    };
}

fn inferPose(parameters: []const Tensor, rows: Tensor) struct { Tensor, Tensor } {
    const crops, const boxes = unpack(rows, crop_values, input_dims);
    const heatmaps = forward(parameters, crops);
    return .{ udp.decode(heatmaps, boxes), heatmaps };
}

pub const Joints = struct {
    pub const schema = pose_parameters;
    pub const input_dims = .{crop_values + 3};
    pub const output_dims = .{ udp.joints, 3 };

    pub fn forward(parameters: []const Tensor, rows: Tensor) Tensor {
        return inferPose(parameters, rows)[0];
    }
};

pub const JointsAndHeatmaps = struct {
    pub const schema = pose_parameters;
    pub const input_dims = Joints.input_dims;
    pub const output_dims = .{point_values + heatmap_values};

    pub fn forward(parameters: []const Tensor, rows: Tensor) Tensor {
        const points, const heatmaps = inferPose(parameters, rows);
        const frames = rows.dim(0);
        return Tensor.concatenate(&.{ points.reshape(.{ frames, point_values }), heatmaps.reshape(.{ frames, heatmap_values }) }, 1);
    }
};

const heatmap_dims = output_dims;

pub const HeatmapDecoder = struct {
    pub const schema = [_]l.Parameter{};
    pub const input_dims = .{heatmap_values + 3};
    pub const output_dims = Joints.output_dims;

    pub fn forward(_: []const Tensor, rows: Tensor) Tensor {
        const heatmaps, const boxes = unpack(rows, heatmap_values, heatmap_dims);
        return udp.decode(heatmaps, boxes);
    }
};
