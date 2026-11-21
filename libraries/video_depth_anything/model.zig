const std = @import("std");
const runtime = @import("zml_lib/runtime");
const dinov2 = @import("zml_lib/dinov2");
const depth = @import("zml_lib/depth_anything_v2");
const motion = @import("motion.zig");
const Tensor = runtime.layers.Tensor;
const backbone = dinov2.small;
const encoder_schema = dinov2.schemaFor(backbone);
const head_schema = depth.dptSchema("head", backbone.channels, .{ 48, 96, 192, 384 }, 64) ++ motion.schema;
const schema = encoder_schema ++ head_schema;

fn forward(parameters: []const Tensor, video: Tensor) Tensor {
    const features = dinov2.encode(backbone, parameters[0..encoder_schema.len], video);
    const head: runtime.layers.Weights = .{ .schema = &head_schema, .tensors = parameters[encoder_schema.len..] };
    return depth.dptGraph(head, "head", features, video.dim(2), video.dim(3), motion.forward);
}

pub fn pixelCount(frames: u32, height: u32, width: u32) !usize {
    if (frames == 0 or frames > motion.max_frames or height == 0 or width == 0 or
        height % dinov2.patch_size != 0 or width % dinov2.patch_size != 0)
        return error.InvalidVideoDimensions;
    const count = try std.math.mul(usize, frames, try std.math.mul(usize, height, width));
    _ = try std.math.mul(usize, count, 3 * @sizeOf(f32));
    return count;
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn infer(self: *Model, frames: u32, height: u32, width: u32, input: []const f32, output: []f32) !u32 {
        _ = try pixelCount(frames, height, width);
        const t, const h, const w = .{ @as(i64, frames), @as(i64, height), @as(i64, width) };
        return self.graph.infer(forward, .init(.{ t, 3, h, w }, .f32), .init(.{ t, h, w }, .f32), input, output);
    }
};
