const dinov2 = @import("zml_lib/dinov2");
const runtime = @import("zml_lib/runtime");
const Tensor = runtime.layers.Tensor;
const head = @import("head.zig");
pub const dptGraph = head.graph;
pub const dptSchema = @import("weights.zig").schemaFor;

pub const Image = dinov2.Image;
pub const schema = dinov2.schema ++ @import("weights.zig").schema;
const encoder_count = dinov2.schema.len;

fn forward(parameters: []const Tensor, image: Tensor) Tensor {
    const blocks = dinov2.forward(parameters[0..encoder_count], image);
    return head.forward(parameters[encoder_count..], blocks, image.dim(2), image.dim(3));
}

pub fn trace(parameters: []const Tensor, image: Tensor) Tensor {
    const blocks = dinov2.forward(parameters[0..encoder_count], image);
    const depth = head.forward(parameters[encoder_count..], blocks, image.dim(2), image.dim(3));
    return Tensor.concatenate(&.{
        blocks[0].flatten(),
        blocks[1].flatten(),
        blocks[2].flatten(),
        blocks[3].flatten(),
        depth.flatten(),
    }, 0);
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .loadCheckpoint(session, &schema, path, .f16) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn predict(self: *Model, height: i64, width: i64, input: []const f32, output: []f32) !u32 {
        const size = try Image.init(height, width);
        return self.graph.infer(forward, .init(.{ 1, 3, size.height, size.width }, .f32), .init(.{ 1, size.height, size.width }, .f32), input, output);
    }
};
