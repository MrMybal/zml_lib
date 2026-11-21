const runtime = @import("zml_lib/runtime");
const heads = @import("zml_lib/sam_heads");
const weights = @import("weights.zig");
const encoder = @import("encoder.zig");
const Tensor = runtime.layers.Tensor;

pub const image_size = heads.image_size;
pub const embedding_count = heads.embed_count;
pub const output_count = heads.base_output;
pub fn predictInputSize(points: usize) !usize {
    return heads.predictInputSize(.sam, points);
}

fn decode(parameters: []const Tensor, input: Tensor) Tensor {
    return heads.decode(.sam, .{ .schema = &weights.decoder, .tensors = parameters }, input);
}

pub const Encoder = struct {
    graph: runtime.TensorModel,
    pub fn load(session: *const runtime.Session, path: []const u8) !Encoder {
        return .{ .graph = try .load(session, &weights.encoder, path) };
    }
    pub fn deinit(self: *Encoder) void {
        self.graph.deinit();
    }
    pub fn encode(self: *Encoder, input: []const f32, output: []f32) !u32 {
        return self.graph.infer(encoder.forward, .init(.{ 1, 3, image_size, image_size }, .f32), .init(.{ 1, 256, 64, 64 }, .f32), input, output);
    }
};

pub const Decoder = struct {
    graph: runtime.TensorModel,
    pub fn load(session: *const runtime.Session, path: []const u8) !Decoder {
        return .{ .graph = try .load(session, &weights.decoder, path) };
    }
    pub fn deinit(self: *Decoder) void {
        self.graph.deinit();
    }
    pub fn predict(self: *Decoder, points: usize, input: []const f32, output: []f32) !u32 {
        return self.graph.infer(decode, .init(.{@as(i64, @intCast(try predictInputSize(points)))}, .f32), .init(.{output_count}, .f32), input, output);
    }
};
