const runtime = @import("zml_lib/runtime");
const heads = @import("zml_lib/sam_heads");
const weights = @import("weights.zig");
const encoder = @import("encoder.zig");
pub const memory = @import("memory.zig");
const l = runtime.layers;
const Tensor = l.Tensor;

pub const image_size = heads.image_size;
pub const embedding_count = heads.embed_count + heads.sam2_high_count;
pub const output_count = heads.sam2_output;
pub fn predictInputSize(points: usize) !usize {
    return (try heads.predictInputSize(.sam2, points)) + 1;
}

fn decode(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &weights.decoder, .tensors = parameters };
    const primary = input.slice(0, .{ .end = heads.embed_count }).reshape(.{ 1, 256, 64, 64 });
    const mode = input.slice(0, .single(input.dim(0) - 1));
    const embedding = w.get("no_mem_embed", "");
    const no_memory = embedding.mul(mode.broad(embedding.shape())).reshape(.{256});
    const image = l.biasAdd(primary, no_memory, 1);
    const conditioned = Tensor.concatenate(&.{ image.flatten(), input.slice(0, .{
        .start = heads.embed_count,
        .end = input.dim(0) - 1,
    }) }, 0);
    return heads.decode(.sam2, w, conditioned);
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
        return self.graph.infer(encoder.forward, .init(.{ 1, 3, image_size, image_size }, .f32), .init(.{embedding_count}, .f32), input, output);
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
        if (input.len != try predictInputSize(points)) return error.InvalidBufferSize;
        const mode = input[input.len - 1];
        if (mode != 0 and mode != 1) return error.InvalidMemoryMode;
        return self.graph.infer(decode, .init(.{@as(i64, @intCast(try predictInputSize(points)))}, .f32), .init(.{output_count}, .f32), input, output);
    }
};
