const runtime = @import("zml_lib/runtime");
const heads = @import("zml_lib/sam_heads");
const weights = @import("weights.zig");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;

const pixels = 1024 * 1024;
const spatial_positions = 64 * 64;
const memory_channels = 64;
pub const spatialMemory = spatial_positions * memory_channels;
pub const pointerSize = 256;
pub const memoryOutputSize = spatialMemory + pointerSize;
pub const memoryInputSize = heads.embed_count + pixels + pointerSize + 2;
pub const attentionOutputSize = heads.embed_count;
const slot_width = spatialMemory + 1;
const valid = [4]i64{ 0, 0, 0, 0 };
const same = [4]i64{ 1, 1, 1, 1 };

fn encode(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &weights.memory, .tensors = parameters };
    const features = input.slice(0, .{ .end = heads.embed_count }).reshape(.{ 1, 256, 64, 64 });
    const masks = input.slice(0, .{ .start = heads.embed_count, .end = heads.embed_count + pixels })
        .reshape(.{ 1, 1, 1024, 1024 });
    const pointer = input.slice(0, .{ .start = heads.embed_count + pixels, .end = memoryInputSize - 2 }).reshape(.{ 1, 256 });
    const threshold = Tensor.scalar(@as(f32, 0), .f32);
    const visible = input.slice(0, .single(memoryInputSize - 2)).cmp(.GT, threshold);
    const from_prompt = input.slice(0, .single(memoryInputSize - 1)).cmp(.GT, threshold);
    const binary = masks.cmp(.GT, threshold.broad(masks.shape())).convert(.f32);
    var mask = Tensor.select(from_prompt.broad(binary.shape().withDtype(.bool)), binary, masks.sigmoid())
        .scale(@as(f32, 20)).addConstant(@as(f32, -10));
    var key: [104]u8 = undefined;
    for (0..4) |i| {
        mask = l.conv(w, l.name(&key, "memory_encoder.mask_downsampler.encoder.{d}", .{3 * i}), mask, 2, same, true);
        mask = t.gelu(t.normNchw(w, l.name(&key, "memory_encoder.mask_downsampler.encoder.{d}", .{3 * i + 1}), mask, 1e-6));
    }
    mask = l.conv(w, "memory_encoder.mask_downsampler.encoder.12", mask, 1, valid, true);
    var y = l.conv(w, "memory_encoder.pix_feat_proj", features, 1, valid, true).add(mask);
    var prefix: [72]u8 = undefined;
    for (0..2) |i| {
        const base = l.name(&prefix, "memory_encoder.fuser.layers.{d}", .{i});
        const convolution = l.convGroups(w, l.name(&key, "{s}.dwconv", .{base}), y, 1, @splat(3), true, 256);
        const normalized = t.normNchw(w, l.name(&key, "{s}.norm", .{base}), convolution, 1e-6).transpose(.{ 0, 2, 3, 1 });
        const hidden = t.gelu(t.linear(w, l.name(&key, "{s}.pwconv1", .{base}), normalized, true));
        const projected = t.linear(w, l.name(&key, "{s}.pwconv2", .{base}), hidden, true);
        const gamma = w.get(base, ".gamma").broadcast(projected.shape(), &.{3});
        y = y.add(projected.mul(gamma).transpose(.{ 0, 3, 1, 2 }));
    }
    const spatial = l.conv(w, "memory_encoder.out_proj", y, 1, valid, true).flatten();
    const object = heads.reluMlp(w, "obj_ptr_proj", pointer, 3);
    const object_pointer = Tensor.select(visible.broad(object.shape().withDtype(.bool)), object, w.get("no_obj_ptr", ""));
    return Tensor.concatenate(&.{ spatial, object_pointer.flatten() }, 0);
}

fn rotate(w: l.Weights, x: Tensor, blocks: i64) Tensor {
    const c = w.get("constants.rope_cos", "");
    const s = w.get("constants.rope_sin", "");
    const cos = c.broadcast(.init(.{ blocks, spatial_positions, 128 }, .f32), &.{ 1, 2 })
        .reshape(.{ 1, blocks * spatial_positions, 128 });
    const sin = s.broadcast(.init(.{ blocks, spatial_positions, 128 }, .f32), &.{ 1, 2 })
        .reshape(cos.shape());
    const pairs = x.reshape(.{ 1, blocks * spatial_positions, 128, 2 });
    const re = pairs.slice(3, .single(0));
    const im = pairs.slice(3, .single(1));
    return Tensor.stack(&.{ re.mul(cos).sub(im.mul(sin)), re.mul(sin).add(im.mul(cos)) }, 3, .pair)
        .reshape(x.shape());
}

fn ropeAttention(w: l.Weights, prefix: []const u8, q_in: Tensor, k_in: Tensor, v_in: Tensor, slots: i64) Tensor {
    var key: [104]u8 = undefined;
    const q = rotate(w, t.linear(w, l.name(&key, "{s}.q_proj", .{prefix}), q_in, true), 1);
    const keys = t.linear(w, l.name(&key, "{s}.k_proj", .{prefix}), k_in, true);
    const length = slots * spatial_positions;
    const rotated = rotate(w, keys.slice(1, .{ .end = length }), slots);
    const k = if (length == keys.dim(1)) rotated else Tensor.concatenate(&.{ rotated, keys.slice(1, .{ .start = length }) }, 1);
    const v = t.linear(w, l.name(&key, "{s}.v_proj", .{prefix}), v_in, true);
    // Bound attention scratch space when the past contains several frames.
    const rows = 256;
    var chunks: [spatial_positions / rows]Tensor = undefined;
    for (&chunks, 0..) |*chunk, i| {
        const start: i64 = @intCast(i * rows);
        chunk.* = t.attention(q.slice(1, .{ .start = start, .end = start + rows }), k, v, 1);
    }
    const context = Tensor.concatenate(&chunks, 1);
    return t.linear(w, l.name(&key, "{s}.out_proj", .{prefix}), context, true);
}

fn attend(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &weights.attention, .tensors = parameters };
    const count = input.dim(0) - heads.embed_count;
    const slot_count = @divFloor(count, slot_width);
    const pointer_count = @divExact(@mod(count, slot_width), pointerSize);
    const current = input.slice(0, .{ .end = heads.embed_count }).reshape(.{ 1, 256, spatial_positions }).transpose(.{ 0, 2, 1 });
    const pos = w.get("constants.memory_position", "");
    const tpos = w.get("maskmem_tpos_enc", "").reshape(.{ .temporal = 7, .channel = 64 });
    var values: [8]Tensor = undefined;
    var keys: [8]Tensor = undefined;
    for (0..@as(usize, @intCast(slot_count))) |i| {
        const from: i64 = @intCast(heads.embed_count + i * slot_width);
        const temporal = input.slice(0, .{ .start = from, .end = from + 1 }).convert(.i32).withTags(.{.slot});
        const encoded = tpos.gather(.{ .temporal = temporal }, .{}).reshape(.{ 1, 1, 64 });
        const feature = input.slice(0, .{ .start = from + 1, .end = from + slot_width })
            .reshape(.{ 1, 64, spatial_positions }).transpose(.{ 0, 2, 1 });
        values[i] = feature;
        keys[i] = feature.add(pos).add(encoded.broad(feature.shape()));
    }
    var segments: usize = @intCast(slot_count);
    if (pointer_count != 0) {
        const ptrs = input.slice(0, .{ .start = heads.embed_count + slot_width * slot_count })
            .reshape(.{ 1, pointer_count * 4, 64 });
        values[segments] = ptrs;
        keys[segments] = ptrs;
        segments += 1;
    }
    const value = Tensor.concatenate(values[0..segments], 1);
    const key_memory = Tensor.concatenate(keys[0..segments], 1);
    var target = current.add(w.get("constants.vision_position", "").scale(@as(f32, 0.1)));
    var prefix: [80]u8 = undefined;
    var key: [104]u8 = undefined;
    for (0..4) |i| {
        const base = l.name(&prefix, "memory_attention.layers.{d}", .{i});
        const a = t.norm(w, l.name(&key, "{s}.norm1", .{base}), target, 1e-5);
        target = target.add(ropeAttention(w, l.name(&key, "{s}.self_attn", .{base}), a, a, a, 1));
        const b = t.norm(w, l.name(&key, "{s}.norm2", .{base}), target, 1e-5);
        target = target.add(ropeAttention(w, l.name(&key, "{s}.cross_attn_image", .{base}), b, key_memory, value, slot_count));
        const c = t.norm(w, l.name(&key, "{s}.norm3", .{base}), target, 1e-5);
        const ffn = t.linear(w, l.name(&key, "{s}.linear1", .{base}), c, true).relu();
        target = target.add(t.linear(w, l.name(&key, "{s}.linear2", .{base}), ffn, true));
    }
    return t.norm(w, "memory_attention.norm", target, 1e-5).transpose(.{ 0, 2, 1 }).flatten();
}

pub const Encoder = struct {
    graph: runtime.TensorModel,
    pub fn load(session: *const runtime.Session, path: []const u8) !Encoder {
        return .{ .graph = try .load(session, &weights.memory, path) };
    }
    pub fn deinit(self: *Encoder) void {
        self.graph.deinit();
    }
    pub fn run(self: *Encoder, input: []const f32, output: []f32) !u32 {
        return self.graph.infer(encode, .init(.{memoryInputSize}, .f32), .init(.{memoryOutputSize}, .f32), input, output);
    }
};

pub const Attention = struct {
    graph: runtime.TensorModel,
    pub fn load(session: *const runtime.Session, path: []const u8) !Attention {
        return .{ .graph = try .load(session, &weights.attention, path) };
    }
    pub fn deinit(self: *Attention) void {
        self.graph.deinit();
    }
    pub fn run(self: *Attention, input: []const f32, output: []f32) !u32 {
        if (input.len < heads.embed_count + slot_width) return error.InvalidMemorySize;
        const count = input.len - heads.embed_count;
        const slots = count / slot_width;
        const extra = count % slot_width;
        if (slots > 7 or extra % pointerSize != 0 or extra / pointerSize > 16) return error.InvalidMemorySize;
        for (0..slots) |i| {
            const index = input[heads.embed_count + i * slot_width];
            if (!(index >= 0 and index < 7 and index == @floor(index))) return error.InvalidTemporalIndex;
        }
        return self.graph.infer(attend, .init(.{@as(i64, @intCast(input.len))}, .f32), .init(.{attentionOutputSize}, .f32), input, output);
    }
};
