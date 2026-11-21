const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;

pub const Version = enum { sam, sam2 };
pub const image_size = 1024;
pub const grid = 64;
pub const embed_count = 256 * grid * grid;
pub const low_size = 256;
pub const mask_count = low_size * low_size;
pub const base_output = 4 * mask_count + 4;
pub const sam2_high_count = 32 * 256 * 256 + 64 * 128 * 128;
pub const sam2_output = base_output + 4 * 256 + 1;

const valid = [4]i64{ 0, 0, 0, 0 };

fn positional(w: l.Weights, prefix: []const u8, xy: Tensor) Tensor {
    var key: [80]u8 = undefined;
    const matrix = w.get(l.name(&key, "{s}.pe_layer", .{prefix}), ".positional_encoding_gaussian_matrix");
    const phases = xy.scale(@as(f32, 2)).addConstant(@as(f32, -1)).dotGeneral(matrix, &.{.{ 1, 0 }}, &.{})
        .scale(@as(f32, 2 * std.math.pi));
    return Tensor.concatenate(&.{ phases.sin(), phases.cos() }, 1);
}

fn densePosition(w: l.Weights, prefix: []const u8) Tensor {
    const plane: @TypeOf(Tensor.fromShape(.init(.{ grid, grid }, .f32)).shape()) = .init(.{ grid, grid }, .f32);
    const x = Tensor.iota(plane, 1).convert(.f32).addConstant(@as(f32, 0.5)).divByConst(@as(f32, grid));
    const y = Tensor.iota(plane, 0).convert(.f32).addConstant(@as(f32, 0.5)).divByConst(@as(f32, grid));
    const coords = Tensor.stack(&.{ x, y }, 2, .coords).reshape(.{ grid * grid, 2 });
    return positional(w, prefix, coords).reshape(.{ 1, grid * grid, 256 });
}

fn pointEmbeddings(w: l.Weights, prefix: []const u8, points: Tensor) Tensor {
    const count = points.dim(0);
    const xy = points.slice(1, .{ .end = 2 }).addConstant(@as(f32, 0.5)).divByConst(@as(f32, image_size));
    const labels = points.slice(1, .single(2));
    const pe = positional(w, prefix, xy);
    var key: [80]u8 = undefined;
    const absent = labels.cmp(.EQ, Tensor.scalar(@as(f32, -1), .f32).broad(labels.shape()))
        .reshape(.{ count, 1 }).broad(pe.shape().withDtype(.bool));
    const pad = w.get(l.name(&key, "{s}.not_a_point_embed", .{prefix}), ".weight").broad(pe.shape());
    var embedding = Tensor.select(absent, pad, pe);
    for (0..4) |index| {
        const id = Tensor.scalar(@as(f32, @floatFromInt(index)), .f32);
        const match = labels.cmp(.EQ, id.broad(labels.shape())).convert(.f32).reshape(.{ count, 1 }).broad(pe.shape());
        const weight = w.get(l.name(&key, "{s}.point_embeddings.{d}", .{ prefix, index }), ".weight");
        embedding = embedding.add(match.mul(weight.broad(pe.shape())));
    }
    return embedding.reshape(.{ 1, count, 256 });
}

fn maskEmbedding(w: l.Weights, prefix: []const u8, mask: Tensor, has_mask: Tensor) Tensor {
    var key: [80]u8 = undefined;
    var y = l.conv(w, l.name(&key, "{s}.mask_downscaling.0", .{prefix}), mask, 2, valid, true);
    y = t.gelu(t.normNchw(w, l.name(&key, "{s}.mask_downscaling.1", .{prefix}), y, 1e-6));
    y = l.conv(w, l.name(&key, "{s}.mask_downscaling.3", .{prefix}), y, 2, valid, true);
    y = t.gelu(t.normNchw(w, l.name(&key, "{s}.mask_downscaling.4", .{prefix}), y, 1e-6));
    y = l.conv(w, l.name(&key, "{s}.mask_downscaling.6", .{prefix}), y, 1, valid, true);
    const no = w.get(l.name(&key, "{s}.no_mask_embed", .{prefix}), ".weight").reshape(.{256});
    return Tensor.select(has_mask.cmp(.GT, Tensor.scalar(@as(f32, 0), .f32)).broad(y.shape().withDtype(.bool)), y, no.broadcast(y.shape(), &.{1}));
}

fn att(w: l.Weights, base: []const u8, name: []const u8, q: Tensor, k: Tensor, v: Tensor) Tensor {
    var key: [104]u8 = undefined;
    const q_proj = t.linear(w, l.name(&key, "{s}.{s}.q_proj", .{ base, name }), q, true);
    const k_proj = t.linear(w, l.name(&key, "{s}.{s}.k_proj", .{ base, name }), k, true);
    const v_proj = t.linear(w, l.name(&key, "{s}.{s}.v_proj", .{ base, name }), v, true);
    return t.linear(w, l.name(&key, "{s}.{s}.out_proj", .{ base, name }), t.attention(q_proj, k_proj, v_proj, 8), true);
}

const Tokens = struct { queries: Tensor, image: Tensor };

fn twoWay(comptime version: Version, w: l.Weights, decoder: []const u8, tokens: Tensor, keys: Tensor, pe: Tensor) Tokens {
    var prefix: [80]u8 = undefined;
    var key: [96]u8 = undefined;
    var queries = tokens;
    var image = keys;
    const first_mlp = if (version == .sam) "lin1" else "layers.0";
    const last_mlp = if (version == .sam) "lin2" else "layers.1";
    for (0..2) |index| {
        const base = l.name(&prefix, "{s}.transformer.layers.{d}", .{ decoder, index });
        const q = if (index == 0) queries else queries.add(tokens);
        const self = att(w, base, "self_attn", q, q, queries);
        queries = t.norm(w, l.name(&key, "{s}.norm1", .{base}), if (index == 0) self else queries.add(self), 1e-5);
        const cross = att(w, base, "cross_attn_token_to_image", queries.add(tokens), image.add(pe), image);
        queries = t.norm(w, l.name(&key, "{s}.norm2", .{base}), queries.add(cross), 1e-5);
        const hidden = t.linear(w, l.name(&key, "{s}.mlp.{s}", .{ base, first_mlp }), queries, true).relu();
        const mlp = t.linear(w, l.name(&key, "{s}.mlp.{s}", .{ base, last_mlp }), hidden, true);
        queries = t.norm(w, l.name(&key, "{s}.norm3", .{base}), queries.add(mlp), 1e-5);
        const reverse = att(w, base, "cross_attn_image_to_token", image.add(pe), queries.add(tokens), queries);
        image = t.norm(w, l.name(&key, "{s}.norm4", .{base}), image.add(reverse), 1e-5);
    }
    const base = l.name(&prefix, "{s}.transformer", .{decoder});
    const final = att(w, base, "final_attn_token_to_image", queries.add(tokens), image.add(pe), image);
    queries = t.norm(w, l.name(&key, "{s}.norm_final_attn", .{base}), queries.add(final), 1e-5);
    return .{ .queries = queries, .image = image };
}

pub fn reluMlp(w: l.Weights, base: []const u8, x: Tensor, count: usize) Tensor {
    var key: [104]u8 = undefined;
    var y = x;
    for (0..count) |i| {
        y = t.linear(w, l.name(&key, "{s}.layers.{d}", .{ base, i }), y, true);
        if (i + 1 != count) y = y.relu();
    }
    return y;
}

pub fn decode(comptime version: Version, w: l.Weights, input: Tensor) Tensor {
    const prompt = if (version == .sam) "prompt_encoder" else "sam_prompt_encoder";
    const decoder = if (version == .sam) "mask_decoder" else "sam_mask_decoder";
    const first_mask = if (version == .sam) 1 else 2;
    const high_count = if (version == .sam) 0 else sam2_high_count;
    const prompt_start = embed_count + high_count + 1 + mask_count;
    const n = @divExact(input.dim(0) - prompt_start, 3);
    const prompts = input.slice(0, .{ .start = prompt_start }).reshape(.{ n, 3 });
    const has_mask = input.slice(0, .single(embed_count + high_count));
    const mask = input.slice(0, .{ .start = embed_count + high_count + 1, .end = prompt_start })
        .reshape(.{ 1, 1, low_size, low_size });
    const image = input.slice(0, .{ .end = embed_count }).reshape(.{ 1, 256, grid, grid });
    const sparse = pointEmbeddings(w, prompt, prompts);
    const source = image.add(maskEmbedding(w, prompt, mask, has_mask))
        .reshape(.{ 1, 256, grid * grid }).transpose(.{ 0, 2, 1 });
    var key: [104]u8 = undefined;
    const iou_token = w.get(l.name(&key, "{s}.iou_token", .{decoder}), ".weight").reshape(.{ 1, 1, 256 });
    const masks = w.get(l.name(&key, "{s}.mask_tokens", .{decoder}), ".weight").reshape(.{ 1, 4, 256 });
    const out = if (version == .sam)
        Tensor.concatenate(&.{ iou_token, masks, sparse }, 1)
    else
        Tensor.concatenate(&.{ w.get("sam_mask_decoder.obj_score_token", ".weight").reshape(.{ 1, 1, 256 }), iou_token, masks, sparse }, 1);
    const result = twoWay(version, w, decoder, out, source, densePosition(w, prompt));
    var pixels = result.image.transpose(.{ 0, 2, 1 }).reshape(.{ 1, 256, grid, grid });
    pixels = t.deconv2x(w, l.name(&key, "{s}.output_upscaling.0", .{decoder}), pixels);
    if (version == .sam2) {
        const h1 = input.slice(0, .{ .start = embed_count + 32 * 256 * 256, .end = embed_count + high_count })
            .reshape(.{ 1, 64, 128, 128 });
        pixels = pixels.add(h1);
    }
    pixels = t.gelu(t.normNchw(w, l.name(&key, "{s}.output_upscaling.1", .{decoder}), pixels, 1e-6));
    pixels = t.deconv2x(w, l.name(&key, "{s}.output_upscaling.3", .{decoder}), pixels);
    if (version == .sam2) {
        const h0 = input.slice(0, .{ .start = embed_count, .end = embed_count + 32 * 256 * 256 })
            .reshape(.{ 1, 32, 256, 256 });
        pixels = pixels.add(h0);
    }
    pixels = t.gelu(pixels).reshape(.{ 32, mask_count });
    var projections: [4]Tensor = undefined;
    for (0..4) |i| {
        const token = result.queries.slice(1, .single(first_mask + @as(i64, @intCast(i))));
        const base = l.name(&key, "{s}.output_hypernetworks_mlps.{d}", .{ decoder, i });
        projections[i] = reluMlp(w, base, token, 3);
    }
    const logits = Tensor.concatenate(&projections, 0).dotGeneral(pixels, &.{.{ 1, 0 }}, &.{}).flatten();
    const iou = reluMlp(w, l.name(&key, "{s}.iou_prediction_head", .{decoder}), result.queries.slice(1, .single(first_mask - 1)), 3).flatten();
    if (version == .sam) return Tensor.concatenate(&.{ logits, iou }, 0);
    const obj_score = reluMlp(w, "sam_mask_decoder.pred_obj_score_head", result.queries.slice(1, .single(0)), 3).flatten();
    const mask_tokens = result.queries.slice(1, .{ .start = first_mask, .end = first_mask + 4 }).flatten();
    return Tensor.concatenate(&.{ logits, iou.sigmoid(), mask_tokens, obj_score }, 0);
}

pub fn predictInputSize(version: Version, points: usize) !usize {
    if (points == 0 or points > 256) return error.InvalidPointCount;
    return embed_count + (if (version == .sam2) @as(usize, sam2_high_count) else 0) + 1 + mask_count + 3 * points;
}
