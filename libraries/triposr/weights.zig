const std = @import("std");
const runtime = @import("zml_lib/runtime");
const Parameter = runtime.layers.Parameter;

pub const Profile = struct {
    image: i64,
    patch: i64,
    position_grid: i64,
    dino_width: i64,
    dino_layers: usize,
    dino_heads: i64,
    plane: i64,
    width: i64,
    layers: usize,
    heads: i64,
    groups: i64,
    features: i64,
    decoder_width: i64,
    decoder_layers: usize,

    pub fn planeShape(p: Profile) struct { i64, i64, i64, i64 } {
        return .{ 3, p.features, 2 * p.plane, 2 * p.plane };
    }

    pub fn planeCount(p: Profile) usize {
        return @intCast(3 * p.features * (2 * p.plane) * (2 * p.plane));
    }
};

pub const official: Profile = .{
    .image = 512,
    .patch = 16,
    .position_grid = 14,
    .dino_width = 768,
    .dino_layers = 12,
    .dino_heads = 12,
    .plane = 32,
    .width = 1024,
    .layers = 16,
    .heads = 16,
    .groups = 32,
    .features = 40,
    .decoder_width = 64,
    .decoder_layers = 9,
};

pub const fixture: Profile = .{
    .image = 24,
    .patch = 4,
    .position_grid = 4,
    .dino_width = 16,
    .dino_layers = 2,
    .dino_heads = 4,
    .plane = 3,
    .width = 24,
    .layers = 2,
    .heads = 4,
    .groups = 4,
    .features = 4,
    .decoder_width = 12,
    .decoder_layers = 3,
};

fn tensor(comptime key: []const u8, dims: anytype) [1]Parameter {
    return .{.{ .name = key, .shape = .init(dims, .f32) }};
}

fn addParameter(comptime prefix: []const u8, dims: anytype, comptime bias: bool) if (bias) [2]Parameter else [1]Parameter {
    return if (bias)
        tensor(prefix ++ ".weight", dims) ++ tensor(prefix ++ ".bias", .{dims[0]})
    else
        tensor(prefix ++ ".weight", dims);
}

fn backboneAttention(comptime prefix: []const u8, width: i64, input: i64) [5]Parameter {
    return addParameter(prefix ++ ".to_q", .{ width, width }, false) ++
        addParameter(prefix ++ ".to_k", .{ width, input }, false) ++
        addParameter(prefix ++ ".to_v", .{ width, input }, false) ++
        addParameter(prefix ++ ".to_out.0", .{ width, width }, true);
}

const dino = "image_tokenizer.model.";

fn append(entries: []Parameter, cursor: *usize, values: []const Parameter) void {
    @memcpy(entries[cursor.*..][0..values.len], values);
    cursor.* += values.len;
}

pub fn encoderSchema(comptime p: Profile) [15 + 16 * p.dino_layers + 20 * p.layers]Parameter {
    @setEvalBranchQuota(200_000);
    const dw = p.dino_width;
    const w = p.width;
    var entries: [15 + 16 * p.dino_layers + 20 * p.layers]Parameter = undefined;
    var cursor: usize = 0;
    append(&entries, &cursor, &(tensor(dino ++ "embeddings.cls_token", .{ 1, 1, dw }) ++
        tensor(dino ++ "embeddings.position_embeddings", .{ 1, p.position_grid * p.position_grid + 1, dw }) ++
        addParameter(dino ++ "embeddings.patch_embeddings.projection", .{ dw, 3, p.patch, p.patch }, true)));
    inline for (0..p.dino_layers) |i| {
        const key = std.fmt.comptimePrint(dino ++ "encoder.layer.{d}.", .{i});
        append(&entries, &cursor, &(addParameter(key ++ "layernorm_before", .{dw}, true) ++
            addParameter(key ++ "attention.attention.query", .{ dw, dw }, true) ++
            addParameter(key ++ "attention.attention.key", .{ dw, dw }, true) ++
            addParameter(key ++ "attention.attention.value", .{ dw, dw }, true) ++
            addParameter(key ++ "attention.output.dense", .{ dw, dw }, true) ++
            addParameter(key ++ "layernorm_after", .{dw}, true) ++
            addParameter(key ++ "intermediate.dense", .{ 4 * dw, dw }, true) ++
            addParameter(key ++ "output.dense", .{ dw, 4 * dw }, true)));
    }
    append(&entries, &cursor, &(addParameter(dino ++ "layernorm", .{dw}, true) ++
        tensor("tokenizer.embeddings", .{ 3, w, p.plane, p.plane }) ++
        addParameter("backbone.norm", .{w}, true) ++
        addParameter("backbone.proj_in", .{ w, w }, true)));
    inline for (0..p.layers) |i| {
        const key = std.fmt.comptimePrint("backbone.transformer_blocks.{d}.", .{i});
        append(&entries, &cursor, &(addParameter(key ++ "norm1", .{w}, true) ++
            backboneAttention(key ++ "attn1", w, w) ++
            addParameter(key ++ "norm2", .{w}, true) ++
            backboneAttention(key ++ "attn2", w, dw) ++
            addParameter(key ++ "norm3", .{w}, true) ++
            addParameter(key ++ "ff.net.0.proj", .{ 8 * w, w }, true) ++
            addParameter(key ++ "ff.net.2", .{ w, 4 * w }, true)));
    }
    append(&entries, &cursor, &(addParameter("backbone.proj_out", .{ w, w }, true) ++
        tensor("post_processor.upsample.weight", .{ w, p.features, 2, 2 }) ++
        tensor("post_processor.upsample.bias", .{p.features})));
    std.debug.assert(cursor == entries.len);
    return entries;
}

pub fn fieldSchema(comptime p: Profile) [2 * (p.decoder_layers + 1)]Parameter {
    var entries: [2 * (p.decoder_layers + 1)]Parameter = undefined;
    var cursor: usize = 0;
    inline for (0..p.decoder_layers + 1) |i| {
        const key = std.fmt.comptimePrint("decoder.layers.{d}", .{2 * i});
        const in = if (i == 0) 3 * p.features else p.decoder_width;
        const out = if (i == p.decoder_layers) 4 else p.decoder_width;
        append(&entries, &cursor, &addParameter(key, .{ out, in }, true));
    }
    return entries;
}
