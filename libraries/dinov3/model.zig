const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const t = rt.transformer;
const Tensor = zml.Tensor;

pub const channels = 1024;
pub const patch_size = 16;
pub const prefix_tokens = 5;
const heads = 16;
const depth = channels / heads;
const layers = 24;
const intermediate = 4096;
const eps: f32 = 1e-5;

pub const schema = makeSchema();

fn makeSchema() [415]l.Parameter {
    @setEvalBranchQuota(800_000);
    var tensors: [415]l.Parameter = undefined;
    tensors[0] = .{ .name = "embeddings.cls_token", .shape = .init(.{ 1, 1, channels }, .f32) };
    tensors[1] = .{ .name = "embeddings.mask_token", .shape = .init(.{ 1, 1, channels }, .f32) };
    tensors[2] = .{ .name = "embeddings.register_tokens", .shape = .init(.{ 1, 4, channels }, .f32) };
    tensors[3] = .{ .name = "embeddings.patch_embeddings.weight", .shape = .init(.{ channels, 3, patch_size, patch_size }, .f32) };
    tensors[4] = .{ .name = "embeddings.patch_embeddings.bias", .shape = .init(.{channels}, .f32) };
    const names = .{
        "norm1.weight",            "norm1.bias",            "norm2.weight",            "norm2.bias",            "attention.k_proj.weight",
        "attention.q_proj.weight", "attention.q_proj.bias", "attention.v_proj.weight", "attention.v_proj.bias", "attention.o_proj.weight",
        "attention.o_proj.bias",   "mlp.up_proj.weight",    "mlp.up_proj.bias",        "mlp.down_proj.weight",  "mlp.down_proj.bias",
        "layer_scale1.lambda1",    "layer_scale2.lambda1",
    };
    const shapes = .{
        .{channels},             .{channels},                 .{channels},             .{channels},                 .{ channels, channels },
        .{ channels, channels }, .{channels},                 .{ channels, channels }, .{channels},                 .{ channels, channels },
        .{channels},             .{ intermediate, channels }, .{intermediate},         .{ channels, intermediate }, .{channels},
        .{channels},             .{channels},
    };
    var offset: usize = 5;
    inline for (0..layers) |i| {
        inline for (names, shapes) |name, dimensions| {
            tensors[offset] = .{
                .name = std.fmt.comptimePrint("layer.{d}.{s}", .{ i, name }),
                .shape = .init(dimensions, .f32),
            };
            offset += 1;
        }
    }
    tensors[offset] = .{ .name = "norm.weight", .shape = .init(.{channels}, .f32) };
    tensors[offset + 1] = .{ .name = "norm.bias", .shape = .init(.{channels}, .f32) };
    return tensors;
}

fn axisAngles(shape: zml.Shape, axis: u3, size: i64, frequency: Tensor) Tensor {
    const side: f32 = @floatFromInt(size);
    const coordinates = Tensor.iota(shape.withDtype(.i32), axis).convert(.f32)
        .addConstant(@as(f32, 0.5)).divByConst(side).scale(@as(f32, 2)).subConstant(@as(f32, 1));
    return coordinates.scale(@as(f32, 2 * std.math.pi)).mul(frequency);
}

fn angles(rows: i64, cols: i64) Tensor {
    const shape: zml.Shape = .init(.{ rows, cols, depth / 4 }, .f32);
    const exponent = Tensor.iota(shape.withDtype(.i32), 2).convert(.f32).scale(@as(f32, -4.0) / @as(f32, depth));
    const freq = Tensor.scalar(@as(f32, 100), .f32).broad(shape).pow(exponent);
    const pair = Tensor.concatenate(&.{ axisAngles(shape, 0, rows, freq), axisAngles(shape, 1, cols, freq) }, 2);
    const spatial = Tensor.concatenate(&.{ pair, pair }, 2).reshape(.{ rows * cols, depth });
    return spatial.withTags(.{ .token, .channel }).pad(@as(f32, 0), .{ .token = Tensor.Pad{ .low = prefix_tokens } });
}

fn rotate(x: Tensor, sin: Tensor, cos: Tensor) Tensor {
    const turn = Tensor.concatenate(&.{ x.slice(3, .{ .start = depth / 2 }).negate(), x.slice(3, .{ .end = depth / 2 }) }, 3);
    return x.mul(cos.broadcast(x.shape(), &.{ 2, 3 })).add(turn.mul(sin.broadcast(x.shape(), &.{ 2, 3 })));
}

fn block(w: l.Weights, x: Tensor, index: usize, sin: Tensor, cos: Tensor) Tensor {
    var name: [64]u8 = undefined;
    var scope: [24]u8 = undefined;
    const prefix = l.name(&scope, "layer.{d}", .{index});
    const normalized = t.norm(w, l.name(&name, "{s}.norm1", .{prefix}), x, eps);
    const q = rotate(t.splitHeads(t.linear(w, l.name(&name, "{s}.attention.q_proj", .{prefix}), normalized, true), heads), sin, cos);
    const k = rotate(t.splitHeads(t.linear(w, l.name(&name, "{s}.attention.k_proj", .{prefix}), normalized, false), heads), sin, cos);
    const v = t.splitHeads(t.linear(w, l.name(&name, "{s}.attention.v_proj", .{prefix}), normalized, true), heads);
    const scores = q.dotGeneral(k, &.{.{ 3, 3 }}, &.{ .{ 0, 0 }, .{ 1, 1 } }).scale(@as(f32, 1 / @sqrt(@as(f32, depth))));
    const projected = t.linear(w, l.name(&name, "{s}.attention.o_proj", .{prefix}), t.attend(scores, v), true);
    const attention = x.add(projected.mul(w.get(prefix, ".layer_scale1.lambda1").broadcast(x.shape(), &.{2})));
    const norm = t.norm(w, l.name(&name, "{s}.norm2", .{prefix}), attention, eps);
    const mlp = t.gelu(t.linear(w, l.name(&name, "{s}.mlp.up_proj", .{prefix}), norm, true));
    const down = t.linear(w, l.name(&name, "{s}.mlp.down_proj", .{prefix}), mlp, true);
    return attention.add(down.mul(w.get(prefix, ".layer_scale2.lambda1").broadcast(x.shape(), &.{2})));
}

/// ImageNet-normalized [1,3,H,W] -> LocalMesh conditioning [1,5+HW/256,1024].
pub fn forward(parameters: []const Tensor, image: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const patches = l.conv(w, "embeddings.patch_embeddings", image, patch_size, @splat(0), true);
    const rows = patches.dim(2);
    const cols = patches.dim(3);
    const flat = patches.transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, rows * cols, channels });
    var x = Tensor.concatenate(&.{
        w.get("embeddings.cls_token", ""), w.get("embeddings.register_tokens", ""), flat,
    }, 1);
    const angle = angles(rows, cols);
    const sin = angle.sin().reshape(.{ rows * cols + prefix_tokens, depth });
    const cos = angle.cos().reshape(.{ rows * cols + prefix_tokens, depth });
    for (0..layers) |i| x = block(w, x, i, sin, cos);
    // LocalMesh projects the blocks' final residual with an affine-free LayerNorm.
    return t.normAffineFree(x, eps);
}
