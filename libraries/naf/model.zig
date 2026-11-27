const std = @import("std");
const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const Tensor = zml.Tensor;

pub const channels = 256;
pub const heads = 4;
pub const window = 9;
const head_width = channels / heads;
const branch_width = channels / 2;

pub const schema = makeSchema();

fn makeSchema() [37]l.Parameter {
    @setEvalBranchQuota(100_000);
    var result: [37]l.Parameter = undefined;
    var offset: usize = 0;
    inline for (.{ "encoder", "sem_encoder" }, .{ 1, 3 }) |branch_name, kernel| {
        const prefix = "image_encoder." ++ branch_name;
        result[offset] = .{ .name = prefix ++ ".0.weight", .shape = .init(.{ branch_width, 3, kernel, kernel }, .f32) };
        result[offset + 1] = .{ .name = prefix ++ ".0.bias", .shape = .init(.{branch_width}, .f32) };
        offset += 2;
        inline for (1..3) |block| {
            inline for (1..3) |part| {
                const norm = std.fmt.comptimePrint("{s}.{d}.norm{d}", .{ prefix, block, part });
                const conv = std.fmt.comptimePrint("{s}.{d}.conv{d}", .{ prefix, block, part });
                result[offset] = .{ .name = norm ++ ".weight", .shape = .init(.{branch_width}, .f32) };
                result[offset + 1] = .{ .name = norm ++ ".bias", .shape = .init(.{branch_width}, .f32) };
                result[offset + 2] = .{ .name = conv ++ ".weight", .shape = .init(.{ branch_width, branch_width, kernel, kernel }, .f32) };
                result[offset + 3] = .{ .name = conv ++ ".bias", .shape = .init(.{branch_width}, .f32) };
                offset += 4;
            }
        }
    }
    result[offset] = .{ .name = "image_encoder.rope.periods", .shape = .init(.{head_width / 4}, .f32) };
    return result;
}

fn branch(w: l.Weights, comptime prefix: []const u8, x: Tensor, comptime reflect: bool) Tensor {
    var y = l.conv(w, prefix ++ ".0", if (reflect) l.reflectOne(x) else x, 1, @splat(0), true);
    for (1..3) |block| {
        for (1..3) |part| {
            var norm_name: [64]u8 = undefined;
            var conv_name: [64]u8 = undefined;
            const norm = l.groupNorm(w, l.name(&norm_name, "{s}.{d}.norm{d}", .{ prefix, block, part }), y, 8, 1e-5);
            const active = norm.mul(norm.sigmoid());
            y = l.conv(w, l.name(&conv_name, "{s}.{d}.conv{d}", .{ prefix, block, part }), if (reflect) l.reflectOne(active) else active, 1, @splat(0), true);
        }
    }
    return y;
}

fn floatSize(value: i64) f32 {
    return @floatFromInt(value);
}

fn pool(x: Tensor, rows: i64, cols: i64) Tensor {
    const row_stride = @divExact(x.dim(2), rows);
    const col_stride = @divExact(x.dim(3), cols);
    return x.reshape(.{ 1, x.dim(1), rows, row_stride, cols, col_stride }).mean(3).mean(5)
        .reshape(.{ 1, x.dim(1), rows, cols });
}

fn phases(shape: zml.Shape, axis: u3, size: i64, periods: Tensor) Tensor {
    const positions = Tensor.iota(shape.withDtype(.i32), axis).convert(.f32)
        .addConstant(@as(f32, 0.5)).divByConst(floatSize(size)).scale(@as(f32, 2)).subConstant(@as(f32, 1));
    return positions.scale(@as(f32, 2 * std.math.pi)).div(periods.broadcast(shape, &.{2}));
}

fn encode(parameters: []const Tensor, image: Tensor, comptime stride: i64) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const pixel = branch(w, "image_encoder.encoder", image, false);
    const spatial = branch(w, "image_encoder.sem_encoder", image, true);
    const x = Tensor.concatenate(&.{ pixel, spatial }, 1);
    const rows = @divExact(x.dim(2), stride);
    const cols = @divExact(x.dim(3), stride);
    const pooled = if (stride == 1) x else pool(x, rows, cols);
    const periods = w.get("image_encoder.rope", ".periods");
    const phase_shape: zml.Shape = .init(.{ rows, cols, head_width / 4 }, .f32);
    const axes = Tensor.concatenate(&.{ phases(phase_shape, 0, rows, periods), phases(phase_shape, 1, cols, periods) }, 2);
    const angle = Tensor.concatenate(&.{ axes, axes }, 2).reshape(.{ rows * cols, head_width });
    const split = pooled.reshape(.{ heads, head_width, rows * cols }).transpose(.{ 0, 2, 1 });
    const first = split.slice(2, .{ .end = head_width / 2 });
    const second = split.slice(2, .{ .start = head_width / 2 });
    const quarter_turn = Tensor.concatenate(&.{ second.negate(), first }, 2);
    const rotated = split.mul(angle.cos().broadcast(split.shape(), &.{ 1, 2 }))
        .add(quarter_turn.mul(angle.sin().broadcast(split.shape(), &.{ 1, 2 })));
    return rotated.transpose(.{ 0, 2, 1 }).reshape(pooled.shape());
}

pub fn encodeSame(parameters: []const Tensor, image: Tensor) Tensor {
    return encode(parameters, image, 1);
}

pub fn encodeHalf(parameters: []const Tensor, image: Tensor) Tensor {
    return encode(parameters, image, 2);
}

fn lookup(input: Tensor, indices: Tensor, comptime two_axes: bool) Tensor {
    const ids = if (two_axes) indices.withTags(.{ .row, .neighbor }) else indices.withTags(.{.row});
    return input.withTags(.{ .site, .channel }).gather(.{ .site = ids }, .{});
}

fn table(x: Tensor) Tensor {
    return x.transpose(.{ 0, 2, 3, 1 }).reshape(.{ x.dim(2) * x.dim(3), x.dim(1) });
}

fn windowStart(pixels: Tensor, ratio: i64, side: i64) Tensor {
    return pixels.divByConst(@as(i32, @intCast(ratio))).subConstant(@as(i32, window / 2))
        .maximum(.scalar(@as(i32, 0), .i32)).minimum(.scalar(@as(i32, @intCast(side - window)), .i32));
}

fn attention(queries: Tensor, features: Tensor, y: Tensor, x: Tensor) Tensor {
    const height = queries.dim(2);
    const width = queries.dim(3);
    const rows = features.dim(2);
    const cols = features.dim(3);
    const count = y.dim(0);
    const depth = @divExact(features.dim(1), heads);
    const key = table(pool(queries, rows, cols));
    const start_y = windowStart(y, @divExact(height, rows), rows);
    const start_x = windowStart(x, @divExact(width, cols), cols);
    const grid: zml.Shape = .init(.{ count, window, window }, .i32);
    const row_index = start_y.broadcast(grid, &.{0}).add(Tensor.iota(grid, 1));
    const col_index = start_x.broadcast(grid, &.{0}).add(Tensor.iota(grid, 2));
    const neighbors = row_index.scale(@as(i32, @intCast(cols))).add(col_index).reshape(.{ count, window * window });
    const pixel_index = y.scale(@as(i32, @intCast(width))).add(x);
    const q = lookup(table(queries), pixel_index, false).reshape(.{ count, heads, head_width });
    const k = lookup(key, neighbors, true).reshape(.{ count, window * window, heads, head_width });
    const v = lookup(table(features), neighbors, true).reshape(.{ count, window * window, heads, depth });
    const logits = q.dotGeneral(k, &.{.{ 2, 3 }}, &.{ .{ 0, 0 }, .{ 1, 2 } }).scale(@as(f32, 1 / @sqrt(@as(f32, head_width))));
    return logits.softmax(2).dotGeneral(v, &.{.{ 2, 1 }}, &.{ .{ 0, 0 }, .{ 1, 2 } })
        .reshape(.{ count, features.dim(1) });
}

fn borderCoordinate(grid: Tensor, axis: u3, size: i64) Tensor {
    return grid.slice(1, .single(axis)).reshape(.{grid.dim(0)})
        .addConstant(@as(f32, 1)).scale(floatSize(size) / 2).subConstant(@as(f32, 0.5))
        .maximum(.scalar(@as(f32, 0), .f32)).minimum(.scalar(floatSize(size - 1), .f32));
}

pub fn sample(queries: Tensor, features: Tensor, grid: Tensor) Tensor {
    const height = queries.dim(2);
    const width = queries.dim(3);
    const count = grid.dim(0);
    const x = borderCoordinate(grid, 0, width);
    const y = borderCoordinate(grid, 1, height);
    const ix = x.floor().convert(.i32);
    const iy = y.floor().convert(.i32);
    const right = ix.addConstant(@as(i32, 1)).minimum(.scalar(@as(i32, @intCast(width - 1)), .i32));
    const bottom = iy.addConstant(@as(i32, 1)).minimum(.scalar(@as(i32, @intCast(height - 1)), .i32));
    const all_x = Tensor.concatenate(&.{ ix, right, ix, right }, 0);
    const all_y = Tensor.concatenate(&.{ iy, iy, bottom, bottom }, 0);
    const values = attention(queries, features, all_y, all_x).reshape(.{ 4, count, features.dim(1) });
    const dx = x.sub(ix.convert(.f32));
    const dy = y.sub(iy.convert(.f32));
    const left = dx.negate().addConstant(@as(f32, 1));
    const top = dy.negate().addConstant(@as(f32, 1));
    const weights = Tensor.concatenate(&.{ left.mul(top), dx.mul(top), left.mul(dy), dx.mul(dy) }, 0).reshape(.{ 4, count });
    return values.mul(weights.broadcast(values.shape(), &.{ 0, 1 })).sum(0).reshape(.{ count, features.dim(1) });
}
