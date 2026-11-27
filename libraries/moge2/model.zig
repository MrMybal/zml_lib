const rt = @import("zml_lib/runtime");
const dino = @import("zml_lib/dinov2");
const wdef = @import("weights.zig");
const l = rt.layers;
const t = rt.transformer;
const Tensor = l.Tensor;

pub const Grid = struct {
    rows: i64,
    cols: i64,
};

fn duplicateBorder(input: Tensor, comptime axis: u3) Tensor {
    const last = input.dim(axis) - 1;
    return Tensor.concatenate(&.{
        input.slice(axis, .{ .end = 1 }), input, input.slice(axis, .{ .start = last }),
    }, axis);
}

fn strideOne(w: l.Weights, name: []const u8, x: Tensor) Tensor {
    return l.conv(w, name, duplicateBorder(duplicateBorder(x, 2), 3), 1, @splat(0), true);
}

fn plane(rows: i64, cols: i64, height: i64, width: i64) Tensor {
    const shape = @TypeOf(Tensor.scalar(0, .f32).shape()).init(.{ 1, 1, rows, cols }, .f32);
    const h: f32 = @floatFromInt(height);
    const w: f32 = @floatFromInt(width);
    const diag = @sqrt(h * h + w * w);
    const u = Tensor.iota(shape.withDtype(.i32), 3).convert(.f32).addConstant(@as(f32, 0.5))
        .scale(2 / @as(f32, @floatFromInt(cols))).subConstant(@as(f32, 1)).scale(w / diag);
    const v = Tensor.iota(shape.withDtype(.i32), 2).convert(.f32).addConstant(@as(f32, 0.5))
        .scale(2 / @as(f32, @floatFromInt(rows))).subConstant(@as(f32, 1)).scale(h / diag);
    return Tensor.concatenate(&.{ u, v }, 1);
}

fn stack(w: l.Weights, comptime scope: []const u8, input: [5]Tensor, comptime neck: bool) [5]Tensor {
    var x: Tensor = undefined;
    var result: [5]Tensor = undefined;
    var name: [80]u8 = undefined;
    for (input, 0..) |level_input, level| {
        const transformed = l.conv(w, l.name(&name, "{s}.input_blocks.{d}", .{ scope, level }), level_input, 1, @splat(0), true);
        x = if (level == 0) transformed else x.add(transformed);
        if (level > 0 and level < 4) {
            for (0..if (neck) 2 else 1) |block| {
                const hidden = strideOne(w, l.name(&name, "{s}.res_blocks.{d}.{d}.layers.2", .{ scope, level, block }), x.relu());
                const update = strideOne(w, l.name(&name, "{s}.res_blocks.{d}.{d}.layers.5", .{ scope, level, block }), hidden.relu());
                x = x.add(update);
            }
        }
        result[level] = x;
        if (level < 4) {
            const up = if (level < 3)
                t.deconv2x(w, l.name(&name, "{s}.resamplers.{d}.0", .{ scope, level }), x)
            else
                rt.spatial.bilinearHalfPixel(x, 2 * x.dim(2), 2 * x.dim(3));
            x = strideOne(w, l.name(&name, "{s}.resamplers.{d}.1", .{ scope, level }), up);
        }
    }
    return result;
}

fn head(w: l.Weights, comptime scope: []const u8, input: [5]Tensor, h: i64, cols: i64) Tensor {
    const decoded = stack(w, scope, input, false);
    const channels = l.conv(w, scope ++ ".output_blocks.4", decoded[4], 1, @splat(0), true);
    return rt.spatial.bilinearHalfPixel(channels, h, cols);
}

fn vector(x: Tensor) Tensor {
    const dims = x.dim(1) * x.dim(2) * x.dim(3);
    return x.transpose(.{ 0, 2, 3, 1 }).reshape(.{dims});
}

pub fn Graph(comptime with_normals: bool, comptime normals_only: bool) type {
    return struct {
        pub const parameters = wdef.schema(with_normals);

        pub fn elements(height: i64, width: i64) i64 {
            return if (normals_only) 3 * height * width else (if (with_normals) @as(i64, 7) else @as(i64, 4)) * height * width + 1;
        }

        pub fn forward(params: []const Tensor, image: Tensor, grid: Grid) Tensor {
            const w: l.Weights = .{ .schema = &parameters, .tensors = params };
            const h = image.dim(2);
            const cols = image.dim(3);
            const scaled = rt.spatial.bilinearAntialias(image, grid.rows * 14, grid.cols * 14);
            const normalized = scaled.sub(w.get("encoder.image_mean", "").broad(scaled.shape()))
                .div(w.get("encoder.image_std", "").broad(scaled.shape()));
            const features = dino.encode(wdef.backbone, params[0..wdef.backbone_schema.len], normalized);
            var projected: Tensor = undefined;
            var name: [56]u8 = undefined;
            for (features, 0..) |feature, idx| {
                const patches = feature.slice(1, .{ .start = 1 }).transpose(.{ 0, 2, 1 })
                    .reshape(.{ 1, 1024, grid.rows, grid.cols });
                const x = l.conv(w, l.name(&name, "encoder.output_projections.{d}", .{idx}), patches, 1, @splat(0), true);
                projected = if (idx == 0) x else projected.add(x);
            }
            var with_coordinates: [5]Tensor = undefined;
            for (&with_coordinates, 0..) |*entry, i| {
                const ratio = @as(i64, 1) << @as(u6, @intCast(i));
                const uv = plane(grid.rows * ratio, grid.cols * ratio, h, cols);
                entry.* = if (i == 0) Tensor.concatenate(&.{ projected, uv }, 1) else uv;
            }
            const neck = stack(w, "neck", with_coordinates, true);
            const normals = if (with_normals) value: {
                const raw = head(w, "normal_head", neck, h, cols);
                const length = raw.mul(raw).sum(1).sqrt().maximum(.scalar(@as(f32, 1e-12), .f32));
                break :value vector(raw.div(length.broad(raw.shape())));
            } else {};
            if (normals_only) return normals;

            const points = head(w, "points_head", neck, h, cols);
            const depth = points.slice(1, .{ .start = 2 }).exp();
            const xyz = Tensor.concatenate(&.{
                points.slice(1, .{ .end = 2 }).mul(depth.broad(points.shape().setDim(1, 2))), depth,
            }, 1);
            const mask = vector(head(w, "mask_head", neck, h, cols).sigmoid());
            var cls = features[3].slice(1, .single(0)).reshape(.{ 1, 1024 });
            cls = t.linear(w, "scale_head.0", cls, true).relu();
            cls = t.linear(w, "scale_head.2", cls, true).relu();
            const scale = t.linear(w, "scale_head.4", cls, true).exp().reshape(.{1});
            return if (with_normals)
                Tensor.concatenate(&.{ vector(xyz), mask, normals, scale }, 0)
            else
                Tensor.concatenate(&.{ vector(xyz), mask, scale }, 0);
        }
    };
}
