const std = @import("std");
const zml = @import("zml/core");
const stablehlo = @import("mlir/dialects/stablehlo");
pub const Tensor = zml.Tensor;

pub const Parameter = struct {
    name: []const u8,
    shape: zml.Shape,

    pub fn at(schema: []const Parameter, index: usize, _: []u8) Parameter {
        return schema[index];
    }
};

pub const Weights = struct {
    schema: []const Parameter,
    tensors: []const Tensor,

    pub fn get(self: Weights, prefix: []const u8, suffix: []const u8) Tensor {
        for (self.schema, self.tensors) |parameter, tensor| {
            if (parameter.name.len == prefix.len + suffix.len and
                std.mem.startsWith(u8, parameter.name, prefix) and
                std.mem.endsWith(u8, parameter.name, suffix)) return tensor;
        }
        std.debug.panic("Missing graph parameter {s}{s}", .{ prefix, suffix });
    }
};

pub fn name(buffer: []u8, comptime format: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buffer, format, args) catch unreachable;
}

pub fn biasAdd(x: Tensor, bias: Tensor, axis: i64) Tensor {
    return x.add(bias.broadcast(x.shape(), &.{axis}));
}

pub fn conv(w: Weights, name_: []const u8, x: Tensor, stride: i64, padding: [4]i64, has_bias: bool) Tensor {
    return convGroups(w, name_, x, stride, padding, has_bias, 1);
}

pub fn convGroups(w: Weights, name_: []const u8, x: Tensor, stride: i64, padding: [4]i64, has_bias: bool, groups: i64) Tensor {
    const output = x.conv2d(w.get(name_, ".weight"), .{
        .window_strides = &.{ stride, stride },
        .padding = &padding,
        .feature_group_count = groups,
    });
    const ctx = zml.Compiler.current().mlir_ctx;
    const high = stablehlo.precisionAttr(ctx, .HIGHEST);
    output.value().owner().setAttributeByName("precision_config", .array(ctx, &.{ high, high }));
    return if (has_bias) biasAdd(output, w.get(name_, ".bias"), 1) else output;
}

pub fn batchNorm(w: Weights, name_: []const u8, x: Tensor) Tensor {
    return batchNormEpsilon(w, name_, x, 1e-5);
}

pub fn batchNormEpsilon(w: Weights, name_: []const u8, x: Tensor, eps: f32) Tensor {
    const mean = w.get(name_, ".running_mean").broadcast(x.shape(), &.{1});
    const variance = w.get(name_, ".running_var");
    const scale = w.get(name_, ".weight").mul(variance.addConstant(eps).rsqrt());
    return biasAdd(x.sub(mean).mul(scale.broadcast(x.shape(), &.{1})), w.get(name_, ".bias"), 1);
}

fn normalize(x: Tensor, axis: u3, epsilon: f32) Tensor {
    const centered = x.sub(x.mean(axis).broad(x.shape()));
    const variance = centered.mul(centered).mean(axis);
    return centered.mul(variance.addConstant(epsilon).rsqrt().broad(x.shape()));
}

pub fn groupNorm(w: Weights, name_: []const u8, x: Tensor, groups: i64, epsilon: f32) Tensor {
    const grouped = x.reshape(.{ x.dim(0), groups, @divExact(x.dim(1), groups) * x.dim(2) * x.dim(3) });
    const y = normalize(grouped, 2, epsilon).reshape(x.shape());
    return biasAdd(y.mul(w.get(name_, ".weight").broadcast(x.shape(), &.{1})), w.get(name_, ".bias"), 1);
}

pub fn layerNorm(w: Weights, name_: []const u8, x: Tensor) Tensor {
    const y = normalize(x, 1, 1e-5);
    return biasAdd(y.mul(w.get(name_, ".weight").broadcast(x.shape(), &.{1})), w.get(name_, ".bias"), 1);
}

pub fn linear(w: Weights, name_: []const u8, x: Tensor, has_bias: bool) Tensor {
    const y = x.dotGeneral(w.get(name_, ".weight"), &.{.{ 1, 1 }}, &.{});
    return if (has_bias) biasAdd(y, w.get(name_, ".bias"), 1) else y;
}

pub fn leakyRelu(x: Tensor) Tensor {
    return x.maximum(x.scale(@as(f32, 0.2)));
}

pub fn nearest2x(x: Tensor) Tensor {
    const n, const c, const h, const w = x.dims()[0..4].*;
    return x.broadcast(.init(.{ n, c, h, 2, w, 2 }, x.dtype()), &.{ 0, 1, 2, 4 })
        .reshape(.{ n, c, 2 * h, 2 * w });
}

pub fn reflectOne(x: Tensor) Tensor {
    var result = x;
    for (2..4) |axis| {
        const i: u3 = @intCast(axis);
        const d = result.dim(i);
        const first = result.slice(i, .{ .start = 1, .end = 2 });
        const last = result.slice(i, .{ .start = d - 2, .end = d - 1 });
        result = Tensor.concatenate(&.{ first, result, last }, i);
    }
    return result;
}
