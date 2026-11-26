const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
const Parameter = l.Parameter;
const Weights = l.Weights;

const channels = 512;
const aggregated = 3 * channels;
const split = 8;
const intermediate = 128;
const dilations = [_]i64{ 2, 3, 4 };
const prefix = "speaker_encoder.";

pub const max_frames = @divFloor(30 * 24_000, 256);

fn convolutionParameters(comptime name: []const u8, out: i64, in: i64, kernel: i64) [2]Parameter {
    return .{
        .{ .name = name ++ ".weight", .shape = .init(.{ out, in, kernel }, .f32) },
        .{ .name = name ++ ".bias", .shape = .init(.{out}, .f32) },
    };
}

fn project(w: Weights, name: []const u8, x: Tensor) Tensor {
    return t.linearParameters(x.transpose(.{ 0, 2, 1 }), w.get(name, ".weight").squeeze(2), w.get(name, ".bias"))
        .transpose(.{ 0, 2, 1 });
}

fn relu(x: Tensor) Tensor {
    return x.maximum(Tensor.scalar(0, .f32).broad(x.shape()));
}

fn tdnn(w: Weights, name: []const u8, x: Tensor, dilation: i64) Tensor {
    const kernel = w.get(name, ".weight");
    const padding = @divExact((kernel.dim(2) - 1) * dilation, 2);
    const time = x.dim(2);
    const left = x.slice(2, .{ .start = 1, .end = 1 + padding }).reverse(.{2});
    const right = x.slice(2, .{ .start = time - padding - 1, .end = time - 1 }).reverse(.{2});
    const samples = Tensor.concatenate(&.{ left, x, right }, 2);
    const y = samples.conv1d(kernel, .{ .rhs_dilation = dilation });
    return relu(l.biasAdd(y, w.get(name, ".bias"), 1));
}

fn residualBlock(w: Weights, comptime index: usize, x: Tensor) Tensor {
    const name = std.fmt.comptimePrint(prefix ++ "blocks.{d}.", .{index + 1});
    const hidden = relu(project(w, name ++ "tdnn1.conv", x));
    const width = channels / split;
    var groups: [split]Tensor = undefined;
    groups[0] = hidden.slice(1, .{ .end = width });
    for (1..split) |i| {
        var key: [96]u8 = undefined;
        const part = hidden.slice(1, .{ .start = @as(i64, @intCast(i)) * width, .end = @as(i64, @intCast(i + 1)) * width });
        const residual = if (i == 1) part else part.add(groups[i - 1]);
        const layer = l.name(&key, name ++ "res2net_block.blocks.{d}.conv", .{i - 1});
        groups[i] = tdnn(w, layer, residual, dilations[index]);
    }
    const mixed = relu(project(w, name ++ "tdnn2.conv", Tensor.concatenate(&groups, 1)));
    const se = relu(project(w, name ++ "se_block.conv1", mixed.mean(2)));
    const gate = project(w, name ++ "se_block.conv2", se).sigmoid();
    return x.add(mixed.mul(gate.broad(mixed.shape())));
}

const Statistics = struct { mean: Tensor, stddev: Tensor };

fn stats(x: Tensor, weight: Tensor) Statistics {
    const mean = x.mul(weight).sum(2);
    const centered = x.sub(mean.broad(x.shape()));
    const variance = centered.mul(centered).mul(weight).sum(2);
    return .{
        .mean = mean,
        .stddev = variance.maximum(Tensor.scalar(1e-12, .f32).broad(variance.shape())).sqrt(),
    };
}

fn pool(w: Weights, x: Tensor) Tensor {
    const uniform = 1 / @as(f32, @floatFromInt(x.dim(2)));
    const global = stats(x, Tensor.scalar(uniform, .f32).broad(x.shape()));
    const context = Tensor.concatenate(&.{ x, global.mean.broad(x.shape()), global.stddev.broad(x.shape()) }, 1);
    const features = relu(project(w, prefix ++ "asp.tdnn.conv", context)).tanh();
    const weights = project(w, prefix ++ "asp.conv", features).softmax(2);
    const weighted = stats(x, weights);
    return Tensor.concatenate(&.{ weighted.mean, weighted.stddev }, 1);
}

pub fn Graph(comptime width: usize) type {
    return struct {
        pub const input_width = 128;
        pub const output_width = width;

        pub const schema = blk: {
            @setEvalBranchQuota(40_000);
            var parameters: [76]Parameter = undefined;
            parameters[0..2].* = convolutionParameters(prefix ++ "blocks.0.conv", channels, input_width, 5);
            var i: usize = 2;
            for (1..4) |layer| {
                const name = std.fmt.comptimePrint(prefix ++ "blocks.{d}.", .{layer});
                parameters[i..][0..2].* = convolutionParameters(name ++ "tdnn1.conv", channels, channels, 1);
                i += 2;
                for (0..split - 1) |group| {
                    const key = std.fmt.comptimePrint(name ++ "res2net_block.blocks.{d}.conv", .{group});
                    parameters[i..][0..2].* = convolutionParameters(key, channels / split, channels / split, 3);
                    i += 2;
                }
                parameters[i..][0..2].* = convolutionParameters(name ++ "tdnn2.conv", channels, channels, 1);
                parameters[i + 2 ..][0..2].* = convolutionParameters(name ++ "se_block.conv1", intermediate, channels, 1);
                parameters[i + 4 ..][0..2].* = convolutionParameters(name ++ "se_block.conv2", channels, intermediate, 1);
                i += 6;
            }
            parameters[i..][0..2].* = convolutionParameters(prefix ++ "mfa.conv", aggregated, aggregated, 1);
            parameters[i + 2 ..][0..2].* = convolutionParameters(prefix ++ "asp.tdnn.conv", intermediate, aggregated * 3, 1);
            parameters[i + 4 ..][0..2].* = convolutionParameters(prefix ++ "asp.conv", aggregated, intermediate, 1);
            parameters[i + 6 ..][0..2].* = convolutionParameters(prefix ++ "fc", @intCast(width), 2 * aggregated, 1);
            break :blk parameters;
        };

        pub fn check(input: []const f32, count: usize) !void {
            if (count <= 4 or count > max_frames or input.len != count * input_width) return error.InvalidReferenceLength;
            for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteReferenceFeatures;
        }

        pub fn outputRows(_: usize) usize {
            return 1;
        }

        pub fn forward(parameters: []const Tensor, input: Tensor) Tensor {
            const w: Weights = .{ .schema = &schema, .tensors = parameters };
            var x = tdnn(w, prefix ++ "blocks.0.conv", input.transpose(.{ 1, 0 }).reshape(.{ 1, input_width, input.dim(0) }), 1);
            var features: [dilations.len]Tensor = undefined;
            inline for (0..dilations.len) |i| {
                x = residualBlock(w, i, x);
                features[i] = x;
            }
            const merged = relu(project(w, prefix ++ "mfa.conv", Tensor.concatenate(&features, 1)));
            return project(w, prefix ++ "fc", pool(w, merged)).reshape(.{ 1, width });
        }
    };
}
