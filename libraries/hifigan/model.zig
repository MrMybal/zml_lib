const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const Tensor = l.Tensor;

pub const mel_bands = 64;
pub const sample_rate = 16_000;
const strides = [_]i64{ 5, 4, 2, 2, 2 };
const up_kernels = [_]i64{ 16, 16, 8, 4, 4 };
const widths = [_]i64{ 1024, 512, 256, 128, 64, 32 };
const kernels = [_]i64{ 3, 7, 11 };
const dilations = [_]i64{ 1, 3, 5 };

fn param(name: []const u8, shape: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(shape, .f32) };
}

fn pair(comptime name: []const u8, dims: anytype, bias: i64) [2]l.Parameter {
    return .{ param(name ++ ".weight", dims), param(name ++ ".bias", .{bias}) };
}

const schema = blk: {
    @setEvalBranchQuota(75_000);
    var entries: [196]l.Parameter = undefined;
    entries[0..6].* = pair("conv_pre", .{ widths[0], mel_bands, 7 }, widths[0]) ++
        pair("conv_post", .{ 1, widths[5], 7 }, 1) ++ [_]l.Parameter{ param("mean", .{mel_bands}), param("scale", .{mel_bands}) };
    var offset: usize = 6;
    for (up_kernels, 0..) |size, i| {
        const c = widths[i + 1];
        const up = std.fmt.comptimePrint("upsampler.{d}", .{i});
        entries[offset..][0..2].* = pair(up, .{ widths[i], c, size }, c);
        offset += 2;
        for (kernels, 0..) |k, j| {
            for (0..2) |branch| {
                for (0..dilations.len) |leg| {
                    const name = std.fmt.comptimePrint("resblocks.{d}.convs{d}.{d}", .{ i * kernels.len + j, branch + 1, leg });
                    entries[offset..][0..2].* = pair(name, .{ c, c, k }, c);
                    offset += 2;
                }
            }
        }
    }
    break :blk entries;
};

fn lrelu(x: Tensor, coefficient: f32) Tensor {
    return x.maximum(x.scale(coefficient));
}

fn convolution(w: l.Weights, name: []const u8, x: Tensor, dilation: i64) Tensor {
    const kernel = w.get(name, ".weight");
    const padding = @divExact((kernel.dim(2) - 1) * dilation, 2);
    return l.biasAdd(x.conv1d(kernel, .{ .padding = &.{ padding, padding }, .rhs_dilation = dilation }), w.get(name, ".bias"), 1);
}

fn upsample(w: l.Weights, name: []const u8, x: Tensor, stride: i64) Tensor {
    const kernel = w.get(name, ".weight").transpose(.{ 1, 0, 2 }).reverse(.{2});
    const k = kernel.dim(2);
    const p = k - 1 - @divFloor(k - stride, 2);
    return l.biasAdd(x.conv1d(kernel, .{ .padding = &.{ p, p }, .lhs_dilation = stride }), w.get(name, ".bias"), 1);
}

fn residual(w: l.Weights, index: usize, input: Tensor) Tensor {
    var key: [64]u8 = undefined;
    var x = input;
    for (dilations, 0..) |dilation, i| {
        const a = convolution(w, l.name(&key, "resblocks.{d}.convs1.{d}", .{ index, i }), lrelu(x, 0.1), dilation);
        const b = convolution(w, l.name(&key, "resblocks.{d}.convs2.{d}", .{ index, i }), lrelu(a, 0.1), 1);
        x = x.add(b);
    }
    return x;
}

fn forward(parameters: []const Tensor, mel: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    var x = convolution(w, "conv_pre", mel.transpose(.{ 0, 2, 1 }), 1);
    for (strides, 0..) |stride, i| {
        var key: [32]u8 = undefined;
        const expanded = upsample(w, l.name(&key, "upsampler.{d}", .{i}), lrelu(x, 0.1), stride);
        const first = i * kernels.len;
        var sum = residual(w, first, expanded);
        for (1..kernels.len) |j| sum = sum.add(residual(w, first + j, expanded));
        x = sum.divByConst(@as(f32, kernels.len));
    }
    return convolution(w, "conv_post", lrelu(x, 0.01), 1).tanh().flatten();
}

pub fn sampleCount(frames: u32) !usize {
    if (frames == 0) return error.InvalidFrameCount;
    // The stride-five, even-width first upsampler produces one additional sample.
    return std.math.add(usize, try std.math.mul(usize, frames, 160), 32);
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn synthesize(self: *Model, frame_count: u32, mel: []const f32, waveform: []f32) !u32 {
        const samples: i64 = @intCast(try sampleCount(frame_count));
        return self.graph.infer(forward, .init(.{ 1, @as(i64, frame_count), mel_bands }, .f32), .init(.{samples}, .f32), mel, waveform);
    }
};
