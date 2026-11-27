const zml = @import("zml/core");
const std = @import("std");
const rt = @import("zml_lib/runtime");
const wd = @import("weights.zig");
const l = rt.layers;
const t = rt.transformer;
const Tensor = zml.Tensor;
pub const Profile = wd.Profile;

fn kernel(weight: Tensor) Tensor {
    return if (weight.dtype() == .u8) t.decodeE4M3(weight, 1) else weight;
}

fn dense(w: l.Weights, key: []const u8, x: Tensor) Tensor {
    return t.linearParameters(x, kernel(w.get(key, ".weight")), w.get(key, ".bias"));
}

fn convSlice(x: Tensor, map: Tensor, weight: Tensor, bias: Tensor) Tensor {
    const gathered = x.withTags(.{ .voxel, .channel })
        .gather(.{ .voxel = map.withTags(.{ .target, .tap }) }, .{})
        .reshape(.{ map.dim(0), 27 * x.dim(1) });
    return t.linearParameters(gathered, weight, bias);
}

const ConvolutionTiles = struct {
    x: Tensor,
    map: Tensor,
    weight: Tensor,
    bias: Tensor,

    const width = 512;
    pub const State = struct { offset: Tensor, output: Tensor };

    pub fn cond(self: ConvolutionTiles, state: State) Tensor {
        return state.offset.cmp(.LT, .scalar(self.map.dim(0), .i32));
    }

    pub fn body(self: ConvolutionTiles, state: State) State {
        const values = convSlice(self.x, self.map.slice(0, .dyn(state.offset, width)), self.weight, self.bias);
        return .{
            .offset = state.offset.addConstant(@as(i32, width)),
            .output = state.output.dynamicUpdateSlice(.{ state.offset, Tensor.scalar(0, .i32) }, values),
        };
    }
};

fn convolution(w: l.Weights, key: []const u8, x: Tensor, neighbors: Tensor) Tensor {
    const padded = Tensor.concatenate(&.{ x, Tensor.zeroes(x.shape().setDim(0, 1)) }, 0);
    const weight = kernel(w.get(key, ".weight"));
    const bias = w.get(key, ".bias");
    const rows = x.dim(0);
    const tile = ConvolutionTiles.width;
    if (rows <= tile) return convSlice(padded, neighbors, weight, bias);
    const length = @divTrunc(rows + tile - 1, tile) * tile;
    const map = if (rows == length) neighbors else Tensor.concatenate(&.{
        neighbors, Tensor.zeroes(neighbors.shape().setDim(0, length - rows)),
    }, 0);
    const loop = zml.ops.@"while"(ConvolutionTiles, .{ .x = padded, .map = map, .weight = weight, .bias = bias }, .{
        .offset = .scalar(0, .i32),
        .output = .zeroes(.init(.{ length, weight.dim(0) }, .f32)),
    });
    return loop.output.slice(0, .{ .end = rows });
}

pub fn Stage(comptime profile: Profile, comptime level: usize, comptime post: bool) type {
    const width = profile.channels(level);
    const count = profile.blocks(level);
    const scope = wd.resolutionScope(level, count);
    const projected = if (level == 4) 0 else profile.channels(level + 1);
    const conv_width = if (level == 4) 0 else if (profile.encoding()) @divExact(projected, 8) else projected * 8;
    return struct {
        pub const parameters = wd.phaseSchema(profile, level, post);
        pub const input_width = if (post)
            projected + (if (profile.encoding()) width * 8 else @divExact(width, 8))
        else if (level == 0)
            (if (profile.encoding()) @as(i64, 6) else @as(i64, 32))
        else
            width;
        pub const output_width = if (post) projected else if (level == 4)
            profile.finalWidth()
        else
            conv_width + width + @as(i64, if (profile == .shape_decode) 8 else 0);

        pub fn forward(params: []const Tensor, input: Tensor) Tensor {
            const w: l.Weights = .{ .schema = &parameters, .tensors = params };
            const neighbors = input.slice(1, .{ .start = input_width }).convert(.i32);
            var x = input.slice(1, .{ .end = input_width });
            if (post) {
                const features = x.slice(1, .{ .end = projected });
                const skip = x.slice(1, .{ .start = projected });
                const reshaped = if (profile.encoding())
                    skip.reshape(.{ x.dim(0), projected, @divExact(width * 8, projected) }).mean(2).reshape(features.shape())
                else
                    skip.reshape(.{ x.dim(0), @divExact(width, 8), 1 })
                        .broad(.init(.{ x.dim(0), @divExact(width, 8), @divExact(projected * 8, width) }, .f32))
                        .reshape(features.shape());
                return convolution(w, scope ++ ".conv2", t.normAffineFree(features, 1e-6).silu(), neighbors).add(reshaped);
            }
            if (level == 0) x = dense(w, if (profile.encoding()) "input_layer" else "from_latent", x);

            var base: [32]u8 = undefined;
            var name: [64]u8 = undefined;
            for (0..count) |i| {
                const key = l.name(&base, "blocks.{d}.{d}", .{ level, i });
                const spatial = convolution(w, l.name(&name, "{s}.conv", .{key}), x, neighbors);
                const normed = t.norm(w, l.name(&name, "{s}.norm", .{key}), spatial, 1e-6);
                const expanded = dense(w, l.name(&name, "{s}.mlp.0", .{key}), normed).silu();
                x = x.add(dense(w, l.name(&name, "{s}.mlp.2", .{key}), expanded));
            }
            if (level == 4) return dense(w, if (profile.encoding()) "to_latent" else "output_layer", t.normAffineFree(x, 1e-5));

            const updates = convolution(w, scope ++ ".conv1", t.norm(w, scope ++ ".norm1", x, 1e-6).silu(), neighbors);
            return if (profile == .shape_decode)
                Tensor.concatenate(&.{ updates, x, dense(w, scope ++ ".to_subdiv", x) }, 1)
            else
                Tensor.concatenate(&.{ updates, x }, 1);
        }
    };
}
