const zml = @import("zml/core");
const rt = @import("zml_lib/runtime");
const l = rt.layers;
const t = rt.transformer;
const Tensor = zml.Tensor;

fn pixel(x: Tensor, vy: Tensor, vx: Tensor, h: i64, w: i64) Tensor {
    const y = vy.convert(.i32);
    const col = vx.convert(.i32);
    const zero = Tensor.scalar(0, .i32);
    const valid = y.cmp(.GE, zero).logical(.AND, y.cmp(.LT, .scalar(h, .i32)))
        .logical(.AND, col.cmp(.GE, zero)).logical(.AND, col.cmp(.LT, .scalar(w, .i32)));
    const rows = y.scale(@as(i32, @intCast(w))).add(col);
    const safe = valid.select(rows, zero.broad(rows.shape()));
    const gathered = x.withTags(.{ .pixel, .channel }).gather(.{ .pixel = safe.withTags(.{.query}) }, .{})
        .reshape(.{ y.dim(0), x.dim(1) });
    return gathered.mul(valid.convert(.f32).broadcast(gathered.shape(), &.{0}));
}

fn bilinear(x: Tensor, y: Tensor, col: Tensor, h: i64, w: i64) Tensor {
    const y0 = y.floor();
    const x0 = col.floor();
    const y1 = y0.addConstant(@as(f32, 1));
    const x1 = x0.addConstant(@as(f32, 1));
    const y_frac = y.sub(y0);
    const x_frac = col.sub(x0);
    const y_inv = y1.sub(y);
    const x_inv = x1.sub(col);
    const s = x.shape();
    return pixel(x, y0, x0, h, w).mul(y_inv.mul(x_inv).broadcast(s, &.{0}))
        .add(pixel(x, y0, x1, h, w).mul(y_inv.mul(x_frac).broadcast(s, &.{0})))
        .add(pixel(x, y1, x0, h, w).mul(y_frac.mul(x_inv).broadcast(s, &.{0})))
        .add(pixel(x, y1, x1, h, w).mul(y_frac.mul(x_frac).broadcast(s, &.{0})));
}

const Convolution = struct {
    features: Tensor,
    offsets: Tensor,
    modulation: Tensor,
    filters: Tensor,
    h: i64,
    w: i64,
    size: i64,

    pub const State = struct { index: Tensor, output: Tensor };

    pub fn cond(self: Convolution, state: State) Tensor {
        return state.index.cmp(.LT, .scalar(self.size * self.size, .i32));
    }

    pub fn body(self: Convolution, state: State) State {
        const id = state.index;
        const k = Tensor.scalar(self.size, .i32);
        const ky = id.div(k);
        const kx = id.sub(ky.mul(k));
        const shape: zml.Shape = .init(.{ self.h, self.w }, .i32);
        const rows = self.h * self.w;
        const position_y = Tensor.iota(shape, 0).reshape(.{rows}).convert(.f32);
        const position_x = Tensor.iota(shape, 1).reshape(.{rows}).convert(.f32);
        const offset = self.offsets.slice(1, .dynSingle(id)).reshape(.{ rows, 2 });
        const pad: i32 = @intCast(@divFloor(self.size, 2));
        const y = position_y.add(ky.subConstant(pad).convert(.f32).broad(position_y.shape()))
            .add(offset.slice(1, .single(0)).reshape(.{rows}));
        const x = position_x.add(kx.subConstant(pad).convert(.f32).broad(position_x.shape()))
            .add(offset.slice(1, .single(1)).reshape(.{rows}));
        const sample = bilinear(self.features, y, x, self.h, self.w);
        const mask = self.modulation.slice(1, .dynSingle(id)).reshape(.{rows}).broadcast(sample.shape(), &.{0});
        const weights = self.filters.slice(0, .dynSingle(id)).reshape(.{ self.filters.dim(1), self.filters.dim(2) });
        return .{
            .index = id.addConstant(@as(i32, 1)),
            .output = state.output.add(t.linearParameters(sample.mul(mask), weights, null)),
        };
    }
};

fn apply(w: l.Weights, x: Tensor, key: []const u8, size: i64) Tensor {
    var buffer: [128]u8 = undefined;
    const pad = @divFloor(size, 2);
    const taps = size * size;
    const rows = x.dim(2) * x.dim(3);
    const offsets = l.conv(w, l.name(&buffer, "{s}.offset_conv", .{key}), x, 1, @splat(pad), true)
        .transpose(.{ 0, 2, 3, 1 }).reshape(.{ rows, taps, 2 });
    const scale = l.conv(w, l.name(&buffer, "{s}.modulator_conv", .{key}), x, 1, @splat(pad), true).sigmoid()
        .scale(@as(f32, 2)).transpose(.{ 0, 2, 3, 1 }).reshape(.{ rows, taps });
    const filters = w.get(l.name(&buffer, "{s}.regular_conv", .{key}), ".weight");
    const out = filters.dim(0);
    const computation = Convolution{
        .features = x.transpose(.{ 0, 2, 3, 1 }).reshape(.{ rows, x.dim(1) }),
        .offsets = offsets,
        .modulation = scale,
        .filters = filters.transpose(.{ 2, 3, 0, 1 }).reshape(.{ taps, out, x.dim(1) }),
        .h = x.dim(2),
        .w = x.dim(3),
        .size = size,
    };
    const result = zml.ops.@"while"(Convolution, computation, .{
        .index = .scalar(0, .i32),
        .output = .zeroes(.init(.{ rows, out }, .f32)),
    });
    return result.output.reshape(.{ 1, x.dim(2), x.dim(3), out }).transpose(.{ 0, 3, 1, 2 });
}

pub fn branch(w: l.Weights, x: Tensor, scope: []const u8, size: i64) Tensor {
    var key: [100]u8 = undefined;
    const out = apply(w, x, l.name(&key, "{s}.atrous_conv", .{scope}), size);
    return l.batchNorm(w, l.name(&key, "{s}.bn", .{scope}), out).relu();
}
