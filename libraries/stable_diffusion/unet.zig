//! Stable Diffusion v1.5 U-Net using CompVis weight names.
const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;

pub const channels = 4;
pub const context_length = 77;
pub const context_width = 768;
const widths = [_]i64{ 320, 640, 1280, 1280 };
const epsilon: f32 = 1e-5;

const Schema = struct {
    items: [686]l.Parameter = undefined,
    len: usize = 0,

    fn parameter(self: *Schema, comptime name: []const u8, dims: anytype) void {
        self.items[self.len] = .{ .name = name, .shape = .init(dims, .f32) };
        self.len += 1;
    }

    fn affine(self: *Schema, comptime name: []const u8, dims: anytype, out: i64) void {
        self.parameter(name ++ ".weight", dims);
        self.parameter(name ++ ".bias", .{out});
    }

    fn resnet(self: *Schema, comptime name: []const u8, input: i64, out: i64) void {
        self.affine(name ++ ".in_layers.0", .{input}, input);
        self.affine(name ++ ".in_layers.2", .{ out, input, 3, 3 }, out);
        self.affine(name ++ ".emb_layers.1", .{ out, 1280 }, out);
        self.affine(name ++ ".out_layers.0", .{out}, out);
        self.affine(name ++ ".out_layers.3", .{ out, out, 3, 3 }, out);
        if (input != out) self.affine(name ++ ".skip_connection", .{ out, input, 1, 1 }, out);
    }

    fn attention(self: *Schema, comptime name: []const u8, width: i64) void {
        self.affine(name ++ ".norm", .{width}, width);
        self.affine(name ++ ".proj_in", .{ width, width, 1, 1 }, width);
        self.affine(name ++ ".proj_out", .{ width, width, 1, 1 }, width);
        const prefix = name ++ ".transformer_blocks.0.";
        inline for (.{ "norm1", "norm2", "norm3" }) |norm| self.affine(prefix ++ norm, .{width}, width);
        inline for (.{ "attn1", "attn2" }, .{ width, context_width }) |attn, inputs| {
            self.parameter(prefix ++ attn ++ ".to_q.weight", .{ width, width });
            self.parameter(prefix ++ attn ++ ".to_k.weight", .{ width, inputs });
            self.parameter(prefix ++ attn ++ ".to_v.weight", .{ width, inputs });
            self.affine(prefix ++ attn ++ ".to_out.0", .{ width, width }, width);
        }
        self.affine(prefix ++ "ff.net.0.proj", .{ 8 * width, width }, 8 * width);
        self.affine(prefix ++ "ff.net.2", .{ width, 4 * width }, width);
    }
};

pub const schema = blk: {
    @setEvalBranchQuota(200_000);
    var b: Schema = .{};
    b.affine("time_embed.0", .{ 1280, 320 }, 1280);
    b.affine("time_embed.2", .{ 1280, 1280 }, 1280);
    b.affine("input_blocks.0.0", .{ 320, channels, 3, 3 }, 320);
    b.affine("out.0", .{320}, 320);
    b.affine("out.2", .{ channels, 320, 3, 3 }, channels);
    var c: i64 = 320;
    var skips: [12]i64 = undefined;
    skips[0] = c;
    var layer: usize = 1;
    for (widths, 0..) |width, stage| {
        for (0..2) |_| {
            const prefix = std.fmt.comptimePrint("input_blocks.{d}", .{layer});
            b.resnet(prefix ++ ".0", c, width);
            if (stage < 3) b.attention(prefix ++ ".1", width);
            c = width;
            skips[layer] = c;
            layer += 1;
        }
        if (stage < 3) {
            b.affine(std.fmt.comptimePrint("input_blocks.{d}.0.op", .{layer}), .{ c, c, 3, 3 }, c);
            skips[layer] = c;
            layer += 1;
        }
    }
    b.resnet("middle_block.0", c, c);
    b.attention("middle_block.1", c);
    b.resnet("middle_block.2", c, c);
    for (0..4) |stage| {
        const width = widths[3 - stage];
        for (0..3) |block| {
            const prefix = std.fmt.comptimePrint("output_blocks.{d}", .{3 * stage + block});
            layer -= 1;
            b.resnet(prefix ++ ".0", c + skips[layer], width);
            if (stage > 0) b.attention(prefix ++ ".1", width);
            c = width;
        }
        if (stage < 3) {
            const module = if (stage == 0) 1 else 2;
            b.affine(std.fmt.comptimePrint("output_blocks.{d}.{d}.conv", .{ 3 * stage + 2, module }), .{ c, c, 3, 3 }, c);
        }
    }
    if (b.len != b.items.len or layer != 0) @compileError("Invalid SD 1.5 U-Net schema");
    break :blk b.items;
};

fn resnet(w: l.Weights, prefix: []const u8, x: Tensor, time: Tensor) Tensor {
    var key: [112]u8 = undefined;
    const n = l.groupNorm(w, l.name(&key, "{s}.in_layers.0", .{prefix}), x, 32, epsilon).silu();
    const a = l.conv(w, l.name(&key, "{s}.in_layers.2", .{prefix}), n, 1, @splat(1), true);
    const shift = t.linear(w, l.name(&key, "{s}.emb_layers.1", .{prefix}), time, true);
    const conditioned = a.add(shift.broadcast(a.shape(), &.{ 0, 1 }));
    const y = l.groupNorm(w, l.name(&key, "{s}.out_layers.0", .{prefix}), conditioned, 32, epsilon).silu();
    const update = l.conv(w, l.name(&key, "{s}.out_layers.3", .{prefix}), y, 1, @splat(1), true);
    const residual = if (update.dim(1) == x.dim(1)) x else l.conv(w, l.name(&key, "{s}.skip_connection", .{prefix}), x, 1, @splat(0), true);
    return residual.add(update);
}

fn attention(w: l.Weights, name: []const u8, query: Tensor, context: Tensor) Tensor {
    var key: [112]u8 = undefined;
    const q = t.linear(w, l.name(&key, "{s}.to_q", .{name}), query, false);
    const k = t.linear(w, l.name(&key, "{s}.to_k", .{name}), context, false);
    const v = t.linear(w, l.name(&key, "{s}.to_v", .{name}), context, false);
    return t.linear(w, l.name(&key, "{s}.to_out.0", .{name}), t.attention(q, k, v, 8), true);
}

fn transform(w: l.Weights, name: []const u8, x: Tensor, context: Tensor) Tensor {
    var key: [112]u8 = undefined;
    const n = l.groupNorm(w, l.name(&key, "{s}.norm", .{name}), x, 32, 1e-6);
    const projected = l.conv(w, l.name(&key, "{s}.proj_in", .{name}), n, 1, @splat(0), true);
    const width = x.dim(1);
    var tokens = projected.transpose(.{ 0, 2, 3, 1 }).reshape(.{ 1, x.dim(2) * x.dim(3), width });
    var prefix: [96]u8 = undefined;
    const block = l.name(&prefix, "{s}.transformer_blocks.0", .{name});
    const n1 = t.norm(w, l.name(&key, "{s}.norm1", .{block}), tokens, epsilon);
    tokens = tokens.add(attention(w, l.name(&key, "{s}.attn1", .{block}), n1, n1));
    const n2 = t.norm(w, l.name(&key, "{s}.norm2", .{block}), tokens, epsilon);
    tokens = tokens.add(attention(w, l.name(&key, "{s}.attn2", .{block}), n2, context));
    const n3 = t.norm(w, l.name(&key, "{s}.norm3", .{block}), tokens, epsilon);
    const gate = t.linear(w, l.name(&key, "{s}.ff.net.0.proj", .{block}), n3, true);
    const hidden = gate.slice(2, .{ .end = 4 * width }).mul(t.gelu(gate.slice(2, .{ .start = 4 * width })));
    tokens = tokens.add(t.linear(w, l.name(&key, "{s}.ff.net.2", .{block}), hidden, true));
    const image = tokens.reshape(.{ 1, x.dim(2), x.dim(3), width }).transpose(.{ 0, 3, 1, 2 });
    return x.add(l.conv(w, l.name(&key, "{s}.proj_out", .{name}), image, 1, @splat(0), true));
}

fn forward(parameters: []const Tensor, latent: Tensor, embedding: Tensor, context: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    const time = t.linear(w, "time_embed.2", t.linear(w, "time_embed.0", embedding, true).silu(), true).silu();
    var x = l.conv(w, "input_blocks.0.0", latent, 1, @splat(1), true);
    var skips: [12]Tensor = undefined;
    skips[0] = x;
    var layer: usize = 1;
    var name: [72]u8 = undefined;
    for (0..4) |stage| {
        for (0..2) |_| {
            x = resnet(w, l.name(&name, "input_blocks.{d}.0", .{layer}), x, time);
            if (stage < 3) x = transform(w, l.name(&name, "input_blocks.{d}.1", .{layer}), x, context);
            skips[layer] = x;
            layer += 1;
        }
        if (stage < 3) {
            x = l.conv3x3Stride2(w, l.name(&name, "input_blocks.{d}.0.op", .{layer}), x, true);
            skips[layer] = x;
            layer += 1;
        }
    }
    x = resnet(w, "middle_block.0", x, time);
    x = transform(w, "middle_block.1", x, context);
    x = resnet(w, "middle_block.2", x, time);
    for (0..4) |stage| {
        for (0..3) |block| {
            const index = stage * 3 + block;
            layer -= 1;
            const joined = Tensor.concatenate(&.{ x, skips[layer] }, 1);
            x = resnet(w, l.name(&name, "output_blocks.{d}.0", .{index}), joined, time);
            if (stage > 0) x = transform(w, l.name(&name, "output_blocks.{d}.1", .{index}), x, context);
        }
        if (stage < 3) {
            const module: usize = if (stage == 0) 1 else 2;
            x = l.conv(w, l.name(&name, "output_blocks.{d}.{d}.conv", .{ 3 * stage + 2, module }), l.nearest2x(x), 1, @splat(1), true);
        }
    }
    return l.conv(w, "out.2", l.groupNorm(w, "out.0", x, 32, epsilon).silu(), 1, @splat(1), true);
}

pub const Model = struct {
    session: *const runtime.Session,
    weights: runtime.checkpoint.Parameters,
    graph: runtime.Compiled,
    image: zml.Shape,

    const embedding_shape: zml.Shape = .init(.{ 1, 320 }, .f32);
    const context_shape: zml.Shape = .init(.{ 1, context_length, context_width }, .f32);

    pub fn load(session: *const runtime.Session, path: []const u8, height: i64, width: i64) !Model {
        if (height < 8 or height > 64 or @mod(height, 8) != 0 or width < 8 or width > 64 or @mod(width, 8) != 0)
            return error.InvalidLatentShape;
        const image: zml.Shape = .init(.{ 1, channels, height, width }, .f32);
        const weights = try runtime.checkpoint.load(session.allocator, session.io, session.platform, schema.len, &schema, l.Parameter.at, path);
        errdefer weights.deinit(session.allocator);
        var graph = try runtime.Compiled.init(session, forward, .{
            weights.tensors, Tensor.fromShape(image), Tensor.fromShape(embedding_shape), Tensor.fromShape(context_shape),
        });
        graph.args.bake(weights.buffers);
        return .{ .session = session, .weights = weights, .graph = graph, .image = image };
    }

    pub fn condition(self: *const Model, values: []const f32) !zml.Buffer {
        if (values.len != context_shape.count()) return error.InvalidConditioning;
        return .fromBytesOpts(self.session.io, self.session.platform, context_shape, std.mem.sliceAsBytes(values), .{ .wait = true });
    }

    pub fn predict(self: *Model, timestep: usize, latent: []const f32, context: zml.Buffer, output: []f32) !u32 {
        if (latent.len != self.image.count() or output.len != latent.len or timestep >= 1000)
            return error.InvalidInput;
        var embedding: [320]f32 = undefined;
        const time: f32 = @floatFromInt(timestep);
        for (0..160) |i| {
            const exponent = -@log(@as(f32, 10000)) * @as(f32, @floatFromInt(i)) / 160;
            const phase = time * @exp(exponent);
            embedding[i] = @cos(phase);
            embedding[i + 160] = @sin(phase);
        }
        const session = self.session;
        var source = try zml.Buffer.fromBytesOpts(session.io, session.platform, self.image, std.mem.sliceAsBytes(latent), .{ .wait = true });
        defer source.deinit();
        var time_buffer = try zml.Buffer.fromBytesOpts(session.io, session.platform, embedding_shape, std.mem.asBytes(&embedding), .{ .wait = true });
        defer time_buffer.deinit();
        try self.graph.run(.{ source, time_buffer, context });
        var noise = self.graph.results.get(zml.Buffer);
        defer noise.deinit();
        if (!noise.shape().eql(self.image)) return error.InvalidOutputShape;
        const shards = try session.verifyPlacement(&noise);
        try noise.toSlice(session.io, .init(self.image, std.mem.sliceAsBytes(output)));
        return shards;
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit(self.session.allocator);
        self.weights.deinit(self.session.allocator);
    }
};
