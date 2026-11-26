const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = zml.Tensor;
const lora = @import("lora.zig");
pub const LoRAInput = lora.Input;

pub const image_channels = 64;
pub const pooled_channels = 768;
pub const Storage = enum { f32, f16, e4m3, e4m3_scaled };
const time_channels = 256;
const epsilon: f32 = 1e-6;

pub const Profile = enum {
    schnell,
    canny_dev,
    fixture,
    canny_fixture,

    pub fn config(self: Profile) Config {
        return switch (self) {
            .schnell => .{ .width = 3072, .heads = 24, .text = 4096, .axes = .{ 16, 56, 56 }, .dual = 19, .single = 38 },
            .canny_dev => .{ .width = 3072, .heads = 24, .text = 4096, .axes = .{ 16, 56, 56 }, .dual = 19, .single = 38, .control = true },
            .fixture => .{ .width = 48, .heads = 3, .text = 32, .axes = .{ 4, 6, 6 }, .dual = 2, .single = 2 },
            .canny_fixture => .{ .width = 48, .heads = 3, .text = 32, .axes = .{ 4, 6, 6 }, .dual = 2, .single = 2, .control = true },
        };
    }
};

const Config = struct {
    width: i64,
    heads: i64,
    text: i64,
    axes: [3]i64,
    dual: usize,
    single: usize,
    control: bool = false,

    fn inputChannels(self: Config) i64 {
        return if (self.control) 2 * image_channels else image_channels;
    }
};

fn param(name: []const u8, dims: anytype, dtype: zml.DataType) l.Parameter {
    return .{ .name = name, .shape = .init(dims, dtype) };
}

fn pair(comptime name: []const u8, out: i64, in: i64, dtype: zml.DataType) [2]l.Parameter {
    return .{ param(name ++ ".weight", .{ out, in }, dtype), param(name ++ ".bias", .{out}, dtype) };
}

fn streamSchema(comptime name: []const u8, cfg: Config, dtype: zml.DataType) [12]l.Parameter {
    const d = cfg.width;
    const head = @divExact(d, cfg.heads);
    return pair(name ++ "_mod.lin", 6 * d, d, dtype) ++ pair(name ++ "_attn.qkv", 3 * d, d, dtype) ++
        pair(name ++ "_attn.proj", d, d, dtype) ++ pair(name ++ "_mlp.0", 4 * d, d, dtype) ++
        pair(name ++ "_mlp.2", d, 4 * d, dtype) ++ [_]l.Parameter{
        param(name ++ "_attn.norm.query_norm.scale", .{head}, dtype),
        param(name ++ "_attn.norm.key_norm.scale", .{head}, dtype),
    };
}

fn parameters(comptime cfg: Config, dtype: zml.DataType) [(if (cfg.control) 20 else 16) + 24 * cfg.dual + 8 * cfg.single]l.Parameter {
    @setEvalBranchQuota(400_000);
    const d = cfg.width;
    const head = @divExact(d, cfg.heads);
    const fixed = if (cfg.control) 20 else 16;
    var schema: [fixed + 24 * cfg.dual + 8 * cfg.single]l.Parameter = undefined;
    schema[0..16].* = pair("img_in", d, cfg.inputChannels(), dtype) ++ pair("txt_in", d, cfg.text, dtype) ++
        pair("time_in.in_layer", d, time_channels, dtype) ++ pair("time_in.out_layer", d, d, dtype) ++
        pair("vector_in.in_layer", d, pooled_channels, dtype) ++ pair("vector_in.out_layer", d, d, dtype) ++
        pair("final_layer.adaLN_modulation.1", 2 * d, d, dtype) ++ pair("final_layer.linear", image_channels, d, dtype);
    if (cfg.control) schema[16..20].* = pair("guidance_in.in_layer", d, time_channels, dtype) ++
        pair("guidance_in.out_layer", d, d, dtype);
    for (0..cfg.dual) |i| {
        const key = std.fmt.comptimePrint("double_blocks.{d}.", .{i});
        schema[fixed + i * 24 ..][0..24].* = streamSchema(key ++ "img", cfg, dtype) ++ streamSchema(key ++ "txt", cfg, dtype);
    }
    for (0..cfg.single) |i| {
        const key = std.fmt.comptimePrint("single_blocks.{d}.", .{i});
        schema[fixed + cfg.dual * 24 + i * 8 ..][0..8].* = pair(key ++ "modulation.lin", 3 * d, d, dtype) ++
            pair(key ++ "linear1", 7 * d, d, dtype) ++ pair(key ++ "linear2", d, 5 * d, dtype) ++ [_]l.Parameter{
            param(key ++ "norm.query_norm.scale", .{head}, dtype),
            param(key ++ "norm.key_norm.scale", .{head}, dtype),
        };
    }
    return schema;
}

const dequantize = t.decodeE4M3;

const Weights = struct {
    base: l.Weights,
    adapters: lora.Graph,
    scales: []const f32,

    fn get(self: Weights, name: []const u8, suffix: []const u8) Tensor {
        return self.base.get(name, suffix);
    }
};

fn linear(w: Weights, name: []const u8, x: Tensor) Tensor {
    for (w.base.schema, w.base.tensors, 0..) |parameter, weight, index| {
        if (parameter.name.len == name.len + ".weight".len and
            std.mem.startsWith(u8, parameter.name, name) and std.mem.endsWith(u8, parameter.name, ".weight"))
        {
            const scale = if (w.scales.len == 0) 1 else w.scales[index];
            const base = t.linearParameters(x, dequantize(weight, scale), dequantize(w.get(name, ".bias"), 1));
            return w.adapters.apply(index, x, base);
        }
    }
    std.debug.panic("Missing linear parameter {s}.weight", .{name});
}

fn timeEmbedding(time: Tensor) Tensor {
    const freq = Tensor.iota(.init(.{time_channels / 2}, .i32), 0).convert(.f32)
        .scale(@as(f32, -@log(10000.0)) / @as(f32, time_channels / 2)).exp();
    const angles = time.scale(@as(f32, 1000)).broad(freq.shape()).mul(freq);
    return Tensor.concatenate(&.{ angles.cos(), angles.sin() }, 0).reshape(.{ 1, time_channels });
}

const geluTanh = t.geluTanh;

const Rotation = struct { cos: Tensor, sin: Tensor };

fn rotation(cfg: Config, rows: i64, cols: i64, text: i64) Rotation {
    const total = text + rows * cols;
    const grid: zml.Shape = .init(.{ rows, cols }, .i32);
    const zero_text = Tensor.scalar(0, .f32).broad(.init(.{text}, .f32));
    const coordinates = [3]Tensor{
        Tensor.scalar(0, .f32).broad(.init(.{total}, .f32)),
        Tensor.concatenate(&.{ zero_text, Tensor.iota(grid, 0).convert(.f32).reshape(.{rows * cols}) }, 0),
        Tensor.concatenate(&.{ zero_text, Tensor.iota(grid, 1).convert(.f32).reshape(.{rows * cols}) }, 0),
    };
    var angles: [3]Tensor = undefined;
    for (coordinates, cfg.axes, &angles) |position, width, *value| {
        const freq = Tensor.iota(.init(.{@divExact(width, 2)}, .i32), 0).convert(.f32)
            .scale(@as(f32, -2 * @log(10000.0)) / @as(f32, @floatFromInt(width))).exp();
        const shape: zml.Shape = .init(.{ total, @divExact(width, 2) }, .f32);
        value.* = position.broadcast(shape, &.{0}).mul(freq.broadcast(shape, &.{1}));
    }
    const angle = Tensor.concatenate(&angles, 1);
    return .{ .cos = angle.cos(), .sin = angle.sin() };
}

fn rotate(x: Tensor, r: Rotation) Tensor {
    const shape = .{ x.dim(0), x.dim(1), x.dim(2), @divExact(x.dim(3), 2) };
    const pairs = x.reshape(shape ++ .{2});
    const even = pairs.slice(4, .single(0));
    const odd = pairs.slice(4, .single(1));
    const c = r.cos.broadcast(even.shape(), &.{ 2, 3 });
    const s = r.sin.broadcast(even.shape(), &.{ 2, 3 });
    return Tensor.concatenate(&.{
        even.mul(c).sub(odd.mul(s)).reshape(shape ++ .{1}),
        even.mul(s).add(odd.mul(c)).reshape(shape ++ .{1}),
    }, 4).reshape(x.shape());
}

fn rms(x: Tensor, scale: Tensor) Tensor {
    const factor = x.mul(x).mean(3).addConstant(epsilon).rsqrt().broad(x.shape());
    return x.mul(factor).mul(dequantize(scale, 1).broadcast(x.shape(), &.{3}));
}

const Qkv = struct { q: Tensor, k: Tensor, v: Tensor };

fn attentionInput(w: Weights, name: []const u8, qkv: Tensor, cfg: Config) Qkv {
    var key: [96]u8 = undefined;
    const d = cfg.width;
    const q = t.splitHeads(qkv.slice(2, .{ .end = d }), cfg.heads);
    const k = t.splitHeads(qkv.slice(2, .{ .start = d, .end = 2 * d }), cfg.heads);
    return .{
        .q = rms(q, w.get(l.name(&key, "{s}.query_norm", .{name}), ".scale")),
        .k = rms(k, w.get(l.name(&key, "{s}.key_norm", .{name}), ".scale")),
        .v = t.splitHeads(qkv.slice(2, .{ .start = 2 * d }), cfg.heads),
    };
}

fn attend(qkv: Qkv, r: Rotation) Tensor {
    return t.attend(t.scores(rotate(qkv.q, r), rotate(qkv.k, r)), qkv.v);
}

const Modulation = struct { shift: Tensor, scale: Tensor, gate: Tensor };

fn modulation(x: Tensor, start: i64, d: i64) Modulation {
    return .{
        .shift = x.slice(1, .{ .start = start, .end = start + d }),
        .scale = x.slice(1, .{ .start = start + d, .end = start + 2 * d }),
        .gate = x.slice(1, .{ .start = start + 2 * d, .end = start + 3 * d }),
    };
}

fn modulate(x: Tensor, shift: Tensor, scale: Tensor) Tensor {
    return t.normAffineFree(x, epsilon).mul(scale.addConstant(@as(f32, 1)).broadcast(x.shape(), &.{ 0, 2 }))
        .add(shift.broadcast(x.shape(), &.{ 0, 2 }));
}

fn residual(x: Tensor, update: Tensor, gate: Tensor) Tensor {
    return x.add(update.mul(gate.broadcast(x.shape(), &.{ 0, 2 })));
}

const Stream = struct { qkv: Qkv, first: Modulation, second: Modulation };

fn streamInput(w: Weights, prefix: []const u8, x: Tensor, condition: Tensor, cfg: Config) Stream {
    var key: [96]u8 = undefined;
    const mod = linear(w, l.name(&key, "{s}_mod.lin", .{prefix}), condition);
    const first = modulation(mod, 0, cfg.width);
    const qkv = linear(w, l.name(&key, "{s}_attn.qkv", .{prefix}), modulate(x, first.shift, first.scale));
    return .{
        .qkv = attentionInput(w, l.name(&key, "{s}_attn.norm", .{prefix}), qkv, cfg),
        .first = first,
        .second = modulation(mod, 3 * cfg.width, cfg.width),
    };
}

fn streamOutput(w: Weights, prefix: []const u8, x: Tensor, values: Tensor, stream: Stream) Tensor {
    var key: [96]u8 = undefined;
    const after = residual(x, linear(w, l.name(&key, "{s}_attn.proj", .{prefix}), values), stream.first.gate);
    const mod = stream.second;
    const hidden = linear(w, l.name(&key, "{s}_mlp.0", .{prefix}), modulate(after, mod.shift, mod.scale));
    return residual(after, linear(w, l.name(&key, "{s}_mlp.2", .{prefix}), geluTanh(hidden)), mod.gate);
}

const Streams = struct { image: Tensor, text: Tensor };

fn dualBlock(w: Weights, x: Streams, condition: Tensor, r: Rotation, cfg: Config, index: usize) Streams {
    var img_key: [64]u8 = undefined;
    var txt_key: [64]u8 = undefined;
    const img_name = l.name(&img_key, "double_blocks.{d}.img", .{index});
    const txt_name = l.name(&txt_key, "double_blocks.{d}.txt", .{index});
    const img = streamInput(w, img_name, x.image, condition, cfg);
    const txt = streamInput(w, txt_name, x.text, condition, cfg);
    const values = attend(.{
        .q = Tensor.concatenate(&.{ txt.qkv.q, img.qkv.q }, 2),
        .k = Tensor.concatenate(&.{ txt.qkv.k, img.qkv.k }, 2),
        .v = Tensor.concatenate(&.{ txt.qkv.v, img.qkv.v }, 2),
    }, r);
    const count = x.text.dim(1);
    return .{
        .image = streamOutput(w, img_name, x.image, values.slice(1, .{ .start = count }), img),
        .text = streamOutput(w, txt_name, x.text, values.slice(1, .{ .end = count }), txt),
    };
}

fn singleBlock(w: Weights, x: Tensor, condition: Tensor, r: Rotation, cfg: Config, index: usize) Tensor {
    var key: [96]u8 = undefined;
    const d = cfg.width;
    const mod = modulation(linear(w, l.name(&key, "single_blocks.{d}.modulation.lin", .{index}), condition), 0, d);
    const joint = linear(w, l.name(&key, "single_blocks.{d}.linear1", .{index}), modulate(x, mod.shift, mod.scale));
    const qkv = attentionInput(w, l.name(&key, "single_blocks.{d}.norm", .{index}), joint.slice(2, .{ .end = 3 * d }), cfg);
    const hidden = Tensor.concatenate(&.{ attend(qkv, r), geluTanh(joint.slice(2, .{ .start = 3 * d })) }, 2);
    return residual(x, linear(w, l.name(&key, "single_blocks.{d}.linear2", .{index}), hidden), mod.gate);
}

pub fn Model(comptime profile: Profile, comptime storage: Storage) type {
    const cfg = profile.config();
    return struct {
        const Self = @This();
        pub const schema = parameters(cfg, switch (storage) {
            .f32, .e4m3_scaled => .f32,
            .f16 => .f16,
            .e4m3 => .u8,
        });
        const scaled_schema = schema ++ [_]l.Parameter{param("weight_scales", .{schema.len}, .f32)};
        pub const text_channels = cfg.text;
        pub const input_channels = cfg.inputChannels();
        pub const has_control = cfg.control;
        session: *const runtime.Session,
        weights: runtime.checkpoint.Parameters,
        adapters: ?lora.Parameters,
        graph: runtime.Compiled,
        image_shape: zml.Shape,
        output_shape: zml.Shape,
        text: zml.Buffer,
        pooled: zml.Buffer,

        fn forward(weights: []const Tensor, scales: []const f32, adapters: lora.Graph, guidance: f32, image: Tensor, text: Tensor, pooled: Tensor, time: Tensor) Tensor {
            const w: Weights = .{ .base = .{ .schema = &schema, .tensors = weights }, .scales = scales, .adapters = adapters };
            const time_vec = linear(w, "time_in.out_layer", linear(w, "time_in.in_layer", timeEmbedding(time)).silu());
            const pooled_vec = linear(w, "vector_in.out_layer", linear(w, "vector_in.in_layer", pooled).silu());
            const timed = if (cfg.control) time_vec.add(linear(w, "guidance_in.out_layer", linear(
                w,
                "guidance_in.in_layer",
                timeEmbedding(Tensor.scalar(guidance, .f32)),
            ).silu())) else time_vec;
            const condition = timed.add(pooled_vec).silu();
            const rows = image.dim(1);
            const cols = image.dim(2);
            const count = text.dim(1);
            const r = rotation(cfg, rows, cols, count);
            var streams: Streams = .{
                .image = linear(w, "img_in", image.reshape(.{ 1, rows * cols, input_channels })),
                .text = linear(w, "txt_in", text),
            };
            for (0..cfg.dual) |i| streams = dualBlock(w, streams, condition, r, cfg, i);
            var joint = Tensor.concatenate(&.{ streams.text, streams.image }, 1);
            for (0..cfg.single) |i| joint = singleBlock(w, joint, condition, r, cfg, i);
            const mod = linear(w, "final_layer.adaLN_modulation.1", condition);
            const x = modulate(joint.slice(1, .{ .start = count }), mod.slice(1, .{ .end = cfg.width }), mod.slice(1, .{ .start = cfg.width }));
            return linear(w, "final_layer.linear", x).reshape(.{ 1, rows, cols, image_channels });
        }

        pub fn load(session: *const runtime.Session, path: []const u8, rows: i64, cols: i64, tokens: i64, text_values: []const f32, pooled_values: []const f32, adapter_input: ?LoRAInput, guidance: f32) !Self {
            if (rows < 1 or rows > 64 or cols < 1 or cols > 64 or tokens < 1 or tokens > 512)
                return error.InvalidShape;
            if (!std.math.isFinite(guidance) or guidance < 0 or guidance > 100) return error.InvalidGuidance;
            const image_shape: zml.Shape = .init(.{ 1, rows, cols, input_channels }, .f32);
            const output_shape: zml.Shape = .init(.{ 1, rows, cols, image_channels }, .f32);
            const text_shape: zml.Shape = .init(.{ 1, tokens, cfg.text }, .f32);
            const pooled_shape: zml.Shape = .init(.{ 1, pooled_channels }, .f32);
            if (text_values.len != text_shape.count() or pooled_values.len != pooled_shape.count())
                return error.InvalidConditioning;
            const weights = if (storage == .e4m3_scaled) try runtime.checkpoint.loadMixedE4M3(
                session.allocator,
                session.io,
                session.platform,
                scaled_schema.len,
                &scaled_schema,
                l.Parameter.at,
                path,
            ) else try runtime.checkpoint.loadWithStorage(
                session.allocator,
                session.io,
                session.platform,
                schema.len,
                &schema,
                l.Parameter.at,
                path,
                switch (storage) {
                    .f32 => .f32,
                    .f16 => .f16,
                    .e4m3 => .e4m3,
                    .e4m3_scaled => unreachable,
                },
            );
            errdefer weights.deinit(session.allocator);
            var scales: [if (storage == .e4m3_scaled) schema.len else 0]f32 = undefined;
            if (storage == .e4m3_scaled) {
                const scale_buffer = weights.buffers[schema.len];
                try scale_buffer.toSlice(session.io, .init(scale_buffer.shape(), std.mem.asBytes(&scales)));
                var quantized: usize = 0;
                var non_unit: usize = 0;
                for (scales, weights.tensors[0..schema.len]) |scale, tensor| {
                    if (scale <= 0 or !std.math.isFinite(scale * 448)) return error.InvalidFP8Scale;
                    if (tensor.dtype() == .u8) {
                        quantized += 1;
                        non_unit += @intFromBool(scale != 1);
                    } else if (scale != 1) return error.InvalidFP8Scale;
                }
                if (quantized == 0) return error.EmptyFP8Weights;
                std.log.info("weight_storage=e4m3_scaled quantized={d} non_unit_scales={d}", .{ quantized, non_unit });
            }
            const adapters = if (adapter_input) |input| try lora.Parameters.load(session, &schema, input) else null;
            errdefer if (adapters) |set| set.deinit(session.allocator);
            var graph = try runtime.Compiled.init(session, forward, .{
                weights.tensors[0..schema.len],
                &scales,
                if (adapters) |set| set.graph() else .{},
                guidance,
                Tensor.fromShape(image_shape),
                Tensor.fromShape(text_shape),
                Tensor.fromShape(pooled_shape),
                Tensor.fromShape(.init(.{1}, .f32)),
            });
            errdefer graph.deinit(session.allocator);
            graph.args.bake(weights.buffers[0..schema.len]);
            if (adapters) |set| graph.args.bake(set.factors.buffers);
            var text = try zml.Buffer.fromBytesOpts(session.io, session.platform, text_shape, std.mem.sliceAsBytes(text_values), .{ .wait = true });
            errdefer text.deinit();
            const pooled = try zml.Buffer.fromBytesOpts(session.io, session.platform, pooled_shape, std.mem.sliceAsBytes(pooled_values), .{ .wait = true });
            return .{ .session = session, .weights = weights, .adapters = adapters, .graph = graph, .image_shape = image_shape, .output_shape = output_shape, .text = text, .pooled = pooled };
        }

        pub fn predict(self: *Self, image: []const f32, time: f32, output: []f32) !u32 {
            if (image.len != self.image_shape.count() or output.len != self.output_shape.count() or
                !std.math.isFinite(time) or time < 0 or time > 1) return error.InvalidInput;
            const session = self.session;
            var source = try zml.Buffer.fromBytesOpts(session.io, session.platform, self.image_shape, std.mem.sliceAsBytes(image), .{ .wait = true });
            defer source.deinit();
            const time_value = [1]f32{time};
            var clock = try zml.Buffer.fromBytesOpts(session.io, session.platform, .init(.{1}, .f32), std.mem.asBytes(&time_value), .{ .wait = true });
            defer clock.deinit();
            try self.graph.run(.{ source, self.text, self.pooled, clock });
            var result = self.graph.results.get(zml.Buffer);
            defer result.deinit();
            if (!result.shape().eql(self.output_shape)) return error.InvalidOutputShape;
            const shards = try session.verifyPlacement(&result);
            try result.toSlice(session.io, .init(self.output_shape, std.mem.sliceAsBytes(output)));
            return shards;
        }

        pub fn deinit(self: *Self) void {
            self.graph.deinit(self.session.allocator);
            self.pooled.deinit();
            self.text.deinit();
            if (self.adapters) |set| set.deinit(self.session.allocator);
            self.weights.deinit(self.session.allocator);
        }
    };
}
