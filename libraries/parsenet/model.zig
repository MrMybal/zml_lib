const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const Tensor = l.Tensor;
const schema = @import("weights.zig").schema;

pub const size = 512;
pub const channels = 19;
pub const input_count = 3 * size * size;
pub const output_count = (channels + 3) * size * size;
const Mode = enum { keep, up, down };

fn conv(w: l.Weights, block: []const u8, x: Tensor, mode: Mode, normalized: bool, act: bool) Tensor {
    var key: [64]u8 = undefined;
    const y = if (mode == .up) l.nearest2x(x) else x;
    var out = l.conv(w, l.name(&key, "{s}.conv2d", .{block}), l.reflectOne(y), if (mode == .down) 2 else 1, @splat(0), !normalized);
    if (normalized) out = l.batchNorm(w, l.name(&key, "{s}.norm.norm", .{block}), out);
    return if (act) l.leakyRelu(out) else out;
}

fn residual(w: l.Weights, section: []const u8, index: usize, x: Tensor, mode: Mode) Tensor {
    var key: [48]u8 = undefined;
    const first = conv(w, l.name(&key, "{s}.{d}.conv1", .{ section, index }), x, if (mode == .up) .up else .keep, true, true);
    const second = conv(w, l.name(&key, "{s}.{d}.conv2", .{ section, index }), first, if (mode == .down) .down else .keep, true, false);
    const shortcut = if (mode == .keep) x else conv(w, l.name(&key, "{s}.{d}.shortcut_func", .{ section, index }), x, mode, false, false);
    return second.add(shortcut);
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const w: l.Weights = .{ .schema = &schema, .tensors = parameters };
    var features = conv(w, "encoder.0", input, .keep, false, false);
    for (1..5) |i| features = residual(w, "encoder", i, features, .down);
    var y = features;
    for (0..10) |i| y = residual(w, "body", i, y, .keep);
    y = y.add(features);
    for (0..4) |i| y = residual(w, "decoder", i, y, .up);
    const mask = conv(w, "out_mask_conv", y, .keep, false, false);
    const rgb = conv(w, "out_img_conv", y, .keep, false, false);
    return Tensor.concatenate(&.{ mask, rgb }, 1);
}

pub const Model = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        return .{ .graph = try .load(session, &schema, path) };
    }

    pub fn deinit(self: *Model) void {
        self.graph.deinit();
    }

    pub fn infer(self: *Model, input: []const f32, output: []f32) !u32 {
        return self.graph.infer(forward, .init(.{ 1, 3, size, size }, .f32), .init(.{ 1, channels + 3, size, size }, .f32), input, output);
    }
};
