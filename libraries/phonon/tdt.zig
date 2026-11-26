const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const Tensor = zml.Tensor;
const Parameter = runtime.layers.Parameter;
const Weights = runtime.layers.Weights;

pub const encoder_width = 1024;
pub const hidden_size = 640;
pub const token_count = 8193;
pub const duration_count = 5;
pub const blank_token = token_count - 1;
pub const state_size = 4 * hidden_size;
pub const input_count = 1 + encoder_width + state_size;
pub const output_count = token_count + duration_count + state_size;

fn parameter(name: []const u8, dims: anytype) Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

pub const schema = [_]Parameter{
    parameter("encoder_projector.weight", .{ hidden_size, encoder_width }),
    parameter("encoder_projector.bias", .{hidden_size}),
    parameter("decoder.embedding.weight", .{ token_count, hidden_size }),
    parameter("decoder.lstm.weight_ih_l0", .{ 4 * hidden_size, hidden_size }),
    parameter("decoder.lstm.weight_hh_l0", .{ 4 * hidden_size, hidden_size }),
    parameter("decoder.lstm.bias_ih_l0", .{4 * hidden_size}),
    parameter("decoder.lstm.bias_hh_l0", .{4 * hidden_size}),
    parameter("decoder.lstm.weight_ih_l1", .{ 4 * hidden_size, hidden_size }),
    parameter("decoder.lstm.weight_hh_l1", .{ 4 * hidden_size, hidden_size }),
    parameter("decoder.lstm.bias_ih_l1", .{4 * hidden_size}),
    parameter("decoder.lstm.bias_hh_l1", .{4 * hidden_size}),
    parameter("decoder.decoder_projector.weight", .{ hidden_size, hidden_size }),
    parameter("decoder.decoder_projector.bias", .{hidden_size}),
    parameter("joint.head.weight", .{ token_count + duration_count, hidden_size }),
    parameter("joint.head.bias", .{token_count + duration_count}),
};

fn gates(comptime side: []const u8, comptime layer: usize, weights: Weights, input: Tensor) Tensor {
    const suffix = std.fmt.comptimePrint("_{s}_l{d}", .{ side, layer });
    return runtime.transformer.linearParameters(input, weights.get("decoder.lstm.weight", suffix), weights.get("decoder.lstm.bias", suffix));
}

fn forward(parameters: []const Tensor, input: Tensor) Tensor {
    const weights: Weights = .{ .schema = &schema, .tensors = parameters };
    const token = input.slice(0, .{ .end = 1 }).convert(.i32);
    const encoder = input.slice(0, .{ .start = 1, .end = 1 + encoder_width }).reshape(.{ 1, encoder_width });
    const states = input.slice(0, .{ .start = 1 + encoder_width });
    const embedding = weights.get("decoder.embedding", ".weight").withTags(.{ .token, .embedding });
    var x = embedding.gather(.{ .token = token }, .{}).reshape(.{ 1, hidden_size });
    var new_states: [4]Tensor = undefined;
    inline for (0..2) |layer| {
        const h = states.slice(0, .{ .start = (layer * 2) * hidden_size, .end = (layer * 2 + 1) * hidden_size }).reshape(x.shape());
        const c = states.slice(0, .{ .start = (layer * 2 + 1) * hidden_size, .end = (layer * 2 + 2) * hidden_size }).reshape(x.shape());
        const summed = gates("ih", layer, weights, x).add(gates("hh", layer, weights, h));
        const i = summed.slice(1, .{ .end = hidden_size }).sigmoid();
        const f = summed.slice(1, .{ .start = hidden_size, .end = 2 * hidden_size }).sigmoid();
        const g = summed.slice(1, .{ .start = 2 * hidden_size, .end = 3 * hidden_size }).tanh();
        const o = summed.slice(1, .{ .start = 3 * hidden_size }).sigmoid();
        const next_c = f.mul(c).add(i.mul(g));
        x = o.mul(next_c.tanh());
        new_states[layer * 2] = x;
        new_states[layer * 2 + 1] = next_c;
    }
    const encoded = runtime.layers.linear(weights, "encoder_projector", encoder, true);
    const predicted = runtime.layers.linear(weights, "decoder.decoder_projector", x, true);
    const joint = encoded.add(predicted);
    const activated = joint.maximum(Tensor.scalar(0, .f32).broad(joint.shape()));
    const logits = runtime.layers.linear(weights, "joint.head", activated, true);
    return Tensor.concatenate(&.{ logits, Tensor.concatenate(&new_states, 1) }, 1).flatten();
}

pub const Decoder = struct {
    graph: runtime.TensorModel,

    pub fn load(session: *const runtime.Session, weights: []const u8) !Decoder {
        return .{ .graph = try .load(session, &schema, weights) };
    }

    pub fn deinit(self: *Decoder) void {
        self.graph.deinit();
    }

    pub fn step(self: *Decoder, input: []const f32, output: []f32) !u32 {
        if (input.len != input_count or output.len != output_count) return error.InvalidBufferSize;
        for (input) |value| if (!std.math.isFinite(value)) return error.NonfiniteInput;
        const token = input[0];
        if (token < 0 or @trunc(token) != token or token >= token_count) return error.InvalidTokenId;
        return self.graph.infer(forward, .init(.{input_count}, .f32), .init(.{output_count}, .f32), input, output);
    }
};
