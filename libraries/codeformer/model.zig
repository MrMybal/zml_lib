const std = @import("std");
const runtime = @import("zml_lib/runtime");
const graph = @import("graph.zig");
const schema = @import("weights.zig").schema;

pub const size = graph.size;
pub const input_count = graph.input_count;
pub const output_count = graph.output_count;

pub const Model = struct {
    graph_model: runtime.TensorModel,
    input_buffer: []f32,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        var loaded = try runtime.TensorModel.load(session, &schema, path);
        errdefer loaded.deinit();
        return .{ .graph_model = loaded, .input_buffer = try session.allocator.alloc(f32, input_count + 1) };
    }

    pub fn deinit(self: *Model) void {
        self.graph_model.session.allocator.free(self.input_buffer);
        self.graph_model.deinit();
    }

    pub fn infer(self: *Model, input: []const f32, fidelity: f32, output: []f32) !u32 {
        if (input.len != input_count or output.len != output_count) return error.InvalidBufferSize;
        if (!std.math.isFinite(fidelity) or fidelity < 0 or fidelity > 1) return error.InvalidFidelity;
        @memcpy(self.input_buffer[0..input_count], input);
        self.input_buffer[input_count] = fidelity;
        return self.graph_model.infer(graph.forward, .init(.{input_count + 1}, .f32), .init(.{output_count}, .f32), self.input_buffer, output);
    }
};
