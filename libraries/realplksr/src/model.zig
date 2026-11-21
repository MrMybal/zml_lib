const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const graph = @import("graph.zig");

pub const scale = graph.scale;

pub const Model = struct {
    session: *const runtime.Session,
    parameters: runtime.checkpoint.Parameters,
    compiled: runtime.Compiled,
    input_shape: zml.Shape,
    output_shape: zml.Shape,

    pub fn load(session: *const runtime.Session, path: []const u8, width: u32, height: u32) !Model {
        if (width == 0 or height == 0 or @as(u64, width) * height > graph.max_image_area)
            return error.InvalidDimensions;
        const h: i64 = height;
        const w: i64 = width;
        const input_shape: zml.Shape = .init(.{ 1, 3, h, w }, .f32);
        const output_shape: zml.Shape = .init(.{ 1, 3, h * scale, w * scale }, .f32);
        const parameters = try runtime.checkpoint.load(
            session.allocator,
            session.io,
            session.platform,
            graph.tensor_count,
            {},
            graph.weightAt,
            path,
        );
        errdefer parameters.deinit(session.allocator);
        var compiled = try runtime.Compiled.init(session, graph.forward, .{
            parameters.tensors, zml.Tensor.fromShape(input_shape),
        });
        compiled.args.bake(parameters.buffers);
        return .{
            .session = session,
            .parameters = parameters,
            .compiled = compiled,
            .input_shape = input_shape,
            .output_shape = output_shape,
        };
    }

    pub fn deinit(self: *Model) void {
        self.compiled.deinit(self.session.allocator);
        self.parameters.deinit(self.session.allocator);
    }

    pub fn infer(self: *Model, input: []const f32, output: []f32) !u32 {
        if (input.len != self.input_shape.count() or output.len != self.output_shape.count())
            return error.InvalidBufferSize;
        const session = self.session;
        var device_input = try zml.Buffer.fromBytesOpts(
            session.io,
            session.platform,
            self.input_shape,
            std.mem.sliceAsBytes(input),
            .{ .wait = true },
        );
        defer device_input.deinit();
        try self.compiled.run(.{device_input});
        var device_output = self.compiled.results.get(zml.Buffer);
        defer device_output.deinit();
        const shards = try session.verifyPlacement(&device_output);
        try device_output.toSlice(session.io, .init(self.output_shape, std.mem.sliceAsBytes(output)));
        return shards;
    }
};
