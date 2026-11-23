const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("session.zig");
const layers = @import("layers.zig");

pub const TensorModel = struct {
    session: *const runtime.Session,
    parameters: runtime.checkpoint.Parameters,
    input_shape: ?zml.Shape = null,
    compiled: ?runtime.Compiled = null,

    pub fn load(session: *const runtime.Session, schema: []const layers.Parameter, path: []const u8) !TensorModel {
        return loadCheckpoint(session, schema, path, .f32);
    }

    pub fn loadCheckpoint(session: *const runtime.Session, schema: []const layers.Parameter, path: []const u8, storage: runtime.checkpoint.Storage) !TensorModel {
        return .{
            .session = session,
            .parameters = try runtime.checkpoint.loadWithStorage(
                session.allocator,
                session.io,
                session.platform,
                schema.len,
                schema,
                layers.Parameter.at,
                path,
                storage,
            ),
        };
    }

    pub fn deinit(self: *TensorModel) void {
        if (self.compiled) |*compiled| compiled.deinit(self.session.allocator);
        self.parameters.deinit(self.session.allocator);
    }

    pub fn infer(
        self: *TensorModel,
        comptime forward: anytype,
        input_shape: zml.Shape,
        output_shape: zml.Shape,
        input: []const f32,
        output: []f32,
    ) !u32 {
        if (input.len != input_shape.count() or output.len != output_shape.count())
            return error.InvalidBufferSize;
        return self.inferBytes(forward, input_shape, output_shape, std.mem.sliceAsBytes(input), output);
    }

    pub fn inferBytes(
        self: *TensorModel,
        comptime forward: anytype,
        input_shape: zml.Shape,
        output_shape: zml.Shape,
        input: []const u8,
        output: []f32,
    ) !u32 {
        if (output.len != output_shape.count()) return error.InvalidBufferSize;
        var result = try self.inferDevice(forward, input_shape, input);
        defer result.deinit();
        if (!result.shape().eql(output_shape)) return error.UnexpectedOutputShape;
        const shards = try self.session.verifyPlacement(&result);
        try result.toSlice(self.session.io, .init(output_shape, std.mem.sliceAsBytes(output)));
        return shards;
    }

    pub fn inferDevice(self: *TensorModel, comptime forward: anytype, input_shape: zml.Shape, input: []const u8) !zml.Buffer {
        if (input.len != input_shape.byteSize()) return error.InvalidBufferSize;
        const session = self.session;
        if (self.input_shape == null or !self.input_shape.?.eql(input_shape)) {
            var replacement = try runtime.Compiled.init(session, forward, .{
                self.parameters.tensors, zml.Tensor.fromShape(input_shape),
            });
            replacement.args.bake(self.parameters.buffers);
            if (self.compiled) |*previous| previous.deinit(session.allocator);
            self.compiled = replacement;
            self.input_shape = input_shape;
        }
        const executable = &self.compiled.?;
        var source = try zml.Buffer.fromBytesOpts(session.io, session.platform, input_shape, input, .{ .wait = true });
        defer source.deinit();
        try executable.run(.{source});
        return executable.results.get(zml.Buffer);
    }
};
