const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");

pub const image_bytes = 28 * 28;

const Weight = struct { name: []const u8, shape: zml.Shape };
const weights = [_]Weight{
    .{ .name = "fc1.weight", .shape = .init(.{ 500, image_bytes }, .f32) },
    .{ .name = "fc1.bias", .shape = .init(.{500}, .f32) },
    .{ .name = "fc2.weight", .shape = .init(.{ 10, 500 }, .f32) },
    .{ .name = "fc2.bias", .shape = .init(.{10}, .f32) },
};

fn weightAt(_: void, index: usize, _: []u8) Weight {
    return weights[index];
}

fn layer(input: zml.Tensor, w: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return w.withTags(.{ .out, .feature }).dot(input, .feature)
        .add(b.withTags(.{.out})).relu().withTags(.{.feature});
}

fn forward(parameters: []const zml.Tensor, image: zml.Tensor) zml.Tensor {
    const input = image.flatten().convert(.f32).withTags(.{.feature});
    const hidden = layer(input, parameters[0], parameters[1]);
    return layer(hidden, parameters[2], parameters[3]).argMax(0).indices.convert(.u8);
}

const input_shape: zml.Shape = .init(.{ 28, 28 }, .u8);

pub const Result = struct { digit: u8, shards: u32 };

pub const Model = struct {
    session: *const runtime.Session,
    weights_on_device: runtime.checkpoint.Parameters,
    compiled: runtime.Compiled,

    pub fn load(session: *const runtime.Session, path: []const u8) !Model {
        const parameters = try runtime.checkpoint.load(
            session.allocator,
            session.io,
            session.platform,
            weights.len,
            {},
            weightAt,
            path,
        );
        errdefer parameters.deinit(session.allocator);
        const input: zml.Tensor = .fromShape(input_shape);
        var compiled = try runtime.Compiled.init(session, forward, .{ parameters.tensors, input });
        compiled.args.bake(parameters.buffers);
        return .{ .session = session, .weights_on_device = parameters, .compiled = compiled };
    }

    pub fn infer(self: *Model, pixels: []const u8) !Result {
        if (pixels.len != image_bytes) return error.InvalidBufferSize;
        const session = self.session;
        var input: zml.Buffer = try .fromBytes(session.io, session.platform, input_shape, pixels);
        defer input.deinit();
        try self.compiled.run(.{input});
        var output = self.compiled.results.get(zml.Buffer);
        defer output.deinit();
        const shards = try session.verifyPlacement(&output);
        const digit = try output.getValue(u8, session.io);
        if (digit > 9) return error.InvalidPrediction;
        return .{ .digit = digit, .shards = shards };
    }

    pub fn deinit(self: *Model) void {
        self.compiled.deinit(self.session.allocator);
        self.weights_on_device.deinit(self.session.allocator);
    }
};
