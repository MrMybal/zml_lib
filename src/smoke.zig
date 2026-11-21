const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");

fn addOp(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.add(rhs);
}

pub fn add(session: *const runtime.Session, lhs: f32, rhs: f32, output: *f32) !u32 {
    const shape = zml.Shape.scalar(.f32);
    var compiled = try runtime.Compiled.init(session, addOp, .{ zml.Tensor.fromShape(shape), zml.Tensor.fromShape(shape) });
    defer compiled.deinit(session.allocator);
    var a = try zml.Buffer.scalar(session.io, session.platform, lhs, .f32);
    defer a.deinit();
    var b = try zml.Buffer.scalar(session.io, session.platform, rhs, .f32);
    defer b.deinit();
    try compiled.run(.{ a, b });
    var result = compiled.results.get(zml.Buffer);
    defer result.deinit();
    const shards = try session.verifyPlacement(&result);
    output.* = try result.getValue(f32, session.io);
    return shards;
}

fn predict(input: zml.Tensor, weights: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    return weights.dot(input, .feature).add(bias).relu();
}

pub const Model = struct {
    session: *const runtime.Session,
    compiled: runtime.Compiled,
    x: zml.Buffer,
    w: zml.Buffer,
    b: zml.Buffer,

    pub fn load(session: *const runtime.Session) !Model {
        const input = zml.Tensor.init(.{2}, .f32).withTags(.{.feature});
        const weights = zml.Tensor.init(.{ 2, 2 }, .f32).withTags(.{ .label, .feature });
        const bias = zml.Tensor.init(.{2}, .f32).withTags(.{.label});
        const compiled = try runtime.Compiled.init(session, predict, .{ input, weights, bias });
        errdefer compiled.deinit(session.allocator);
        var x: zml.Buffer = try .fromBytes(session.io, session.platform, input.shape(), std.mem.asBytes(&[2]f32{ 2, 3 }));
        errdefer x.deinit();
        var w: zml.Buffer = try .fromBytes(session.io, session.platform, weights.shape(), std.mem.asBytes(&[4]f32{ 1, 2, -2, 1 }));
        errdefer w.deinit();
        const b: zml.Buffer = try .fromBytes(session.io, session.platform, bias.shape(), std.mem.asBytes(&[2]f32{ -1, -1 }));
        return .{ .session = session, .compiled = compiled, .x = x, .w = w, .b = b };
    }

    pub fn infer(self: *Model, output: *[2]f32) !u32 {
        try self.compiled.run(.{ self.x, self.w, self.b });
        var result = self.compiled.results.get(zml.Buffer);
        defer result.deinit();
        const shards = try self.session.verifyPlacement(&result);
        output.* = try result.getValue([2]f32, self.session.io);
        return shards;
    }

    pub fn deinit(self: *Model) void {
        self.compiled.deinit(self.session.allocator);
        self.b.deinit();
        self.w.deinit();
        self.x.deinit();
    }
};
