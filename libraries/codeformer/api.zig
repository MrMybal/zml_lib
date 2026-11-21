const std = @import("std");
const runtime = @import("zml_lib/runtime");
const retina = @import("zml_lib/retinaface");
const parser = @import("zml_lib/parsenet");
const former = @import("zml_lib/codeformer");

pub const std_options: std.Options = .{ .log_level = .info };
const allocator = std.heap.smp_allocator;

const ModelId = enum(u32) { retinaface = 1, parsenet = 2, codeformer = 3 };
const Handle = struct {
    io: std.Io.Threaded,
    session: runtime.Session,
    detector: ?retina.Model = null,
    parser: ?parser.Model = null,
    restorer: ?former.Model = null,
    shards: u32 = 0,
};

const DeviceInfo = extern struct {
    backend: u32,
    output_shards: u32,
    pjrt_platform: [32]u8,
    device: [256]u8,
};

threadlocal var last_error: ?[*:0]const u8 = null;

fn status(result: anyerror!void) u32 {
    result catch |err| {
        last_error = @errorName(err).ptr;
        return 1;
    };
    last_error = null;
    return 0;
}

export fn zml_faces_last_error() ?[*:0]const u8 {
    return last_error;
}

fn open(backend: u32, result: ?*?*Handle) !void {
    const out = result orelse return error.NullOutputHandle;
    out.* = null;
    const selected = std.enums.fromInt(runtime.Backend, backend) orelse return error.InvalidBackend;
    const handle = try allocator.create(Handle);
    errdefer allocator.destroy(handle);
    handle.io = .init(allocator, .{});
    errdefer handle.io.deinit();
    handle.session = try runtime.Session.open(allocator, handle.io.io(), selected);
    handle.detector = null;
    handle.parser = null;
    handle.restorer = null;
    handle.shards = 0;
    out.* = handle;
}

export fn zml_faces_open(backend: u32, result: ?*?*Handle) u32 {
    return status(open(backend, result));
}

export fn zml_faces_close(handle: ?*Handle) void {
    if (handle) |self| {
        if (self.restorer) |*model| model.deinit();
        if (self.parser) |*model| model.deinit();
        if (self.detector) |*model| model.deinit();
        self.session.close();
        self.io.deinit();
        allocator.destroy(self);
    }
    last_error = null;
}

fn load(handle: ?*Handle, model: u32, weights: ?[*:0]const u8) !void {
    const self = handle orelse return error.NullSession;
    const path = std.mem.span(weights orelse return error.NullWeights);
    if (path.len == 0 or !std.unicode.utf8ValidateSlice(path)) return error.InvalidWeightsPath;
    switch (std.enums.fromInt(ModelId, model) orelse return error.InvalidModel) {
        .retinaface => {
            const replacement = try retina.Model.load(&self.session, path);
            if (self.detector) |*old| old.deinit();
            self.detector = replacement;
        },
        .parsenet => {
            const replacement = try parser.Model.load(&self.session, path);
            if (self.parser) |*old| old.deinit();
            self.parser = replacement;
        },
        .codeformer => {
            const replacement = try former.Model.load(&self.session, path);
            if (self.restorer) |*old| old.deinit();
            self.restorer = replacement;
        },
    }
    self.shards = 0;
}

export fn zml_faces_load(handle: ?*Handle, model: u32, weights: ?[*:0]const u8) u32 {
    return status(load(handle, model, weights));
}

fn infer(
    handle: ?*Handle,
    model: u32,
    width: u32,
    height: u32,
    fidelity: f32,
    input: ?[*]const f32,
    input_count: usize,
    output: ?[*]f32,
    output_count: usize,
) !void {
    const self = handle orelse return error.NullSession;
    self.shards = 0;
    const src = (input orelse return error.NullInput)[0..input_count];
    const dst = (output orelse return error.NullOutput)[0..output_count];
    const selected = std.enums.fromInt(ModelId, model) orelse return error.InvalidModel;
    if (selected != .retinaface and (width != 512 or height != 512)) return error.InvalidDimensions;
    self.shards = switch (selected) {
        .retinaface => if (self.detector) |*value| try value.infer(width, height, src, dst) else return error.NoModel,
        .parsenet => if (self.parser) |*value| try value.infer(src, dst) else return error.NoModel,
        .codeformer => if (self.restorer) |*value| try value.infer(src, fidelity, dst) else return error.NoModel,
    };
}

export fn zml_faces_infer(
    handle: ?*Handle,
    model: u32,
    width: u32,
    height: u32,
    fidelity: f32,
    input: ?[*]const f32,
    input_count: usize,
    output: ?[*]f32,
    output_count: usize,
) u32 {
    return status(infer(handle, model, width, height, fidelity, input, input_count, output, output_count));
}

fn decode(width: u32, height: u32, raw: ?[*]const f32, input_len: usize, output: ?[*]f32, output_len: usize, count: ?*usize) !void {
    const result = count orelse return error.NullCount;
    result.* = 0;
    const input = (raw orelse return error.NullInput)[0..input_len];
    const faces = (output orelse return error.NullOutput)[0..output_len];
    result.* = try retina.decode(allocator, width, height, input, faces);
}

export fn zml_faces_decode(
    width: u32,
    height: u32,
    raw: ?[*]const f32,
    input_len: usize,
    output: ?[*]f32,
    output_len: usize,
    count: ?*usize,
) u32 {
    return status(decode(width, height, raw, input_len, output, output_len, count));
}

fn copyZ(dest: []u8, value: []const u8) void {
    @memset(dest, 0);
    const size = @min(dest.len - 1, value.len);
    @memcpy(dest[0..size], value[0..size]);
}

fn info(handle: ?*const Handle, out: ?*DeviceInfo) !void {
    const self = handle orelse return error.NullSession;
    const result = out orelse return error.NullOutput;
    result.backend = @backingInt(std.meta.stringToEnum(runtime.Backend, self.session.target()) orelse .auto);
    result.output_shards = self.shards;
    copyZ(&result.pjrt_platform, self.session.pjrtPlatform());
    copyZ(&result.device, self.session.platform.devices[0].debugString());
}

export fn zml_faces_info(handle: ?*const Handle, out: ?*DeviceInfo) u32 {
    return status(info(handle, out));
}
