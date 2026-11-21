const std = @import("std");
const runtime = @import("zml_lib/runtime");
const realesrgan = @import("zml_lib/realesrgan");
const mnist = @import("zml_lib/mnist");
const smoke = @import("smoke.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const allocator = std.heap.smp_allocator;

const Status = enum(u32) {
    ok = 0,
    invalid_argument = 1,
    no_model = 2,
    backend_error = 3,
    load_error = 4,
    inference_error = 5,
    out_of_memory = 6,
};

const Handle = struct {
    io: std.Io.Threaded,
    session: runtime.Session,
    model: ?Loaded = null,
    checked_output_shards: u32 = 0,
};

const Loaded = union(enum) {
    realesrgan: realesrgan.Model,
    mnist: mnist.Model,
    smoke: smoke.Model,
};

const DeviceInfo = extern struct {
    backend: u32,
    device_count: u32,
    checked_output_shards: u32,
    pjrt_platform: [32]u8,
    device: [256]u8,
};

threadlocal var last_error: ?[*:0]const u8 = null;

fn fail(status: Status, err: anyerror) u32 {
    last_error = @errorName(err).ptr;
    return @backingInt(if (err == error.OutOfMemory) Status.out_of_memory else status);
}

fn success() u32 {
    last_error = null;
    return @backingInt(Status.ok);
}

export fn zml_lib_last_error() ?[*:0]const u8 {
    return last_error;
}

export fn zml_lib_session_open(backend: u32, out: ?*?*Handle) u32 {
    const result = out orelse return fail(.invalid_argument, error.NullOutputHandle);
    result.* = null;
    const selected = std.enums.fromInt(runtime.Backend, backend) orelse return fail(.invalid_argument, error.InvalidBackend);
    const handle = allocator.create(Handle) catch |err| return fail(.out_of_memory, err);
    handle.io = .init(allocator, .{});
    handle.session = runtime.Session.open(allocator, handle.io.io(), selected) catch |err| {
        handle.io.deinit();
        allocator.destroy(handle);
        return fail(.backend_error, err);
    };
    handle.model = null;
    handle.checked_output_shards = 0;
    result.* = handle;
    return success();
}

export fn zml_lib_load_model(handle: ?*Handle, variant: u32, weights: ?[*:0]const u8) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    const selected = std.enums.fromInt(realesrgan.Variant, variant) orelse return fail(.invalid_argument, error.InvalidVariant);
    const path = std.mem.span(weights orelse return fail(.invalid_argument, error.NullWeightsPath));
    if (path.len == 0 or !std.unicode.utf8ValidateSlice(path))
        return fail(.invalid_argument, error.InvalidWeightsPath);
    const replacement = realesrgan.Model.load(&self.session, selected, path) catch |err| return fail(.load_error, err);
    zml_lib_unload_model(self);
    self.model = .{ .realesrgan = replacement };
    return success();
}

export fn zml_lib_infer(
    handle: ?*Handle,
    width: u32,
    height: u32,
    input: ?[*]const f32,
    input_count: usize,
    output: ?[*]f32,
    output_count: usize,
) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    self.checked_output_shards = 0;
    const loaded = if (self.model) |*model| model else return fail(.no_model, error.NoModel);
    if (loaded.* != .realesrgan) return fail(.no_model, error.WrongModel);
    const model = &loaded.realesrgan;
    const size: realesrgan.Size = .{ .width = width, .height = height };
    const out_size = size.scaled(model.variant.scale()) catch |err| return fail(.invalid_argument, err);
    const expected_input = size.elements() catch |err| return fail(.invalid_argument, err);
    const expected_output = out_size.elements() catch |err| return fail(.invalid_argument, err);
    if (input_count != expected_input or output_count != expected_output)
        return fail(.invalid_argument, error.InvalidBufferSize);
    const src = input orelse return fail(.invalid_argument, error.NullInputBuffer);
    const dst = output orelse return fail(.invalid_argument, error.NullOutputBuffer);
    self.checked_output_shards = model.infer(size, src[0..input_count], dst[0..output_count], null) catch |err|
        return fail(.inference_error, err);
    return success();
}

fn copyString(destination: []u8, source: []const u8) void {
    @memset(destination, 0);
    const len = @min(destination.len - 1, source.len);
    @memcpy(destination[0..len], source[0..len]);
}

export fn zml_lib_session_info(handle: ?*Handle, out: ?*DeviceInfo) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    const info = out orelse return fail(.invalid_argument, error.NullOutputBuffer);
    const platform = self.session.platform;
    info.backend = @backingInt(std.meta.stringToEnum(runtime.Backend, self.session.target()) orelse .auto);
    info.device_count = @intCast(platform.devices.len);
    info.checked_output_shards = self.checked_output_shards;
    copyString(&info.pjrt_platform, self.session.pjrtPlatform());
    copyString(&info.device, platform.devices[0].debugString());
    return success();
}

export fn zml_lib_add(handle: ?*Handle, lhs: f32, rhs: f32, output: ?*f32) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    self.checked_output_shards = 0;
    const dst = output orelse return fail(.invalid_argument, error.NullOutputBuffer);
    self.checked_output_shards = smoke.add(&self.session, lhs, rhs, dst) catch |err| return fail(.inference_error, err);
    return success();
}

export fn zml_lib_load_smoke(handle: ?*Handle) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    const replacement = smoke.Model.load(&self.session) catch |err| return fail(.load_error, err);
    zml_lib_unload_model(self);
    self.model = .{ .smoke = replacement };
    return success();
}

export fn zml_lib_infer_smoke(handle: ?*Handle, output: ?*[2]f32) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    self.checked_output_shards = 0;
    const loaded = if (self.model) |*model| model else return fail(.no_model, error.NoModel);
    if (loaded.* != .smoke) return fail(.no_model, error.WrongModel);
    const dst = output orelse return fail(.invalid_argument, error.NullOutputBuffer);
    self.checked_output_shards = loaded.smoke.infer(dst) catch |err| return fail(.inference_error, err);
    return success();
}

export fn zml_lib_load_mnist(handle: ?*Handle, weights: ?[*:0]const u8) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    const path = std.mem.span(weights orelse return fail(.invalid_argument, error.NullWeightsPath));
    if (path.len == 0 or !std.unicode.utf8ValidateSlice(path))
        return fail(.invalid_argument, error.InvalidWeightsPath);
    const replacement = mnist.Model.load(&self.session, path) catch |err| return fail(.load_error, err);
    zml_lib_unload_model(self);
    self.model = .{ .mnist = replacement };
    return success();
}

export fn zml_lib_infer_mnist(handle: ?*Handle, pixels: ?[*]const u8, len: usize, digit: ?*u8) u32 {
    const self = handle orelse return fail(.invalid_argument, error.NullSession);
    self.checked_output_shards = 0;
    const loaded = if (self.model) |*model| model else return fail(.no_model, error.NoModel);
    if (loaded.* != .mnist) return fail(.no_model, error.WrongModel);
    const source = pixels orelse return fail(.invalid_argument, error.NullInputBuffer);
    const out = digit orelse return fail(.invalid_argument, error.NullOutputBuffer);
    if (len != mnist.image_bytes) return fail(.invalid_argument, error.InvalidBufferSize);
    const result = loaded.mnist.infer(source[0..len]) catch |err| return fail(.inference_error, err);
    self.checked_output_shards = result.shards;
    out.* = result.digit;
    return success();
}

export fn zml_lib_unload_model(handle: ?*Handle) void {
    if (handle) |self| {
        if (self.model) |*model| switch (model.*) {
            inline else => |*loaded| loaded.deinit(),
        };
        self.checked_output_shards = 0;
        self.model = null;
    }
}

export fn zml_lib_session_close(handle: ?*Handle) void {
    if (handle) |self| {
        zml_lib_unload_model(self);
        self.session.close();
        self.io.deinit();
        allocator.destroy(self);
    }
}
