const std = @import("std");
const builtin = @import("builtin");
const zml = @import("zml/core");

pub const checkpoint = @import("checkpoint.zig");
pub const kl_decoder = @import("kl_decoder.zig");
pub const layers = @import("layers.zig");
pub const probe_io = @import("probe_io.zig");
pub const spatial = @import("spatial.zig");
pub const TensorModel = @import("tensor_model.zig").TensorModel;
pub const transformer = @import("transformer.zig");
pub const pose_vit = @import("pose_vit.zig");
pub const Backend = enum(u32) { auto = 0, cpu = 1, cuda = 2, rocm = 3, vulkan = 4 };
var platform_init_lock: std.Io.Mutex = .init;

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, backend: Backend) !Session {
        platform_init_lock.lockUncancelable(io);
        defer platform_init_lock.unlock(io);
        const options: zml.platform.CreateOptions = .{ .cpu = .{ .device_count = 1 } };
        const platform = switch (backend) {
            .auto => try zml.Platform.auto(allocator, io, options),
            .cpu => try zml.Platform.init(allocator, io, .cpu, options),
            .cuda => try zml.Platform.init(allocator, io, .cuda, options),
            .rocm => try zml.Platform.init(allocator, io, .rocm, options),
            .vulkan => try zml.Platform.init(allocator, io, .vulkan, options),
        };
        return .{ .allocator = allocator, .io = io, .platform = platform };
    }

    pub fn close(self: Session) void {
        self.platform.deinit(self.allocator, self.io);
    }

    pub fn target(self: Session) []const u8 {
        return @tagName(self.platform.target);
    }

    pub fn pjrtPlatform(self: Session) []const u8 {
        return self.platform.pjrt_client.platformName(self.platform.pjrt_api);
    }

    pub fn verifyPlacement(self: Session, buffer: *const zml.Buffer) !u32 {
        const platform = self.platform;
        var shards = buffer.shards();
        var seen: u32 = 0;
        while (shards.next()) |shard| {
            if (shard._pjrt_buffer.isOnCpu(platform.pjrt_api) != (platform.target == .cpu))
                return error.UnexpectedOutputPlacement;
            const device = try shard._pjrt_buffer.device(platform.pjrt_api);
            if (seen >= platform.physical_mesh.devices_in_canonical_order.len)
                return error.UnexpectedOutputPlacement;
            const placement = platform.physical_mesh.devices_in_canonical_order[seen];
            if (device != platform.devices[placement.id].pjrt_device)
                return error.UnexpectedOutputPlacement;
            seen += 1;
        }
        if (seen == 0) return error.UnexpectedOutputPlacement;
        return seen;
    }
};

pub const Compiled = struct {
    exe: zml.Exe,
    args: zml.Exe.Arguments,
    results: zml.Exe.Results,

    pub fn init(session: *const Session, comptime forward: anytype, parameters: std.meta.ArgsTuple(@TypeOf(forward))) !Compiled {
        // IREE's Windows CUDA loader expects the caller's CUDA context when it
        // loads modules. Keep the PJRT compile callback on the session thread.
        var inline_io: std.Io.Threaded = .init_single_threaded;
        const compile_io = if (builtin.os.tag == .windows and session.platform.target == .cuda and
            std.mem.eql(u8, session.pjrtPlatform(), "iree_cuda")) inline_io.io() else session.io;
        const exe = try session.platform.compileFn(
            session.allocator,
            compile_io,
            forward,
            parameters,
            .{},
        );
        errdefer exe.deinit();
        const args = try exe.args(session.allocator);
        errdefer args.deinit(session.allocator);
        const results = try exe.results(session.allocator);
        return .{ .exe = exe, .args = args, .results = results };
    }

    pub fn deinit(self: *const Compiled, allocator: std.mem.Allocator) void {
        self.results.deinit(allocator);
        self.args.deinit(allocator);
        self.exe.deinit();
    }

    pub fn run(self: *Compiled, inputs: anytype) !void {
        self.args.set(inputs);
        // Exe.call panics on a PJRT error. These graphs do not donate input buffers.
        try self.exe.exe.execute(self.exe.platform.pjrt_api, .{
            .arguments = self.args.flat_buffers.buffers,
            .num_args = self.args.expected_shapes.len,
            .results = self.results.flat_buffers.buffers,
            .events = null,
            .context = self.exe.context,
        });
    }
};
