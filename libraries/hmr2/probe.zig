const std = @import("std");
const rt = @import("zml_lib/runtime");
const model = @import("zml_lib/hmr2");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    return rt.probe_io.poseProbe(model, init, args[1..]);
}
