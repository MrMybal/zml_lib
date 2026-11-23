const std = @import("std");
const rt = @import("zml_lib/runtime");
const model = @import("zml_lib/vitpose");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 1) {
        if (std.mem.eql(u8, args[1], "udp")) return rt.probe_io.poseProbe(model.HeatmapDecoder, init, args[2..]);
        if (std.mem.eql(u8, args[1], "joints")) return rt.probe_io.poseProbe(model.Joints, init, args[2..]);
        if (std.mem.eql(u8, args[1], "verify-joints")) return rt.probe_io.poseProbe(model.JointsAndHeatmaps, init, args[2..]);
    }
    return rt.probe_io.poseProbe(model, init, args[1..]);
}
