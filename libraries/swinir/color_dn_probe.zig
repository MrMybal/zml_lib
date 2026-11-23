const std = @import("std");
const runtime = @import("zml_lib/runtime");
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    try runtime.probe_io.restorationProbe(@import("zml_lib/swinir").ColorDenoise, init);
}
