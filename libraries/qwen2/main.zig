const std = @import("std");
const qwen = @import("zml_lib/qwen_decoder");
pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !void {
    try qwen.run(qwen.qwen2_05b, init);
}
