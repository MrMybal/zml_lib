const std = @import("std");
const runtime = @import("zml_lib/realesrgan");

extern fn stbi_load_from_memory([*]const u8, c_int, *c_int, *c_int, *c_int, c_int) ?[*]u8;
extern fn stbi_image_free(?*anyopaque) void;
extern fn stbi_write_png_to_func(
    *const fn (?*anyopaque, ?[*]const u8, c_int) callconv(.c) void,
    ?*anyopaque,
    c_int,
    c_int,
    c_int,
    [*]const u8,
    isize,
) c_int;

pub const Image = struct { size: runtime.Size, pixels: []f32 };

pub fn read(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Image {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(std.math.maxInt(c_int)));
    defer allocator.free(bytes);
    var width: c_int = 0;
    var height: c_int = 0;
    var channels: c_int = 0;
    const pixels = stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &width, &height, &channels, 3) orelse
        return error.ImageDecodeFailed;
    defer stbi_image_free(pixels);
    if (width <= 0 or height <= 0) return error.InvalidDimensions;
    const size: runtime.Size = .{ .width = @intCast(width), .height = @intCast(height) };
    const count = try size.elements();
    const plane = count / 3;
    const rgb = try allocator.alloc(f32, count);
    for (0..plane) |pixel| {
        for (0..3) |channel| {
            rgb[channel * plane + pixel] = @as(f32, @floatFromInt(pixels[3 * pixel + channel])) / 255.0;
        }
    }
    return .{ .size = size, .pixels = rgb };
}

pub fn writePng(
    allocator: std.mem.Allocator,
    size: runtime.Size,
    rgb: []const f32,
    writer: *std.Io.Writer,
) !void {
    const WriteContext = struct {
        writer: *std.Io.Writer,
        failed: bool = false,

        fn write(context: ?*anyopaque, data: ?[*]const u8, length: c_int) callconv(.c) void {
            const self: *@This() = @ptrCast(@alignCast(context.?));
            self.writer.writeAll(data.?[0..@intCast(length)]) catch {
                self.failed = true;
            };
        }
    };

    const length = try size.elements();
    if (rgb.len != length) return error.InvalidBufferSize;
    const width = std.math.cast(c_int, size.width) orelse return error.InvalidDimensions;
    const height = std.math.cast(c_int, size.height) orelse return error.InvalidDimensions;
    const pitch = try std.math.mul(c_int, width, 3);
    // stb_image_write uses int for the filtered buffer size and PNG chunk lengths.
    if (length > @divTrunc(std.math.maxInt(c_int), 2)) return error.InvalidDimensions;

    const pixels = try allocator.alloc(u8, length);
    defer allocator.free(pixels);
    const plane = length / 3;
    for (0..plane) |pixel| {
        for (0..3) |channel| {
            const value = rgb[channel * plane + pixel];
            if (!std.math.isFinite(value)) return error.NonFiniteOutput;
            const scaled = @min(1.0, @max(0.0, value)) * 255.0;
            const floor = @floor(scaled);
            const base: u8 = @intFromFloat(floor);
            const remainder = scaled - floor;
            pixels[pixel * 3 + channel] = base + @as(u8, @intFromBool(remainder > 0.5 or
                (remainder == 0.5 and base % 2 == 1)));
        }
    }
    var context: WriteContext = .{ .writer = writer };
    if (stbi_write_png_to_func(WriteContext.write, &context, width, height, 3, pixels.ptr, pitch) == 0)
        return error.PngEncodeFailed;
    if (context.failed) return error.WriteFailed;
}
