const std = @import("std");
const runtime = @import("zml_lib/runtime");
const l = runtime.layers;
const t = runtime.transformer;
const Tensor = l.Tensor;
pub const max_frames = 32;
const widths = [_]i64{ 192, 384, 64, 64 };

fn param(name: []const u8, dims: anytype) l.Parameter {
    return .{ .name = name, .shape = .init(dims, .f32) };
}

fn pair(comptime name: []const u8, dims: anytype, c: i64) [2]l.Parameter {
    return .{ param(name ++ ".weight", dims), param(name ++ ".bias", .{c}) };
}

pub const schema = blk: {
    @setEvalBranchQuota(80_000);
    var specs: [4 * 28]l.Parameter = undefined;
    for (widths, 0..) |c, i| {
        const module = std.fmt.comptimePrint("head.motion_modules.{d}.temporal_transformer", .{i});
        const block = module ++ ".transformer_blocks.0";
        specs[28 * i ..][0..12].* =
            pair(module ++ ".norm", .{c}, c) ++ pair(module ++ ".proj_in", .{ c, c }, c) ++
            pair(module ++ ".proj_out", .{ c, c }, c) ++ pair(block ++ ".ff_norm", .{c}, c) ++
            pair(block ++ ".ff.net.0.proj", .{ c * 8, c }, c * 8) ++ pair(block ++ ".ff.net.2", .{ c, c * 4 }, c);
        for (0..2) |a| {
            const sub = block ++ std.fmt.comptimePrint(".attention_blocks.{d}", .{a});
            const norm_name = block ++ std.fmt.comptimePrint(".norms.{d}", .{a});
            specs[28 * i + 12 + 8 * a ..][0..8].* = [_]l.Parameter{
                param(sub ++ ".to_q.weight", .{ c, c }),
                param(sub ++ ".to_k.weight", .{ c, c }),
                param(sub ++ ".to_v.weight", .{ c, c }),
                param(sub ++ ".pos_encoder.pe", .{ 1, max_frames, c }),
            } ++ pair(sub ++ ".to_out.0", .{ c, c }, c) ++ pair(norm_name, .{c}, c);
        }
    }
    break :blk specs;
};

fn attention(w: l.Weights, block: []const u8, i: usize, x: Tensor) Tensor {
    var name: [128]u8 = undefined;
    var key: [128]u8 = undefined;
    const base = l.name(&name, "{s}.attention_blocks.{d}", .{ block, i });
    const normalized = t.norm(w, l.name(&key, "{s}.norms.{d}", .{ block, i }), x, 1e-5)
        .transpose(.{ 1, 0, 2 });
    const pos = w.get(base, ".pos_encoder.pe").slice(1, .{ .end = normalized.dim(1) });
    const encoded = normalized.add(pos.broadcast(normalized.shape(), &.{ 0, 1, 2 }));
    const q = t.linear(w, l.name(&key, "{s}.to_q", .{base}), encoded, false);
    const k = t.linear(w, l.name(&key, "{s}.to_k", .{base}), encoded, false);
    const v = t.linear(w, l.name(&key, "{s}.to_v", .{base}), encoded, false);
    const result = t.linear(w, l.name(&key, "{s}.to_out.0", .{base}), t.attention(q, k, v, 8), true);
    return x.add(result.transpose(.{ 1, 0, 2 }));
}

pub fn forward(w: l.Weights, i: usize, input: Tensor) Tensor {
    const frames, const c, const h, const width = input.dims()[0..4].*;
    var module_name: [96]u8 = undefined;
    var block_name: [96]u8 = undefined;
    var key: [128]u8 = undefined;
    const module = l.name(&module_name, "head.motion_modules.{d}.temporal_transformer", .{i});
    const block = l.name(&block_name, "{s}.transformer_blocks.0", .{module});
    const normalized = l.groupNorm(w, l.name(&key, "{s}.norm", .{module}), input, 32, 1e-6)
        .transpose(.{ 0, 2, 3, 1 }).reshape(.{ frames, h * width, c });
    var x = t.linear(w, l.name(&key, "{s}.proj_in", .{module}), normalized, true);
    for (0..2) |a| x = attention(w, block, a, x);
    const ff = t.norm(w, l.name(&key, "{s}.ff_norm", .{block}), x, 1e-5);
    const values = t.linear(w, l.name(&key, "{s}.ff.net.0.proj", .{block}), ff, true);
    const gated = values.slice(2, .{ .end = c * 4 }).mul(t.gelu(values.slice(2, .{ .start = c * 4 })));
    x = x.add(t.linear(w, l.name(&key, "{s}.ff.net.2", .{block}), gated, true));
    const result = t.linear(w, l.name(&key, "{s}.proj_out", .{module}), x, true)
        .reshape(.{ frames, h, width, c }).transpose(.{ 0, 3, 1, 2 });
    return input.add(result);
}
