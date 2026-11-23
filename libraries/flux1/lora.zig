const std = @import("std");
const zml = @import("zml/core");
const runtime = @import("zml_lib/runtime");
const Tensor = zml.Tensor;
const layers = runtime.layers;
const transformer = runtime.transformer;

pub const Input = struct {
    manifest: []const u8,
    weights: []const u8,
};

const Patch = struct {
    index: usize,
    offset: i64,
    rows: i64,
    rank: i64,
    scale: f32,
};

pub const Graph = struct {
    patches: []const Patch = &.{},
    factors: []const Tensor = &.{},

    pub fn apply(self: Graph, index: usize, x: Tensor, base: Tensor) Tensor {
        var result = base;
        const width = base.dim(base.rank() - 1);
        for (self.patches, 0..) |patch, i| {
            if (patch.index != index) continue;
            const hidden = transformer.linearParameters(x, self.factors[2 * i], null);
            const delta = transformer.linearParameters(hidden, self.factors[2 * i + 1], null).scale(patch.scale);
            const update = if (patch.rows == width) delta else blk: {
                const rows = @divExact(@as(i64, @intCast(delta.shape().count())), patch.rows);
                break :blk delta.reshape(.{ .tokens = rows, .channels = patch.rows })
                    .pad(@as(f32, 0), .{ .channels = Tensor.Pad{ .low = patch.offset, .high = width - patch.rows - patch.offset } })
                    .reshape(base.shape());
            };
            result = result.add(update);
        }
        return result;
    }
};

const FactorSpecs = struct {
    base: []const layers.Parameter,
    patches: []const Patch,

    fn at(specs: FactorSpecs, index: usize, scratch: []u8) layers.Parameter {
        const patch = specs.patches[index / 2];
        const base = specs.base[patch.index].shape;
        const down = index % 2 == 0;
        const key = layers.name(scratch, "{s}.{d}", .{ if (down) "down" else "up", index / 2 });
        const shape: zml.Shape = if (down) .init(.{ patch.rank, base.dim(1) }, .f32) else .init(.{ patch.rows, patch.rank }, .f32);
        return .{ .name = key, .shape = shape };
    }
};

pub const Parameters = struct {
    manifest: std.json.Parsed([]Patch),
    factors: runtime.checkpoint.Parameters,

    pub fn load(session: *const runtime.Session, schema: []const layers.Parameter, input: Input) !Parameters {
        const allocator = session.allocator;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(session.io, input.manifest, allocator, .limited(1024 * 1024));
        defer allocator.free(bytes);
        const manifest = try std.json.parseFromSlice([]Patch, allocator, bytes, .{});
        errdefer manifest.deinit();
        if (manifest.value.len == 0) return error.EmptyLoRA;
        for (manifest.value) |patch| {
            if (patch.index >= schema.len) return error.InvalidLoRATarget;
            const target = schema[patch.index];
            if (!std.mem.endsWith(u8, target.name, ".weight") or target.shape.rank() != 2)
                return error.InvalidLoRATarget;
            const width = target.shape.dim(0);
            if (patch.rank < 1 or patch.rank > @min(target.shape.dim(1), patch.rows) or
                patch.offset < 0 or patch.rows < 1 or patch.rows > width or patch.offset > width - patch.rows or
                !std.math.isFinite(patch.scale)) return error.InvalidLoRAShape;
        }
        const factors = try runtime.checkpoint.load(
            allocator,
            session.io,
            session.platform,
            manifest.value.len * 2,
            FactorSpecs{ .base = schema, .patches = manifest.value },
            FactorSpecs.at,
            input.weights,
        );
        return .{ .manifest = manifest, .factors = factors };
    }

    pub fn graph(self: Parameters) Graph {
        return .{ .patches = self.manifest.value, .factors = self.factors.tensors };
    }

    pub fn deinit(self: Parameters, allocator: std.mem.Allocator) void {
        self.factors.deinit(allocator);
        self.manifest.deinit();
    }
};
