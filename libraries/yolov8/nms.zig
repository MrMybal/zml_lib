const zml = @import("zml/core");
const letterbox = @import("letterbox.zig");
const Tensor = zml.Tensor;

pub const fields = 84;
pub const max_cells = 8400;
pub const limit = 300;
pub const width = 6;
const score_threshold = 0.5;
const iou_threshold = 0.7;

fn f(value: f32) Tensor {
    return .scalar(value, .f32);
}

fn i(value: i32) Tensor {
    return .scalar(value, .i32);
}

fn row(matrix: Tensor, index: i64) Tensor {
    return matrix.slice(0, .single(index));
}

const Greedy = struct {
    pub const State = struct {
        count: Tensor,
        scores: Tensor,
        best: Tensor.ArgMaxRes,
        boxes: Tensor,
        areas: Tensor,
        kept: Tensor,
    };

    pub fn cond(_: @This(), state: State) Tensor {
        return state.count.cmp(.LT, i(limit)).logical(.AND, state.best.values.reshape(.{}).cmp(.GT, f(score_threshold)));
    }

    pub fn body(_: @This(), state: State) State {
        const index = state.best.indices.reshape(.{});
        const selected = state.boxes.slice(1, .dyn(index, 1));
        const left = row(state.boxes, 0).maximum(row(selected, 0));
        const top = row(state.boxes, 1).maximum(row(selected, 1));
        const right = row(state.boxes, 2).minimum(row(selected, 2));
        const bottom = row(state.boxes, 3).minimum(row(selected, 3));
        const overlap = right.sub(left).maximum(f(0)).mul(bottom.sub(top).maximum(f(0)));
        const area = state.areas.slice(0, .dyn(index, 1));
        const union_area = state.areas.add(area).sub(overlap);
        const suppressed = overlap.div(union_area).cmp(.GT, f(iou_threshold));
        const anchors = Tensor.iota(state.scores.shape().withDtype(.i32), 0);
        const scores = suppressed.logical(.OR, anchors.cmp(.EQ, index)).select(f(0), state.scores);
        return .{
            .count = state.count.add(i(1)),
            .scores = scores,
            .best = scores.argMax(0),
            .boxes = state.boxes,
            .areas = state.areas,
            .kept = state.kept.dynamicUpdateSlice1d(index.reshape(.{1}), 0, state.count),
        };
    }
};

pub fn forward(flat: Tensor) Tensor {
    const input_cells = @divExact(flat.dim(0), fields);
    // One stride-aligned YOLO cell contributes 21 anchors. Padding smaller
    // standalone inputs keeps the reductions on the same graph path.
    const cells = @max(21, input_cells);
    const input = flat.reshape(.{ fields, input_cells });
    const predictions = if (cells == input_cells) input else input.withTags(.{ .field, .anchor }).pad(@as(f32, 0), .{
        .anchor = Tensor.Pad{ .high = cells - input_cells },
    });
    const scores = row(predictions, 4);
    const other = predictions.slice(0, .{ .start = 5 }).max(0).reshape(.{cells});
    // The person class is index zero. A tie for the best class keeps that class.
    const candidates = scores.cmp(.GT, f(score_threshold)).logical(.AND, scores.cmp(.GE, other)).select(scores, f(0));
    const cx = row(predictions, 0);
    const cy = row(predictions, 1);
    const dx = row(predictions, 2).scale(@as(f32, 0.5));
    const dy = row(predictions, 3).scale(@as(f32, 0.5));
    const x1 = cx.sub(dx);
    const y1 = cy.sub(dy);
    const x2 = cx.add(dx);
    const y2 = cy.add(dy);
    const boxes = Tensor.stack(&.{ x1, y1, x2, y2 }, 0, .coordinate);
    const result = zml.ops.while_(Greedy, .{}, .{
        .count = i(0),
        .scores = candidates,
        .best = candidates.argMax(0),
        .boxes = boxes,
        .areas = x2.sub(x1).mul(y2.sub(y1)),
        .kept = i(-1).broad(.init(.{limit}, .i32)),
    });

    const indices = result.kept.maximum(i(0)).withTags(.{.detection});
    const rows = Tensor.stack(&.{ x1, y1, x2, y2, scores }, 1, .field).withTags(.{ .anchor, .field });
    const picked = rows.gather(.{ .anchor = indices }, .{}).reshape(.{ limit, 5 });
    const visible = result.kept.cmp(.GE, i(0)).broadcast(picked.shape().withDtype(.bool), &.{0});
    const output = Tensor.concatenate(&.{ visible.select(picked, f(0)), result.kept.convert(.f32).reshape(.{ limit, 1 }) }, 1);
    const finite = flat.isFinite().convert(.i32).min(0).reshape(.{}).cmp(.EQ, i(1));
    return finite.broad(output.shape().withDtype(.bool)).select(output, f(-2));
}

pub fn forwardSource(transform: letterbox.SourceTransform, flat: Tensor) Tensor {
    const chosen = forward(flat);
    const coords = chosen.slice(1, .{ .end = 4 });
    const fields_out = chosen.slice(1, .{ .start = 4 });
    const padding = Tensor.stack(&.{ f(transform.pad_x), f(transform.pad_y), f(transform.pad_x), f(transform.pad_y) }, 0, .coordinate);
    const bounds = Tensor.stack(&.{ f(transform.width), f(transform.height), f(transform.width), f(transform.height) }, 0, .coordinate);
    const xy = coords.sub(padding.broadcast(coords.shape(), &.{1})).div(f(transform.gain));
    const clipped = xy.maximum(f(0)).minimum(bounds.broadcast(coords.shape(), &.{1}));
    // Keep the padding and nonfinite-input sentinel in their original form.
    const kept = fields_out.slice(1, .{ .start = 1 }).cmp(.GE, f(0)).broad(coords.shape().withDtype(.bool));
    return Tensor.concatenate(&.{ kept.select(clipped, coords), fields_out }, 1);
}
