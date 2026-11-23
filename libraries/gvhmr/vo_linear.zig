const zml = @import("zml/core");
const m = @import("post_math.zig");
const Tensor = zml.Tensor;

pub fn s(value: f64) Tensor {
    return .scalar(value, .f64);
}

pub fn safe(x: Tensor, epsilon: f64) Tensor {
    const size = x.abs().maximum(s(epsilon));
    return x.cmp(.LT, s(0)).select(size.negate(), size);
}

pub fn scaleRows(x: Tensor, gain: Tensor) Tensor {
    return x.mul(gain.broadcast(x.shape(), &.{ 0, 1 }));
}

pub fn unit(x: Tensor) Tensor {
    return scaleRows(x, s(1).div(m.norm(x).maximum(s(1e-25))));
}

pub fn transpose(x: Tensor) Tensor {
    return x.transpose(.{ 0, 2, 1 });
}

pub fn outer(x: Tensor, y: Tensor) Tensor {
    return x.dotGeneral(y, &.{}, &.{.{ 0, 0 }});
}

pub fn entry(x: Tensor, row: i64, col: i64) Tensor {
    return m.component(x.slice(1, .single(row)), col);
}

pub fn rows(values: []const Tensor) Tensor {
    const shape = zml.Shape.init(.{ values[0].dim(0), values[0].dim(1) }, values[0].dtype());
    var same: [9]Tensor = undefined;
    for (same[0..values.len], values) |*out, value| out.* = value.withTags(shape);
    return Tensor.stack(same[0..values.len], 1, .row);
}

pub fn columns(values: []const Tensor) Tensor {
    return transpose(rows(values));
}

pub fn eye(n: i64, dim: i64) Tensor {
    const row = Tensor.iota(.init(.{ dim, dim }, .i32), 0);
    return row.cmp(.EQ, Tensor.iota(row.shape(), 1)).convert(.f64).reshape(.{ 1, dim, dim })
        .broad(.init(.{ n, dim, dim }, .f64));
}

pub fn determinant(h: Tensor) Tensor {
    return m.dot(m.column(h, 0), m.cross(m.column(h, 1), m.column(h, 2)));
}

pub fn inverse3(h: Tensor) Tensor {
    const a = m.column(h, 0);
    const b = m.column(h, 1);
    const c = m.column(h, 2);
    const cof = m.cross(b, c);
    const adj = rows(&.{ cof, m.cross(c, a), m.cross(a, b) });
    return adj.div(safe(m.dot(a, cof), 1e-18).broadcast(adj.shape(), &.{ 0, 2 }));
}

const Cholesky = struct {
    pub const State = struct { i: Tensor, remainder: Tensor, lower: Tensor };

    pub fn cond(_: @This(), state: State) Tensor {
        return state.i.cmp(.LT, .scalar(state.lower.dim(1), .i32));
    }

    pub fn body(_: @This(), state: State) State {
        const column = state.remainder.slice(2, .dyn(state.i, 1));
        const pivot = column.slice(1, .dyn(state.i, 1)).maximum(s(1e-18)).sqrt();
        const tail = Tensor.iota(column.shape().withDtype(.i32), 1).cmp(.GE, state.i);
        const next = tail.select(column.div(pivot.broad(column.shape())), s(0));
        const update = outer(next.squeeze(2), next.squeeze(2));
        return .{
            .i = state.i.addConstant(@as(i32, 1)),
            .remainder = state.remainder.sub(update),
            .lower = state.lower.dynamicUpdateSlice1d(next, 2, state.i),
        };
    }
};

pub fn factor(symmetric: Tensor) Tensor {
    return zml.ops.while_(Cholesky, .{}, .{
        .i = .scalar(0, .i32),
        .remainder = symmetric,
        .lower = s(0).broad(symmetric.shape()),
    }).lower;
}

const Triangular = struct {
    reverse: bool,
    pub const State = struct { step: Tensor, matrix: Tensor, work: Tensor };

    pub fn cond(_: @This(), state: State) Tensor {
        return state.step.cmp(.LT, .scalar(state.work.dim(1), .i32));
    }

    pub fn body(self: @This(), state: State) State {
        const i = if (self.reverse) Tensor.scalar(state.work.dim(1) - 1, .i32).sub(state.step) else state.step;
        const column = state.matrix.slice(2, .dyn(i, 1)).squeeze(2);
        const value = state.work.slice(1, .dyn(i, 1)).div(column.slice(1, .dyn(i, 1)));
        const updated = state.work.sub(scaleRows(column, value));
        return .{
            .step = state.step.addConstant(@as(i32, 1)),
            .matrix = state.matrix,
            .work = updated.dynamicUpdateSlice1d(value, 1, i),
        };
    }
};

pub fn solve(lower: Tensor, rhs: Tensor) Tensor {
    const start: Tensor = .scalar(0, .i32);
    const intermediate = zml.ops.while_(Triangular, .{ .reverse = false }, .{ .step = start, .matrix = lower, .work = rhs }).work;
    return zml.ops.while_(Triangular, .{ .reverse = true }, .{ .step = start, .matrix = transpose(lower), .work = intermediate }).work;
}

fn jacobiBasis(n: i64, a: Tensor, p: usize, q: usize) Tensor {
    const b = entry(a, @intCast(p), @intCast(q));
    const delta = entry(a, @intCast(q), @intCast(q)).sub(entry(a, @intCast(p), @intCast(p)));
    const t = delta.div(safe(b.scale(@as(f64, 2)), 1e-25));
    const sign = t.cmp(.LT, s(0)).select(s(-1), s(1));
    const tangent = b.abs().cmp(.GT, s(1e-15)).select(sign.div(t.abs().add(t.mul(t).add(s(1)).sqrt())), s(0));
    const c = s(1).div(tangent.mul(tangent).add(s(1)).sqrt());
    const sine = c.mul(tangent);
    const zero = s(0).broad(c.shape());
    const one = s(1).broad(c.shape());
    var matrix: [3]Tensor = undefined;
    for (&matrix, 0..) |*out, i| {
        var row = [_]Tensor{ zero, zero, zero };
        row[i] = if (i == p or i == q) c else one;
        if (i == p) row[q] = sine;
        if (i == q) row[p] = sine.negate();
        out.* = m.join(&row);
    }
    return rows(&matrix).reshape(.{ n, 3, 3 });
}

pub const Svd = struct { u: Tensor, v: Tensor, values: [3]Tensor };

pub fn svd3(a: Tensor) Svd {
    const n = a.dim(0);
    var vectors = eye(n, 3);
    var symmetric = m.product(transpose(a), a);
    for (0..6) |_| {
        for ([_][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 } }) |pq| {
            const turn = jacobiBasis(n, symmetric, pq[0], pq[1]);
            vectors = m.product(vectors, turn);
            symmetric = m.product(transpose(turn), m.product(symmetric, turn));
        }
    }
    var values: [3]Tensor = undefined;
    var basis: [3]Tensor = undefined;
    for (0..3) |i| {
        values[i] = entry(symmetric, @intCast(i), @intCast(i)).maximum(s(0)).sqrt();
        basis[i] = m.column(vectors, @intCast(i));
    }
    for ([_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 0, 1 } }) |ij| {
        const i, const j = ij;
        const better = values[j].cmp(.GT, values[i]);
        const swap = better.broad(basis[i].shape().withDtype(.bool));
        const d = values[i];
        const v = basis[i];
        values[i] = better.select(values[j], d);
        basis[i] = swap.select(basis[j], v);
        values[j] = better.select(d, values[j]);
        basis[j] = swap.select(v, basis[j]);
    }
    const v0 = unit(basis[0]);
    const v1 = unit(basis[1].sub(scaleRows(v0, m.dot(v0, basis[1]))));
    const first_u = unit(m.apply(a, v0));
    const second_raw = m.apply(a, v1);
    const second_u = unit(second_raw.sub(scaleRows(first_u, m.dot(first_u, second_raw))));
    return .{ .u = columns(&.{ first_u, second_u, m.cross(first_u, second_u) }), .v = columns(&.{ v0, v1, m.cross(v0, v1) }), .values = values };
}
