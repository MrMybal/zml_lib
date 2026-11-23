const zml = @import("zml/core");
const l = @import("vo_linear.zig");
const m = @import("post_math.zig");
const Tensor = zml.Tensor;
const s = l.s;

pub const homography_trials = 128;
const essential_trials = 96;
pub const sample_size = 8;
pub const output_width = 15;
const homography_groups = 4;

fn rays(points: Tensor, axis: i64, start: i64) Tensor {
    const plane = points.slice(axis, .{ .start = start, .end = start + 2 });
    return Tensor.concatenate(&.{ plane, s(1).broad(plane.shape().set(axis, 1)) }, axis);
}

fn quadrangle(points: Tensor) Tensor {
    const base = l.transpose(points.slice(1, .{ .end = 3 }));
    const scale = m.apply(l.inverse3(base), points.slice(1, .single(3)));
    return base.mul(scale.broadcast(base.shape(), &.{ 0, 2 }));
}

fn homographySamples(samples: Tensor) Tensor {
    const corners = samples.slice(1, .{ .end = 4 });
    return m.product(quadrangle(rays(corners, 2, 2)), l.inverse3(quadrangle(rays(corners, 2, 0))));
}

fn equation(x: Tensor, y: Tensor) Tensor {
    return l.outer(y, x).reshape(.{ x.dim(0), 9 });
}

fn essentialSamples(samples: Tensor) Tensor {
    const n = samples.dim(0);
    const x = rays(samples, 2, 0);
    const y = rays(samples, 2, 2);
    // Project onto the orthogonal complement of each epipolar constraint.
    var projector = l.eye(n, 9);
    for (0..sample_size) |i| {
        const p = equation(x.slice(1, .single(@intCast(i))), y.slice(1, .single(@intCast(i))));
        const direction = l.unit(m.apply(projector, p));
        projector = projector.sub(l.outer(direction, direction));
    }
    var diag = l.entry(projector, 0, 0);
    var column = m.column(projector, 0);
    for (1..9) |i| {
        const value = l.entry(projector, @intCast(i), @intCast(i));
        const better = value.cmp(.GT, diag);
        column = better.broad(column.shape().withDtype(.bool)).select(m.column(projector, @intCast(i)), column);
        diag = value.maximum(diag);
    }
    return l.unit(column).reshape(.{ n, 3, 3 });
}

fn project(h: Tensor, rays_: Tensor) Tensor {
    return h.dotGeneral(rays_, &.{.{ 2, 1 }}, &.{}).transpose(.{ 0, 2, 1 });
}

fn square(x: Tensor) Tensor {
    return x.mul(x);
}

fn residuals(comptime planar: bool, h: Tensor, x: Tensor, y: Tensor) Tensor {
    const hx = project(h, x);
    if (planar) {
        const xy = hx.slice(2, .{ .end = 2 });
        const depth = l.safe(hx.slice(2, .{ .start = 2 }), 1e-14);
        const errors = xy.div(depth.broad(xy.shape())).sub(y.slice(1, .{ .end = 2 }).broadcast(xy.shape(), &.{ 1, 2 }));
        return square(errors).sum(2).squeeze(2);
    }
    const hy = project(l.transpose(h), y);
    const numerator = square(hx.mul(y.broadcast(hx.shape(), &.{ 1, 2 })).sum(2));
    const a = square(hx.slice(2, .{ .end = 2 })).sum(2);
    const b = square(hy.slice(2, .{ .end = 2 })).sum(2);
    return numerator.div(a.add(b).maximum(s(1e-25))).squeeze(2);
}

const Fit = struct { matrix: Tensor, mask: Tensor, support: Tensor, score: Tensor };

fn evaluate(comptime planar: bool, h: Tensor, x: Tensor, y: Tensor, observed: Tensor, cutoff: Tensor) Fit {
    const errors = residuals(planar, h, x, y);
    const mask = errors.cmp(.LT, cutoff).logical(.AND, observed.broadcast(errors.shape().withDtype(.bool), &.{1}));
    const cost = mask.select(errors.div(cutoff), s(0)).sum(1).scale(@as(f64, 0.5) / @as(f64, @floatFromInt(x.dim(0))));
    const support = mask.convert(.f64).sum(1);
    const valid = if (planar) l.determinant(h).abs().cmp(.GT, s(1e-12)) else support.cmp(.GT, s(0));
    const score = valid.select(support.sub(cost), s(-1));
    return .{ .matrix = h, .mask = mask, .support = valid.select(support, s(0)), .score = score };
}

fn winner(fit: Fit) Fit {
    const index = fit.score.argMax(0).indices.reshape(.{});
    return .{
        .matrix = fit.matrix.slice(0, .dyn(index, 1)),
        .mask = fit.mask.slice(0, .dyn(index, 1)),
        .support = fit.support.slice(0, .dyn(index, 1)),
        .score = fit.score.slice(0, .dyn(index, 1)),
    };
}

fn keepBetter(a: Fit, b: Fit) Fit {
    const ok = b.score.cmp(.GT, a.score);
    return .{
        .matrix = ok.reshape(.{ a.matrix.dim(0), 1, 1 }).broad(a.matrix.shape().withDtype(.bool)).select(b.matrix, a.matrix),
        .mask = ok.broad(a.mask.shape()).select(b.mask, a.mask),
        .support = ok.select(b.support, a.support),
        .score = ok.select(b.score, a.score),
    };
}

fn groupWinners(fit: Fit) Fit {
    const size = homography_trials / homography_groups;
    const shape: zml.Shape = .init(.{homography_groups}, .i32);
    const local = fit.score.reshape(.{ homography_groups, size }).argMax(1).indices.reshape(shape);
    const index = local.add(Tensor.iota(shape, 0).mul(.scalar(size, .i32)));
    const count = fit.mask.dim(1);
    const proposals = m.join(&.{ fit.matrix.reshape(.{ homography_trials, 9 }), fit.mask.convert(.f64), fit.support, fit.score });
    const rows = proposals.withTags(.{ .proposal, .data }).gather(.{ .proposal = index.withTags(.{.winner}) }, .{})
        .reshape(.{ homography_groups, 11 + count });
    return .{
        .matrix = rows.slice(1, .{ .end = 9 }).reshape(.{ homography_groups, 3, 3 }),
        .mask = rows.slice(1, .{ .start = 9, .end = 9 + count }).cmp(.GT, s(0)),
        .support = m.component(rows, 9 + count),
        .score = m.component(rows, 10 + count),
    };
}

fn homographyFit(x: Tensor, y: Tensor, fit: Fit, k: Tensor, ki: Tensor) Tensor {
    const n = x.dim(0);
    const batches = fit.matrix.dim(0);
    const u = m.component(y, 0);
    const v = m.component(y, 1);
    const blank = s(0).broad(x.shape());
    const top = m.join(&.{ x, blank, x.mul(u.broad(x.shape())).negate() });
    const bottom = m.join(&.{ blank, x, x.mul(v.broad(x.shape())).negate() });
    const design = Tensor.concatenate(&.{ top, bottom }, 0);
    const weights = Tensor.concatenate(&.{ fit.mask, fit.mask }, 1).convert(.f64).reshape(.{ batches, n * 2, 1 });
    const design_batch = design.broadcast(.init(.{ batches, n * 2, 9 }, .f64), &.{ 1, 2 });
    const weighted = design_batch.mul(weights.broad(design_batch.shape()));
    const gram = design.dotGeneral(weighted, &.{.{ 0, 1 }}, &.{}).transpose(.{ 1, 0, 2 }).add(l.eye(batches, 9).scale(@as(f64, 1e-9)));
    const factor = l.factor(gram);
    // Preserve the pixel-space homogeneous DLT metric while solving in the
    // better conditioned normalized coordinates. K*H*K^-1 is the change of basis.
    var h = fit.matrix;
    for (0..3) |_| {
        const pixels = m.product(m.product(k, h), ki);
        const adjoint = m.product(m.product(l.transpose(k), pixels), l.transpose(ki));
        h = l.unit(l.solve(factor, adjoint.reshape(.{ batches, 9 }))).reshape(.{ batches, 3, 3 });
    }
    return h;
}

const RefineHomography = struct {
    pub const State = struct {
        iteration: Tensor,
        fit: Fit,
        x: Tensor,
        y: Tensor,
        observed: Tensor,
        cutoff: Tensor,
        k: Tensor,
        ki: Tensor,
    };

    pub fn cond(_: @This(), state: State) Tensor {
        return state.iteration.cmp(.LT, .scalar(8, .i32));
    }

    pub fn body(_: @This(), state: State) State {
        const model = homographyFit(state.x, state.y, state.fit, state.k, state.ki);
        var next = state;
        next.fit = keepBetter(state.fit, evaluate(true, model, state.x, state.y, state.observed, state.cutoff));
        next.iteration = state.iteration.addConstant(@as(i32, 1));
        return next;
    }
};

fn essentialFit(x: Tensor, y: Tensor, fit: Fit) Tensor {
    const design = equation(x, y);
    const weighted = design.mul(fit.mask.reshape(.{ x.dim(0), 1 }).convert(.f64).broad(design.shape()));
    const gram = design.dotGeneral(weighted, &.{.{ 0, 0 }}, &.{}).reshape(.{ 1, 9, 9 });
    const factor = l.factor(gram.add(l.eye(1, 9).scale(@as(f64, 1e-9))));
    var e = fit.matrix.reshape(.{ 1, 9 });
    for (0..4) |_| e = l.unit(l.solve(factor, e));
    return e.reshape(.{ 1, 3, 3 });
}

const Pose = struct { rotation: Tensor, translation: Tensor, score: Tensor };

fn solveSign(rot: Tensor, translation: Tensor, x: Tensor, y: Tensor, support: Tensor) Pose {
    const n = x.dim(0);
    const a = l.unit(project(rot, x).reshape(.{ n, 3 }));
    const b = l.unit(y);
    const t = translation.broad(x.shape());
    const ab = m.dot(a, b);
    const at = m.dot(a, t);
    const bt = m.dot(b, t);
    const denom = s(1).sub(square(ab)).maximum(s(1e-15));
    const depth0 = ab.mul(bt).sub(at).div(denom);
    const depth1 = bt.sub(ab.mul(at)).div(denom);
    const plus = depth0.cmp(.GT, s(1e-10)).logical(.AND, depth1.cmp(.GT, s(1e-10)));
    const minus = depth0.cmp(.LT, s(-1e-10)).logical(.AND, depth1.cmp(.LT, s(-1e-10)));
    const keep = support.reshape(.{ n, 1 });
    const midpoint = l.scaleRows(a, depth0).add(t).add(l.scaleRows(b, depth1)).scale(@as(f64, 0.5));
    const consistency = m.dot(a, l.unit(midpoint.sub(t))).add(m.dot(b, l.unit(midpoint)));
    const signs = [_]Tensor{ plus.logical(.AND, keep), minus.logical(.AND, keep) };
    var scores: [2]Tensor = undefined;
    for (signs, 0..) |mask, i| {
        const sign: f64 = if (i == 0) 1 else -1;
        const angle = s(2).sub(consistency.scale(sign)).clamp(s(0), s(4));
        const cost = mask.select(angle, s(0)).sum(0).scale(@as(f64, 0.12) / @as(f64, @floatFromInt(n)));
        scores[i] = mask.convert(.f64).sum(0).sub(cost);
    }
    const neg = scores[1].cmp(.GT, scores[0]);
    return .{ .rotation = rot, .translation = neg.broad(translation.shape().withDtype(.bool)).select(translation.negate(), translation), .score = scores[0].maximum(scores[1]) };
}

fn choosePose(a: Pose, b: Pose) Pose {
    const better = b.score.cmp(.GT, a.score);
    return .{
        .rotation = better.reshape(.{ 1, 1, 1 }).broad(a.rotation.shape().withDtype(.bool)).select(b.rotation, a.rotation),
        .translation = better.broad(a.translation.shape().withDtype(.bool)).select(b.translation, a.translation),
        .score = better.select(b.score, a.score),
    };
}

fn yTurn(c: Tensor, sine: Tensor) Tensor {
    const zero = s(0).broad(c.shape());
    const one = s(1).broad(c.shape());
    return l.rows(&.{ m.join(&.{ c, zero, sine }), m.join(&.{ zero, one, zero }), m.join(&.{ sine.negate(), zero, c }) });
}

fn planePose(fit: Fit, x: Tensor, y: Tensor) Pose {
    const h = l.determinant(fit.matrix).cmp(.LT, s(0)).reshape(.{ 1, 1, 1 }).broad(fit.matrix.shape().withDtype(.bool))
        .select(fit.matrix.negate(), fit.matrix);
    const svd = l.svd3(h);
    const mid = svd.values[1].maximum(s(1e-15));
    const scale = s(1).div(mid).reshape(.{ 1, 1, 1 });
    const calibrated = h.mul(scale.broad(h.shape()));
    const a = svd.values[0].div(mid);
    const b = svd.values[2].div(mid);
    const divisor = a.add(b).maximum(s(1e-15));
    const cosine = a.mul(b).add(s(1)).div(divisor);
    const sine = square(a).sub(s(1)).maximum(s(0)).mul(s(1).sub(square(b)).maximum(s(0))).sqrt().div(divisor);
    var hypotheses: [2]Pose = undefined;
    for (&hypotheses, 0..) |*out, i| {
        const angle = if (i == 0) sine else sine.negate();
        const rotation = m.product(m.product(svd.u, yTurn(cosine, angle)), l.transpose(svd.v));
        const rank1 = calibrated.sub(rotation);
        var translation = m.column(rank1, 0);
        for (1..3) |j| {
            const option = m.column(rank1, @intCast(j));
            translation = m.norm(option).cmp(.GT, m.norm(translation)).broad(translation.shape().withDtype(.bool)).select(option, translation);
        }
        const magnitude = square(rank1).reshape(.{ 1, 9 }).sum(1).sqrt();
        out.* = solveSign(rotation, l.scaleRows(l.unit(translation), magnitude), x, y, fit.mask);
    }
    const selected = choosePose(hypotheses[0], hypotheses[1]);
    const panoramic = a.sub(s(1)).abs().maximum(b.sub(s(1)).abs()).cmp(.LT, s(5e-4));
    const pure = Pose{ .rotation = m.product(svd.u, l.transpose(svd.v)), .translation = s(0).broad(.init(.{ 1, 3 }, .f64)), .score = fit.support };
    return .{
        .rotation = panoramic.reshape(.{ 1, 1, 1 }).broad(selected.rotation.shape().withDtype(.bool)).select(pure.rotation, selected.rotation),
        .translation = panoramic.broad(selected.translation.shape().withDtype(.bool)).select(pure.translation, selected.translation),
        .score = panoramic.select(pure.score, selected.score),
    };
}

fn essentialPose(fit: Fit, x: Tensor, y: Tensor) Pose {
    const svd = l.svd3(fit.matrix);
    const w = l.eye(1, 3).slice(1, .single(0));
    const zero = m.component(w, 1);
    const one = m.component(w, 0);
    const turn = l.rows(&.{ m.join(&.{ zero, one.negate(), zero }), m.join(&.{ one, zero, zero }), m.join(&.{ zero, zero, one }) });
    const translation = m.column(svd.u, 2);
    const r0 = m.product(m.product(svd.u, turn), l.transpose(svd.v));
    const r1 = m.product(m.product(svd.u, l.transpose(turn)), l.transpose(svd.v));
    return choosePose(solveSign(r0, translation, x, y, fit.mask), solveSign(r1, translation, x, y, fit.mask));
}

pub fn forward(matches: Tensor, selected: Tensor, camera: Tensor) Tensor {
    const points = matches.convert(.f64);
    const samples = selected.convert(.f64);
    const x = rays(points, 1, 0);
    const y = rays(points, 1, 2);
    const observed = points.slice(1, .single(4)).cmp(.GT, s(0));
    const calib = camera.convert(.f64);
    const cutoff = square(m.component(calib, 0)).reshape(.{});
    const focal = m.component(calib, 1);
    const blank = s(0).broad(focal.shape());
    const one = s(1).broad(focal.shape());
    const k = l.rows(&.{ m.join(&.{ focal, blank, m.component(calib, 2) }), m.join(&.{ blank, focal, m.component(calib, 3) }), m.join(&.{ blank, blank, one }) });
    const ki = l.inverse3(k);
    // Refine several independent hypotheses so a minimal four-point sample
    // cannot win solely because its unrefined consensus starts out larger.
    const hypotheses = groupWinners(evaluate(true, homographySamples(samples), x, y, observed, cutoff));
    const grouped_shape: zml.Shape = .init(.{ homography_groups, 3, 3 }, .f64);
    const h = winner(zml.ops.while_(RefineHomography, .{}, .{
        .iteration = .scalar(0, .i32),
        .fit = hypotheses,
        .x = x,
        .y = y,
        .observed = observed,
        .cutoff = cutoff,
        .k = k.broad(grouped_shape),
        .ki = ki.broad(grouped_shape),
    }).fit);
    const epipolar_samples = samples.slice(0, .{ .end = essential_trials });
    var e = winner(evaluate(false, essentialSamples(epipolar_samples), x, y, observed, cutoff));
    for (0..2) |_| {
        e = keepBetter(e, evaluate(false, essentialFit(x, y, e), x, y, observed, cutoff));
    }
    // The epipolar consensus can cover points that lie outside the selected
    // plane. Test cheirality with the points that support this homography.
    const plane = planePose(h, x, y);
    const essential = essentialPose(e, x, y);
    const plane_ok = h.support.cmp(.GE, s(10)).logical(.AND, plane.score.cmp(.GT, s(0)));
    const essential_ok = e.support.cmp(.GE, s(10)).logical(.AND, essential.score.cmp(.GT, s(0)));
    const large_plane = h.support.cmp(.GE, e.support.scale(@as(f64, 0.9)));
    const planar = plane_ok.logical(.AND, large_plane.logical(.OR, essential_ok.cmp(.EQ, .scalar(false, .bool))));
    const r = planar.reshape(.{ 1, 1, 1 }).broad(plane.rotation.shape().withDtype(.bool)).select(plane.rotation, essential.rotation);
    const t = planar.broad(plane.translation.shape().withDtype(.bool)).select(plane.translation, essential.translation);
    const support = planar.select(h.support, e.support);
    const config = planar.select(s(1), essential_ok.select(s(2), s(0)));
    return m.join(&.{ r.reshape(.{ 1, 9 }), t, support, config, planar.select(plane.score, essential.score) }).convert(.f32);
}
