"""Extract a colored FlexiCubes surface from TRELLIS v1's sparse mesh field."""

import numpy as np

from flexicubes.tables import CHECK_TABLE, DMC_TABLE


CASES = np.asarray(DMC_TABLE, dtype=np.int8)
AMBIGUITY = np.asarray(CHECK_TABLE, dtype=np.int16)
CORNERS = (np.arange(8, dtype=np.int32)[:, None] >> [0, 1, 2]) & 1
EDGES = np.array([(0, 1), (1, 5), (4, 5), (0, 4), (2, 3), (3, 7), (6, 7), (2, 6),
                  (2, 0), (3, 1), (7, 5), (6, 4)])


def linear_keys(coordinates, side):
    return (coordinates[..., 0] * side + coordinates[..., 1]) * side + coordinates[..., 2]


def from_keys(keys, side):
    return np.column_stack((keys // (side * side), keys // side % side, keys % side))


def lookup(keys, query):
    """Index a sorted sparse grid, using len(keys) as its missing value."""
    indices = np.searchsorted(keys, query)
    found = keys[np.minimum(indices, len(keys) - 1)] == query
    return np.where(found, indices, len(keys))


def surface_cells(cells, values, res):
    corner_keys = linear_keys(cells[:, None, :] + CORNERS, res + 1)
    keys, inverse = np.unique(corner_keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    corners = np.column_stack((values[:, :8].reshape(-1) - np.float32(1 / res),
                               values[:, 8:32].reshape(-1, 3), values[:, 53:].reshape(-1, 6)))
    sums = np.zeros((len(keys), 10), dtype=np.float32)
    np.add.at(sums, inverse, corners)
    sums /= np.bincount(inverse, minlength=len(keys)).astype(np.float32)[:, None]
    negative = from_keys(keys[sums[:, 0] < 0], res + 1)
    if not len(negative):
        return None
    candidates = (negative[:, None, :] - CORNERS).reshape(-1, 3)
    candidates = candidates[np.all((candidates >= 0) & (candidates < res), axis=1)]
    surface_keys = np.unique(linear_keys(candidates, res))
    surface = from_keys(surface_keys, res)
    vertex_keys = linear_keys(surface[:, None, :] + CORNERS, res + 1)
    indices = lookup(keys, vertex_keys)
    outside = np.zeros((1, 10), dtype=np.float32)
    outside[0, 0] = 1
    sampled = np.concatenate((sums, outside))[indices]
    case = (sampled[:, :, 0] < 0) @ (1 << np.arange(8))
    keep = (case != 0) & (case != 255)
    if not keep.any():
        return None
    surface, surface_keys, vertex_keys, sampled, case = (data[keep] for data in (surface, surface_keys, vertex_keys, sampled, case))

    problem = AMBIGUITY[case]
    mates = surface + problem[:, 1:4]
    valid = (problem[:, 0] != 0) & np.all((mates >= 0) & (mates < res), axis=1)
    tested = np.flatnonzero(valid)
    mate_flags = np.r_[problem[:, 0], 0][lookup(surface_keys, linear_keys(mates[tested], res))]
    flipped = tested[mate_flags != 0]
    case[flipped] = problem[flipped, 4]

    # Zero coefficients on neighboring cells absent from the learned field.
    order = np.argsort(linear_keys(cells, res))
    cell_rows = lookup(linear_keys(cells[order], res), surface_keys)
    coefficients = np.concatenate((values[order, 32:53], np.zeros((1, 21), dtype=np.float32)))[cell_rows]
    return surface, vertex_keys, sampled, case, coefficients


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


def dual_vertices(cells, samples, cases, coefficients, res):
    beta = 1 + np.float32(0.99) * np.tanh(coefficients[:, :12])
    alpha = 1 + np.float32(0.99) * np.tanh(coefficients[:, 12:20])
    gamma_cell = np.float32(0.99) * sigmoid(coefficients[:, 20]) + np.float32(0.005)
    positions = (cells[:, None, :] + CORNERS).astype(np.float32) / res - 0.5
    positions += np.tanh(samples[:, :, 1:4]) / (2 * res)
    data = np.concatenate((positions, sigmoid(samples[:, :, 4:])), axis=2)
    groups = CASES[cases]
    counts = (groups[:, :, 0] >= 0).sum(1)
    sorted_cells = np.argsort(counts, kind="stable")
    owner = np.repeat(sorted_cells, counts[sorted_cells])
    edge_groups = groups[sorted_cells].reshape(-1, groups.shape[-1])
    edge_groups = edge_groups[edge_groups[:, 0] >= 0]
    dual, slot = np.nonzero(edge_groups >= 0)
    cell = owner[dual]
    edge = edge_groups[dual, slot]
    endpoints = EDGES[edge]
    sdf = samples[cell[:, None], endpoints, 0] * alpha[cell[:, None], endpoints]
    weighted = beta[cell, edge]
    values = data[cell[:, None], endpoints]
    # Interpolations share the alpha-adjusted zero crossing for position and color.
    ratio = sdf[:, 0] / (sdf[:, 0] - sdf[:, 1])
    interpolated = values[:, 0] + ratio[:, None] * (values[:, 1] - values[:, 0])
    result = np.zeros((len(owner), 9), dtype=np.float32)
    denominator = np.zeros(len(owner), dtype=np.float32)
    np.add.at(result, dual, interpolated * weighted[:, None])
    np.add.at(denominator, dual, weighted)
    result /= denominator[:, None]
    edge_to_dual = np.full((len(cells), 12), -1, dtype=np.int64)
    edge_to_dual[cell, edge] = dual
    return result, gamma_cell[owner], edge_to_dual


def triangles(vertex_keys, sample_sdf, edge_to_dual, gamma, res):
    cells, edges = np.nonzero(edge_to_dual >= 0)
    endpoints = EDGES[edges]
    corner_keys = vertex_keys[cells[:, None], endpoints].astype(np.int64)
    keys = corner_keys[:, 0] * (res + 1) ** 3 + corner_keys[:, 1]
    order = np.argsort(keys, kind="stable")
    _, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    groups = order[starts[counts == 4, None] + np.arange(4)]
    quads = edge_to_dual[cells[groups], edges[groups]]
    first = groups[:, 0]
    positive = sample_sdf[cells[first], endpoints[first, 0]] > 0
    quads = np.concatenate((quads[positive][:, [0, 1, 3, 2]], quads[~positive][:, [2, 3, 1, 0]]))
    scores = gamma[quads]
    split = scores[:, 0] * scores[:, 2] > scores[:, 1] * scores[:, 3]
    return np.where(split[:, None], quads[:, [0, 1, 2, 0, 2, 3]], quads[:, [0, 1, 3, 3, 1, 2]]).reshape(-1, 3)


def extract_mesh(cells, values, resolution):
    surface = surface_cells(cells, values, resolution)
    if surface is None:
        return {"vertices": np.empty((0, 3), np.float32), "faces": np.empty((0, 3), np.int64),
                "vertex_attrs": np.empty((0, 6), np.float32)}
    cells, keys, samples, cases, coefficients = surface
    data, gamma, edge_to_dual = dual_vertices(cells, samples, cases, coefficients, resolution)
    faces = triangles(keys, samples[:, :, 0], edge_to_dual, gamma, resolution)
    return {"vertices": data[:, :3], "faces": faces, "vertex_attrs": data[:, 3:]}
