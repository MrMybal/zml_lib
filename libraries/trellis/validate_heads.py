"""Compare the TRELLIS representation decoders on a forced backend with CPU PyTorch."""

import argparse
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
import torch
from torch.nn import functional as F

from head_io import FORK_TOOLS, PROFILES, SCHEMAS, decode, schema_for, write_output
from mesh import extract_mesh
from build import run_environment, scratch_directory
from probe_placement import PLATFORMS
from structure_io import read_checkpoint


def reference(state, coords, features, kind, profile="fixture"):
    w = {name: torch.from_numpy(data) for name, data in state.items()}
    xyz = torch.from_numpy(coords[:, 1:]).long()
    width = w["input_layer.bias"].numel()
    heads, blocks, norm_groups = (12, 12, 32) if profile == "official" else (4, 2, 4)
    resolution = PROFILES[profile]
    angle = xyz[..., None] / (10000 ** (torch.arange(width // 6) / (width // 6)))
    pos = torch.cat((angle.sin(), angle.cos()), dim=-1).flatten(1)
    pos = F.pad(pos, (0, width - pos.shape[1]))

    def linear(key, x):
        return F.linear(x, w[key + ".weight"], w[key + ".bias"])

    x = linear("input_layer", torch.from_numpy(features)) + pos
    for i in range(blocks):
        shift = 4 * (i % 2)
        key = f"blocks.{i}."
        qkv = linear(key + "attn.to_qkv", F.layer_norm(x, (width,), eps=1e-6))
        attended = torch.empty_like(x)
        groups = torch.div(xyz + shift, 8, rounding_mode="floor")
        for cell in groups.unique(dim=0):
            select = (groups == cell).all(dim=1)
            rows = qkv[select].reshape(-1, 3, heads, width // heads)
            q, k, v = rows.permute(1, 2, 0, 3)
            attended[select] = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(-1, width)
        x = x + linear(key + "attn.to_out", attended)
        h = linear(key + "mlp.mlp.0", F.layer_norm(x, (width,), eps=1e-6))
        x = x + linear(key + "mlp.mlp.2", F.gelu(h, approximate="tanh"))

    if kind == "mesh":
        offsets = torch.cartesian_prod(*(torch.arange(2) for _ in range(3)))
        for i in range(2):
            key = f"upsample.{i}."

            def norm(name, value):
                return F.group_norm(value.T[None], norm_groups, w[key + name + ".weight"], w[key + name + ".bias"])[0].T

            h = F.silu(norm("act_layers.0", x)).repeat_interleave(8, dim=0)
            x = x.repeat_interleave(8, dim=0)
            xyz = (xyz[:, None] * 2 + offsets).reshape(-1, 3)
            resolution *= 2
            local = xyz - xyz.min(dim=0).values
            a, b, c = local.T
            extent = tuple((local.max(dim=0).values + 1).tolist())

            def conv(name, value):
                weight = w[key + name + ".weight"].permute(0, 4, 1, 2, 3)
                dense = torch.zeros((1, value.shape[1], *extent))
                dense[0, :, a, b, c] = value.T
                out = F.conv3d(dense, weight, w[key + name + ".bias"], padding=weight.shape[-1] // 2)
                return out[0, :, a, b, c].T

            h = conv("out_layers.0.conv", h)
            x = conv("out_layers.3.conv", F.silu(norm("out_layers.1", h))) + conv("skip_connection.conv", x)
        return xyz.numpy(), linear("out_layer", x).numpy()

    output = linear("out_layer", F.layer_norm(x, (width,)))
    centers = (xyz + 0.5) / resolution
    if kind == "gaussian":
        offset = output[:, :96].reshape(-1, 32, 3) + w["offset_perturbation"]
        points = centers[:, None] + torch.tanh(offset) * (0.75 / resolution)
        output = torch.cat((points.flatten(1), output[:, 96:288], output[:, 288:416] * 0.1, output[:, 416:]), dim=1)
    else:
        output = torch.cat((centers, output[:, :384] + 1, output[:, 384:]), dim=1)
    return xyz.numpy(), output.numpy()


def check_mesh_fixture():
    # One negative interior grid vertex makes a closed dual octant surface.
    cells = np.array([[2, 2, 2]], np.int32)
    values = np.zeros((1, 101), np.float32)
    values[0, :8] = 1 / 8 + 1
    values[0, 0] -= 2
    mesh = extract_mesh(cells, values, 8)
    vertices, faces = mesh["vertices"], mesh["faces"]
    assert vertices.shape == (8, 3) and faces.shape == (12, 3)
    expected_axes = (np.array([2 - 1 / 6, 2 + 1 / 6]) / 8 - 0.5).astype(np.float32)
    np.testing.assert_allclose(np.sort(vertices, axis=0), np.repeat(expected_axes, 4)[:, None].repeat(3, axis=1), atol=2e-7)
    np.testing.assert_allclose(mesh["vertex_attrs"], 0.5)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, count = np.unique(edges, axis=0, return_counts=True)
    np.testing.assert_array_equal(count, np.full(18, 2))
    points = vertices[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    assert np.all(np.sum(normals * (points.mean(axis=1) + 0.25), axis=1) > 0)
    assert not extract_mesh(cells, np.full_like(values, 1), 8)["faces"].size


def check_output(kind, path, cells, actual, expected, profile="fixture"):
    resolution = PROFILES[profile]
    write_output(kind, path, cells, actual, resolution)
    if kind == "gaussian":
        with path.open("rb") as stream:
            assert stream.readline() == b"ply\n"
            for line in stream:
                if line == b"end_header\n":
                    break
            data = np.frombuffer(stream.read(), dtype="<f4").reshape(-1, 17)
        xyz = expected[:, :96].reshape(-1, 3) - 0.5
        np.testing.assert_allclose(data[:, :3], xyz[:, [0, 2, 1]] * [1, -1, 1], atol=1e-5)
        np.testing.assert_allclose(data[:, 6:9], expected[:, 96:192].reshape(-1, 3), atol=1e-5)
        np.testing.assert_allclose(data[:, 9], expected[:, 416:].reshape(-1) + np.log(0.1 / 0.9), atol=1e-5)
        scale = F.softplus(torch.from_numpy(expected[:, 192:288].reshape(-1, 3)) + np.log(np.expm1(0.004)))
        np.testing.assert_allclose(data[:, 10:13], (scale.square() + 0.0009 ** 2).sqrt().log().numpy(), atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(data[:, -4:], axis=1), 1, atol=1e-6)
        assert np.isfinite(data).all()
        return f"{len(data)} splats"
    with np.load(path, allow_pickle=False) as saved:
        if kind == "radiance":
            np.testing.assert_allclose(saved["position"], expected[:, :3], atol=1e-6)
            np.testing.assert_allclose(saved["trivec"].reshape(-1, 384), expected[:, 3:387], atol=1e-5)
            np.testing.assert_allclose(saved["density"], expected[:, 387:403], atol=1e-5)
            np.testing.assert_allclose(saved["features_dc"].reshape(-1, 48), expected[:, 403:], atol=1e-5)
            assert (saved["depth"] == resolution.bit_length() - 1).all()
            return f"{len(cells)} Strivec cells"
        check_mesh_fixture()
        target = extract_mesh(cells, expected, resolution * 4)
        if profile == "fixture":
            assert len(saved["vertices"]) and len(saved["faces"])
        np.testing.assert_array_equal(saved["faces"], target["faces"])
        np.testing.assert_allclose(saved["vertices"], target["vertices"], atol=2e-6)
        np.testing.assert_allclose(saved["vertex_attrs"], target["vertex_attrs"], atol=2e-6)
        return f"{len(saved['vertices'])} vertices, {len(saved['faces'])} triangles"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--kind", choices=tuple(SCHEMAS))
    parser.add_argument("--checkpoint", type=Path, help="official SafeTensors for the selected head")
    parser.add_argument("--backend", choices=PLATFORMS, default="cpu")
    parser.add_argument("--expected-device", help="substring of the reported output device name")
    args = parser.parse_args()
    if args.checkpoint and not args.kind:
        parser.error("--checkpoint requires --kind")
    torch.set_num_threads(2)
    probe = args.probe.resolve(strict=True)
    rng = np.random.default_rng(81)
    coords = np.array([[0, 3, 3, 4], [0, 3, 3, 2], [0, 4, 3, 2], [0, 6, 5, 4], [0, 2, 3, 4],
                       [0, 2, 3, 3], [0, 4, 3, 3], [0, 2, 3, 2], [0, 3, 3, 3]], dtype=np.int32)
    features = rng.normal(0, 0.5, (len(coords), 8)).astype(np.float32)
    for kind in (args.kind,) if args.kind else SCHEMAS:
        official = schema_for(probe, "official", kind)
        profile = "official" if args.checkpoint else "fixture"
        with scratch_directory() as work:
            os.environ.update(run_environment(work, FORK_TOOLS))
            if args.checkpoint:
                schema = official
                source = args.checkpoint
            else:
                schema = schema_for(probe, profile, kind)
                fixture = {name: rng.normal(1 if "layers" in name and len(dims) == 1 and name.endswith("weight") else 0,
                                            0.045, dims).astype(np.float16) for name, dims in schema.items()}
                if kind == "mesh":
                    fixture["out_layer.bias"][:8] = -0.16
                source = work / f"{kind}-fp16.safetensors"
                save_file(fixture, source)
            state = read_checkpoint(source, schema)
            checkpoint = work / f"{kind}.safetensors"
            save_file(state, checkpoint)
            cells, actual = decode(probe, profile, kind, args.backend, checkpoint, coords, features, work,
                                   repeats=2, expected_device=args.expected_device)
            expected_cells, expected = reference(state, coords, features, kind, profile)
            np.testing.assert_array_equal(cells, expected_cells)
            tolerance = 1e-3 if args.checkpoint else 2e-5
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            path = work / ("out.ply" if kind == "gaussian" else "out.npz")
            info = check_output(kind, path, cells, actual, actual if args.checkpoint else expected, profile)
            print(f"TRELLIS {kind}, official schema={len(official)}, {info}, max_abs={np.max(np.abs(actual - expected)):.8g}", flush=True)


if __name__ == "__main__":
    main()
