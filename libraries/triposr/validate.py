"""Compare the Windows TripoSR port with FP32 PyTorch on the same inputs."""

import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import torch
from torch.nn import functional as F

from mesh import (
    FORK_TOOLS,
    IMAGE_SIZE,
    PLANE_SHAPE,
    RADIUS,
    encode,
    extract,
    image_tensor,
    load_checkpoint,
    query,
    save_weights,
    schema_for,
)
from build import run_environment, scratch_directory


def torch_encoder(w, image, profile):
    initial_grid, patch, dino_heads, heads, groups = {
        "fixture": (4, 4, 4, 4, 4),
        "official": (14, 16, 12, 16, 32),
    }[profile]

    def linear(x, key):
        return F.linear(x, w[key + ".weight"], w.get(key + ".bias"))

    def norm(x, key, epsilon):
        return F.layer_norm(x, (x.shape[-1],), w[key + ".weight"], w[key + ".bias"], epsilon)

    def attend(q, k, v, head_count):
        def split(x):
            return x.unflatten(-1, (head_count, -1)).transpose(1, 2)

        return F.scaled_dot_product_attention(split(q), split(k), split(v)).transpose(1, 2).flatten(2)

    dino = "image_tokenizer.model."
    key = dino + "embeddings.patch_embeddings.projection"
    patches = F.conv2d(image, w[key + ".weight"], w[key + ".bias"], stride=patch)
    channels, grid = patches.shape[1:3]
    positions = w[dino + "embeddings.position_embeddings"]
    positions_2d = positions[:, 1:].transpose(1, 2).reshape(1, channels, initial_grid, initial_grid)
    interpolated = F.interpolate(positions_2d, scale_factor=(grid + 0.1) / initial_grid,
                                 mode="bicubic", align_corners=False)
    token = w[dino + "embeddings.cls_token"] + positions[:, :1]
    x = torch.cat((token, (patches + interpolated).flatten(2).transpose(1, 2)), dim=1)
    layers = sum(name.endswith(".layernorm_before.weight") for name in w)
    for i in range(layers):
        prefix = f"{dino}encoder.layer.{i}."
        y = norm(x, prefix + "layernorm_before", 1e-12)
        q, k, v = (linear(y, prefix + "attention.attention." + projection) for projection in ("query", "key", "value"))
        x = x + linear(attend(q, k, v, dino_heads), prefix + "attention.output.dense")
        y = norm(x, prefix + "layernorm_after", 1e-12)
        x = x + linear(F.gelu(linear(y, prefix + "intermediate.dense")), prefix + "output.dense")
    context = norm(x, dino + "layernorm", 1e-12)

    embeddings = w["tokenizer.embeddings"]
    planes, width, size, _ = embeddings.shape
    residual = embeddings.transpose(0, 1).reshape(1, width, -1)
    x = F.group_norm(residual, groups, w["backbone.norm.weight"], w["backbone.norm.bias"], eps=1e-6)
    x = linear(x.transpose(1, 2), "backbone.proj_in")
    blocks = sum(name.startswith("backbone.") and name.endswith(".norm1.weight") for name in w)
    for i in range(blocks):
        prefix = f"backbone.transformer_blocks.{i}"
        for j, mode in enumerate(("self", "cross"), start=1):
            normalized = norm(x, f"{prefix}.norm{j}", 1e-5)
            source = normalized if mode == "self" else context
            attn = f"{prefix}.attn{j}"
            y = attend(linear(normalized, attn + ".to_q"), linear(source, attn + ".to_k"),
                       linear(source, attn + ".to_v"), heads)
            x = x + linear(y, attn + ".to_out.0")
        linear_value, gate = linear(norm(x, prefix + ".norm3", 1e-5), prefix + ".ff.net.0.proj").chunk(2, dim=-1)
        x = x + linear(linear_value * F.gelu(gate), prefix + ".ff.net.2")
    x = (linear(x, "backbone.proj_out").transpose(1, 2) + residual).reshape(width, planes, size, size)
    return F.conv_transpose2d(x.transpose(0, 1), w["post_processor.upsample.weight"],
                              w["post_processor.upsample.bias"], stride=2)


def torch_field(w, planes, points):
    coordinates = torch.stack([points[:, [0, 1]], points[:, [0, 2]], points[:, [1, 2]]]).unsqueeze(1) / RADIUS
    sampled = F.grid_sample(planes, coordinates, align_corners=False)
    values = sampled[:, :, 0].flatten(0, 1).T
    layers = sum(name.startswith("decoder.layers.") and name.endswith(".weight") for name in w)
    for i in range(layers):
        prefix = f"decoder.layers.{i * 2}"
        values = F.linear(values, w[prefix + ".weight"], w[prefix + ".bias"])
        if i + 1 < layers:
            values = F.silu(values)
    return torch.cat(((values[:, :1] - 1).exp(), values[:, 1:].sigmoid()), dim=1)


def synthetic_checkpoint(schema, checkpoint):
    rng = torch.Generator().manual_seed(76520)
    weights = {}
    for name, shape in (schema["encoder"] | schema["field"]).items():
        values = torch.randn(shape, generator=rng)
        if len(shape) == 1:
            values = values * 0.04
            if name.endswith(".weight"):
                values = 1 + values
        elif len(shape) == 2:
            values = values * (0.8 / np.sqrt(shape[1]))
        else:
            values *= 0.12
        weights[name] = values
    torch.save(weights, checkpoint)


def compare(name, actual, expected, *, atol, rtol):
    print(f"{name}: max absolute error {np.max(np.abs(actual - expected)):.8g}", flush=True)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@torch.inference_mode()
def validate(args, work):
    profile = "official" if args.checkpoint else "fixture"
    official = schema_for(args.probe, "official")
    manifest = "\n".join(
        key + ":" + ",".join(map(str, shape)) for key, shape in sorted((official["encoder"] | official["field"]).items())
    )
    if sha256(manifest.encode("ascii")).hexdigest() != "b8f2c5192603373d87bf6ee6459138b681f7a3c07a1d57ed4d95931e47d78d50":
        raise ValueError("The graph schema differs from the official checkpoint header")
    print("Official checkpoint schema: 527 encoder and 20 field tensors match", flush=True)
    schema = official if profile == "official" else schema_for(args.probe, "fixture")
    if profile == "fixture":
        checkpoint = work / "fixture.ckpt"
        synthetic_checkpoint(schema, checkpoint)
        image_path = work / "fixture.png"
        pixels = np.random.default_rng(703).integers(0, 256, (27, 39, 4), dtype=np.uint8)
        Image.fromarray(pixels).save(image_path)
    else:
        checkpoint, image_path = args.checkpoint, args.image
    weights = load_checkpoint(checkpoint, schema)
    directory = work / "weights"
    save_weights(weights, schema, directory, profile)
    image = image_tensor(image_path, IMAGE_SIZE[profile])
    expected = torch_encoder(weights, torch.from_numpy(image[None]), profile)
    plane_path = encode(args.probe, profile, args.backend, directory, image, work, repeat=2)
    actual = np.fromfile(plane_path, dtype="<f4").reshape(PLANE_SHAPE[profile])
    tolerance = 3e-5 if profile == "fixture" else 5e-4
    # Checkpoint activations span thousands of units. Near-zero components need
    # an absolute floor scaled to the feature tensor. Field and mesh checks stay in source units.
    plane_absolute = tolerance if profile == "fixture" else max(tolerance, 5e-6 * expected.abs().max().item())
    compare("triplanes", actual, expected.numpy(), atol=plane_absolute, rtol=tolerance)
    points = torch.rand((8357, 3), generator=torch.Generator().manual_seed(4308)) * 3 - 1.5
    points[:3] = torch.tensor([[0, 0, 0], [-RADIUS, RADIUS, 0], [RADIUS, RADIUS, RADIUS]])
    expected_field = torch_field(weights, expected, points).numpy()
    actual_field = query(args.probe, profile, args.backend, directory, plane_path, points.numpy(), work)
    compare("density and RGB", actual_field, expected_field, atol=tolerance, rtol=tolerance)

    mesh_dir = work / "mesh"
    mesh_dir.mkdir()
    resolution = args.resolution
    grid = torch.linspace(-RADIUS, RADIUS, resolution)
    grid_points = torch.cartesian_prod(grid, grid, grid)
    densities = torch_field(weights, expected, grid_points)[:, 0].numpy()
    level = float((densities.min() + densities.max()) / 2) if profile == "fixture" else args.threshold
    output = mesh_dir / "reconstruction.ply"
    vertices, faces = extract(args.probe, profile, args.backend, directory, plane_path, output,
                              resolution, level, mesh_dir)
    native_volume = np.fromfile(mesh_dir / "density.f32", dtype="<f4")
    compare("density grid", native_volume, densities, atol=tolerance, rtol=tolerance)
    with output.open("rb") as mesh:
        header = []
        for line in mesh:
            if line == b"end_header\n":
                break
            header.append(line.strip())
        payload = mesh.read()
    if not vertices or not faces or f"element vertex {vertices}".encode() not in header or len(payload) != 15 * vertices + 13 * faces:
        raise ValueError("Invalid binary PLY")
    vertex_type = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
    face_type = np.dtype([("size", "u1"), ("index", "<i4", 3)])
    vertex_data = np.frombuffer(payload, dtype=vertex_type, count=vertices)
    face_data = np.frombuffer(payload, dtype=face_type, offset=15 * vertices)
    if np.any(face_data["size"] != 3) or face_data["index"].min() < 0 or face_data["index"].max() >= vertices:
        raise ValueError("Invalid PLY face indices")
    xyz = vertex_data["xyz"].copy()
    coordinates = np.clip((xyz.T + RADIUS) * ((resolution - 1) / (2 * RADIUS)), 0, resolution - 1)
    at_surface = map_coordinates(native_volume.reshape((resolution,) * 3), coordinates, order=1)
    compare("vertex isosurface", at_surface, np.full_like(at_surface, level), atol=tolerance, rtol=tolerance)
    expected_rgb = torch_field(weights, expected, torch.from_numpy(xyz))[:, 1:].numpy()
    compare("PLY vertex colors", vertex_data["rgb"].astype(np.float32) / 255, expected_rgb,
            atol=0.5 / 255 + tolerance, rtol=tolerance)
    print(f"colored PLY: {vertices} vertices, {faces} triangles", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm", "vulkan"), default="cpu")
    parser.add_argument("--resolution", type=int, choices=range(2, 65), default=24, metavar="[2-64]")
    parser.add_argument("--threshold", type=float, default=25)
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.image):
        parser.error("--checkpoint and --image must be supplied together")
    args.probe = args.probe.resolve(strict=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if build_work := os.environ.get("ZML_TEST_DIR"):
        validate(args, Path(build_work))
    else:
        with scratch_directory() as work:
            os.environ.update(run_environment(work, FORK_TOOLS))
            validate(args, work)


if __name__ == "__main__":
    main()
