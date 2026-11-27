"""Compare native sparse VAE grids with the LocalMesh PyTorch models."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F


MODELS = {
    "shape_encode": "shape_enc_next_dc_f16c32_fp8",
    "shape_decode": "shape_dec_next_dc_f16c32_fp8",
    "texture_encode": "tex_enc_next_dc_f16c32_fp8",
    "texture_decode": "tex_dec_next_dc_f16c32_fp8",
}


def arrays(tensor):
    return tensor.coords.detach().cpu().numpy(), tensor.feats.detach().float().cpu().numpy()


def compare(coords, features, expected, threshold):
    reference_coords, reference_values = arrays(expected)
    a, b = np.lexsort(coords.T[::-1]), np.lexsort(reference_coords.T[::-1])
    if not np.array_equal(coords[a], reference_coords[b]):
        raise RuntimeError(f"Sparse supports differ ({len(a)} vs {len(b)} rows)")
    delta = features[a].astype(np.float64) - reference_values[b].astype(np.float64)
    error = float(np.sqrt(np.mean(delta**2)))
    if not np.isfinite(error) or error > threshold:
        raise RuntimeError(f"Sparse VAE RMSE {error:.8g} exceeds {threshold}")
    return {"rows": len(a), "channels": features.shape[1], "rmse": error, "max_abs": float(np.max(np.abs(delta)))}


def dense_convolution(module, sparse):
    """PyTorch functional oracle for submanifold convolution on a small grid."""
    coords = sparse.coords.long()
    origin = coords[:, 1:].amin(dim=0)
    spatial = coords[:, 1:] - origin
    dimensions = tuple((spatial.amax(dim=0) + 1).tolist())
    if int(coords[:, 0].max()) != 0 or max(dimensions) > 64:
        raise ValueError("The dense VAE oracle expects one small sparse grid")
    grid = sparse.feats.new_zeros((1, *dimensions, sparse.feats.shape[1]))
    z, y, x = spatial.unbind(dim=-1)
    grid[0, z, y, x] = sparse.feats
    kernel = module.weight.permute(0, 4, 1, 2, 3).contiguous()
    output = F.conv3d(grid.permute(0, 4, 1, 2, 3), kernel, module.bias, padding=1)
    return sparse.replace(output.permute(0, 2, 3, 4, 1)[0, z, y, x].contiguous())


def validate(args):
    sys.path.insert(0, str(args.fork.resolve(strict=True)))
    from localmesh_engine import config
    from localmesh_engine.zml_backend import CONFIG_ENV, configured
    from localmesh_engine.zml_sparse_vae import NativeVae

    os.environ[CONFIG_ENV] = str(args.config.resolve(strict=True))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sp = config.trellis("modules.sparse")
    models = config.trellis("models")
    fdg = config.trellis("models.sc_vaes.fdg_vae")
    with patch.object(sp.SparseConv3d, "forward", dense_convolution):
        return validate_models(args, configured, NativeVae, sp, models, fdg)


def validate_models(args, configured, NativeVae, sp, models, fdg):

    def native_decoder(stage):
        stage_key = f"pixal_{stage}" if args.pixal_weights_root is not None else stage
        return NativeVae(configured(stage_key), stage, stage=stage_key)

    def reference(stage):
        base = args.weights_root / MODELS[stage]
        if args.pixal_weights_root is not None and stage.endswith("_decode"):
            base = args.pixal_weights_root / MODELS[stage].replace("_fp8", "_fp16")
        info = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        parameters = dict(info["args"])
        for key in ("use_fp8", "use_fp16"):
            if key in parameters:
                parameters[key] = False
        with torch.device("meta"):
            model = getattr(models, info["name"])(**parameters)
        state = load_file(base.with_suffix(".safetensors"), device="cpu")
        model.load_state_dict({key: value.float() for key, value in state.items()}, strict=True, assign=True)
        return model.eval().cuda()

    coords = np.stack(np.meshgrid(*(np.arange(5, 13) for _ in range(3)), indexing="ij"), axis=-1).reshape(-1, 3)
    shell = np.abs(np.linalg.norm(coords - 8.5, axis=1) - 3) < 0.65
    coords = np.column_stack((np.zeros(shell.sum(), dtype=np.int32), coords[shell])).astype(np.int32)
    rng = np.random.default_rng(417)
    points = rng.uniform(0.4, 0.6, (len(coords), 3)).astype(np.float32)
    hit = rng.integers(0, 2, (len(coords), 3)).astype(np.float32)

    def sparse(coords, values):
        return sp.SparseTensor(feats=torch.from_numpy(values).cuda(), coords=torch.from_numpy(coords).cuda())

    summary = {}
    model = reference("shape_encode")
    vertices, hits = sparse(coords, points), sparse(coords, hit)
    z_ref, mean, lv = model(vertices, hits, return_raw=True)
    expected = z_ref.replace(torch.cat((mean, lv), dim=-1))
    native = NativeVae(configured("shape_encode"), "shape_encode")
    got_coord, got_values, _ = native.run(coords, np.concatenate((points - 0.5, hit - 0.5), axis=1))
    summary["shape_encode"] = compare(got_coord, got_values, expected, args.rmse)
    lat_coord, lat_values = arrays(z_ref)
    del vertices, hits, z_ref, mean, lv, expected, model
    gc.collect()
    torch.cuda.empty_cache()
    print("shape_encode", summary["shape_encode"], flush=True)

    model = reference("texture_encode")
    pbr = rng.uniform(-1, 1, (len(coords), 6)).astype(np.float32)
    z, mean, lv = model(sparse(coords, pbr), return_raw=True)
    expected = z.replace(torch.cat((mean, lv), dim=-1))
    native = NativeVae(configured("texture_encode"), "texture_encode")
    got_coord, got_values, _ = native.run(coords, pbr)
    summary["texture_encode"] = compare(got_coord, got_values, expected, args.rmse)
    del model, z, mean, lv, expected
    gc.collect()
    torch.cuda.empty_cache()
    print("texture_encode", summary["texture_encode"], flush=True)

    model = reference("shape_decode")
    latent = sparse(lat_coord.copy(), lat_values.copy())
    expected, guides = fdg.SparseUnetVaeDecoder.forward(model, latent, return_subs=True)
    native = native_decoder("shape_decode")
    c, features, sub = native.run(lat_coord, lat_values)
    summary["shape_decode"] = compare(c, features, expected, args.rmse)
    summary["shape_decode"]["subdivisions"] = [
        compare(c, logits, decoder, args.rmse) for (c, logits), decoder in zip(sub, guides, strict=True)
    ]
    del expected, model
    gc.collect()
    torch.cuda.empty_cache()
    print("shape_decode", summary["shape_decode"], flush=True)

    model = reference("texture_decode")
    texture = rng.normal(0, 0.3, lat_values.shape).astype(np.float32)
    material = sparse(lat_coord, texture)
    expected = model(material, guide_subs=guides)
    native = native_decoder("texture_decode")
    c, features, _ = native.run(lat_coord, texture, [arrays(tensor) for tensor in guides])
    summary["texture_decode"] = compare(c, features, expected, args.rmse)
    print("texture_decode", summary["texture_decode"], flush=True)
    return {
        "torch_version": torch.__version__, "reference_device": "cuda:0", "threshold": args.rmse,
        "reference_convolution": "torch.nn.functional.conv3d on a zero-filled grid",
        "native_pjrt": native.runtime.pjrt, "native_device": native.runtime.device, "cases": summary,
        "decoder_storage": "fp16" if args.pixal_weights_root is not None else "fp8",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "fork", "weights-root", "report"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--pixal-weights-root", type=Path)
    p.add_argument("--rmse", default=0.001, type=float)
    args = p.parse_args()
    with torch.inference_mode():
        summary = validate(args)
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(summary, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
