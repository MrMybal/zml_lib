"""Compare the first block's Q and K stages using exact PyTorch CUDA inputs."""

import json

import numpy as np


PROJECTIONS = ("image_query", "image_key", "text_query", "text_key")


def capture_projections(model, values):
    attention = model.transformer_blocks[0].attn

    def record(name, tensor):
        values[name] = tensor.detach().cpu().numpy().copy()

    def capture_input(name):
        return lambda _module, inputs: record(name, inputs[0])

    def capture_output(name):
        return lambda _module, _inputs, output: record(name, output)

    handles = [model.pe_embedder.register_forward_hook(capture_output("rope"))]
    linear = (attention.to_q, attention.to_k, attention.add_q_proj, attention.add_k_proj)
    norm = (attention.norm_q, attention.norm_k, attention.norm_added_q, attention.norm_added_k)
    handles += [linear[0].register_forward_pre_hook(capture_input("image")),
                linear[2].register_forward_pre_hook(capture_input("text"))]
    for name, projection, normalization in zip(PROJECTIONS, linear, norm, strict=True):
        handles += [projection.register_forward_hook(capture_output(name + "_raw")),
                    normalization.register_forward_hook(capture_output(name + "_normalized"))]
    return handles


def compare_projections(args, run, errors, work, checkpoint, schema, values, geometry, packed_input):
    import torch
    from comfy.ldm.flux.math import apply_rope1
    from safetensors import safe_open
    from safetensors.torch import save_file
    from prepare import schema_for

    _, images, text_tokens, _ = geometry
    image_coords = images * 64
    text_coords = image_coords + images * 3 + text_tokens * 3584
    denoiser_input = np.fromfile(packed_input, "<f4")
    coords = np.concatenate((denoiser_input[text_coords:text_coords + text_tokens * 3],
                             denoiser_input[image_coords:image_coords + images * 3]))
    source, weights = work / "qk.input.f32", work / "qk.safetensors"

    raw = [values[name + "_raw"] for name in PROJECTIONS]
    normalized = [values[name + "_normalized"] for name in PROJECTIONS]
    with source.open("xb") as stream:
        for value in (values["image"], values["text"]):
            np.asarray(value, dtype="<f4", order="C").tofile(stream)
        for linear, norm in zip(raw, normalized, strict=True):
            heads = linear.reshape(1, -1, 24, 128).transpose(0, 2, 1, 3)
            np.asarray(heads, dtype="<f4", order="C").tofile(stream)
            np.asarray(norm, dtype="<f4", order="C").tofile(stream)
        coords.tofile(stream)

    names = schema_for(args.probe, "project_qk")
    scales = {name: i for i, name in enumerate(schema)}
    with safe_open(checkpoint, framework="pt", device="cpu") as saved:
        selected = {name: saved.get_tensor(name) for name in names}
        all_scales = saved.get_tensor("weight_scales")
        selected["weight_scales"] = torch.stack([all_scales[scales[name]] for name in names])
        save_file(selected, weights)
    del selected

    rope = torch.from_numpy(values["rope"]).to("cuda:0")
    rotated = [apply_rope1(torch.from_numpy(norm).to("cuda:0"),
                           rope[:, :, text_tokens:] if i < 2 else rope[:, :, :text_tokens]).cpu().numpy()
               for i, norm in enumerate(normalized)]
    del rope
    torch.cuda.empty_cache()

    for mode in ("project_qk", "project_qk_split"):
        output = work / f"{mode}.f32"
        placement = run(args, args.probe, mode, weights, (images, text_tokens), source, output, 1)
        data = np.fromfile(output, "<f4")
        offset = 0
        for i, name in enumerate(PROJECTIONS):
            stages = (("linear", raw[i]), ("rms", normalized[i]), ("rope", rotated[i]))
            for stage, expected in stages:
                actual = data[offset:offset + expected.size].reshape(expected.shape)
                offset += expected.size
                if mode == "project_qk_split":
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32),
                                                  err_msg=f"{name} {stage}")
                print(json.dumps({"stage": "First QK projection", "operation": stage, "projection": name,
                                  "probe": mode, **errors(actual, expected), "placement": placement}), flush=True)
        if offset != data.size:
            raise AssertionError("Unexpected QK diagnostic output size")
