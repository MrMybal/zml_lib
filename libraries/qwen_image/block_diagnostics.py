"""Compare the first denoising step on shared PyTorch hidden states."""

from contextlib import contextmanager
import json

import numpy as np

from inputs import unpack_latent, write_denoise_input


class BlockReference:
    def __init__(self, *, projections=False):
        self.condition = None
        self.streams = []
        self.projections = {} if projections else None

    @staticmethod
    def pack(*tensors):
        return np.concatenate([tensor.detach().cpu().numpy().ravel() for tensor in tensors], dtype="<f4")

    @contextmanager
    def capture(self, model):
        import torch

        def capture_input(_module, _args, kwargs):
            if not self.streams:
                self.condition = self.pack(torch.nn.functional.silu(kwargs["temb"]))
            self.streams.append(self.pack(kwargs["hidden_states"], kwargs["encoder_hidden_states"]))

        def capture_end(_module, _args, output):
            text, image = output
            self.streams.append(self.pack(image, text))

        handles = [block.register_forward_pre_hook(capture_input, with_kwargs=True)
                   for block in model.transformer_blocks]
        handles.append(model.transformer_blocks[-1].register_forward_hook(capture_end))
        if self.projections is not None:
            from qk_diagnostics import capture_projections

            handles.extend(capture_projections(model, self.projections))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


def errors(actual, expected):
    a = actual.astype(np.float64)
    b = expected.astype(np.float64)
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff**2)))
    reference_rms = float(np.sqrt(np.mean(b**2)))
    return {"max_abs": float(np.max(np.abs(diff))), "rmse": rmse,
            "relative_rmse": rmse / reference_rms if reference_rms else None}


def compare_blocks(args, run, work, profile, weights, target, references, text, mask, time,
                   trace, expected_velocity, standard_velocity, *, tolerance=None):
    source, output = work / "blocks.input.f32", work / "blocks.output.f32"
    geometry = write_denoise_input(source, target, references, text, mask, time)
    target_tokens, image_tokens, text_tokens, _ = geometry
    width = trace.condition.size // 2
    velocity_count, image_count = target_tokens * 64, image_tokens * width
    stride = (image_tokens + text_tokens) * width
    with source.open("ab") as stream:
        np.zeros(1, dtype="<f4").tofile(stream)
        trace.condition.tofile(stream)
        for states in trace.streams:
            states.tofile(stream)
    placement = run(args, args.probe, profile, weights, geometry, source, output, 2, flags=("--reference-blocks",))

    def report(stage, actual, expected, **metadata):
        print(json.dumps({"stage": stage, **errors(actual, expected), **metadata}, allow_nan=False), flush=True)
        if tolerance is not None:
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance, err_msg=stage)

    for path, mode in ((output, "chained"), (output.with_suffix(".f32.forced"), "shared PyTorch inputs")):
        values = np.fromfile(path, dtype="<f4")
        condition_end = velocity_count + trace.condition.size
        velocity = unpack_latent(values[:velocity_count], *target.shape[1:])
        report("First velocity in diagnostic graph", velocity, expected_velocity, mode=mode, placement=placement)
        if path == output:
            report("Diagnostic graph versus standard velocity", velocity, standard_velocity, placement=placement)
            report("Time conditioning", values[velocity_count:condition_end], trace.condition, placement=placement)
        stream_values = values[condition_end:].reshape(len(trace.streams), stride)
        for block, (actual, expected) in enumerate(zip(stream_values, trace.streams, strict=True)):
            if block == 0 and path != output:
                continue
            report("Projected streams" if block == 0 else "Transformer block", actual[:image_count],
                   expected[:image_count], mode=mode, block=block, stream="image")
            report("Projected streams" if block == 0 else "Transformer block", actual[image_count:],
                   expected[image_count:], mode=mode, block=block, stream="text")
