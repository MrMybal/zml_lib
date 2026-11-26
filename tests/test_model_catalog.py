import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.build import scratch_directory
from scripts import model_catalog as catalog


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.work = self.enterContext(scratch_directory())
        keys = ("TEMP", "TMP", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        self.enterContext(patch.dict(os.environ, {name: str(self.work) for name in keys}))

    def test_rrdb_uses_shapes_and_prefers_params_ema(self):
        import torch

        for index, profile in enumerate(catalog.rrdb.RUNTIME_VARIANTS):
            with self.subTest(profile=profile):
                checkpoint = self.work / f"unlabelled-{index}.pth"
                state = {name: torch.empty(shape, device="meta")
                         for name, shape in catalog.rrdb.expected_shapes(profile).items()}
                torch.save({"params": {"invalid": torch.empty(1)}, "params_ema": state}, checkpoint)
                self.assertEqual(catalog.identify(checkpoint), catalog.Match(profile, "rrdbnet"))
                state["conv_first.weight"] = torch.empty((64, 5, 3, 3), device="meta")
                torch.save({"params_ema": state}, checkpoint)
                with self.assertRaisesRegex(ValueError, "No supported checkpoint"):
                    catalog.identify(checkpoint)

    def test_sd15_dispatches_the_detected_profile(self):
        import torch

        for profile, channels in catalog.SD_PROFILES.items():
            with self.subTest(profile=profile):
                work = self.work / profile
                work.mkdir()
                checkpoint = work / f"{channels}.ckpt"
                prefixes = {"unet": "model.diffusion_model.", "clip": "cond_stage_model.transformer.",
                            "vae": "first_stage_model.decoder.", "encoder": "first_stage_model.encoder."}
                state = {}
                for role, (count, shapes) in catalog.sd_signatures(channels).items():
                    prefix = prefixes[role]
                    anchor_prefix = "first_stage_model." if role in ("vae", "encoder") else prefix
                    state.update({anchor_prefix + key: torch.empty(shape, device="meta")
                                  for key, shape in shapes.items()})
                    for index in range(count - len(shapes)):
                        state[f"{prefix}fixture.{index}"] = torch.empty(1, device="meta")
                torch.save({"state_dict": state}, checkpoint)
                self.assertEqual(catalog.identify(checkpoint), catalog.Match(profile, "compvis"))
                with self.assertRaisesRegex(ValueError, "--bin-dir"):
                    catalog.prepare(checkpoint, work / "cache")
                # Exercise dispatch and publishing separately from the native conversion.
                bin_dir = work / "bin"
                probe = bin_dir / "libraries/stable_diffusion/sd15_probe.exe"
                probe.parent.mkdir(parents=True, exist_ok=True)
                probe.write_bytes(b"schema fixture")

                def delegate(command, **_):
                    self.assertEqual(command[2], catalog.SD_DIR / "prepare.py")
                    self.assertEqual(command[command.index("--profile") + 1], profile)
                    bundle = Path(command[-1])
                    bundle.mkdir()
                    (bundle / "sd15.json").write_text(json.dumps({"profile": profile, "storage": "f32"}))
                    roles = ("unet", "clip", "vae", "encoder") if channels == 9 else ("unet", "clip", "vae")
                    for role in roles:
                        (bundle / (role + ".safetensors")).write_bytes(b"conversion fixture")

                with patch("scripts.model_catalog.subprocess.run", side_effect=delegate) as worker:
                    prepared = catalog.prepare(checkpoint, work / "cache", bin_dir)
                    reused = catalog.prepare(checkpoint, work / "cache", bin_dir)
                worker.assert_called_once()
                self.assertEqual((prepared["status"], reused["status"]), ("prepared", "cached"))
                self.assertEqual(prepared["path"], reused["path"])

    def test_cache_rejects_modified_bytes_and_uses_ready_weights(self):
        import torch
        from safetensors import safe_open

        profile = "RealESRGAN_x4plus_anime_6B"
        source = self.work / "input.pth"
        torch.save({"params": {name: torch.zeros(shape, dtype=torch.float16)
                              for name, shape in catalog.rrdb.expected_shapes(profile).items()}}, source)
        cache = self.work / "cache"
        result = catalog.prepare(source, cache)
        self.assertEqual(result["status"], "prepared")
        path = Path(result["path"])
        self.assertEqual(catalog.prepare(source, cache)["status"], "cached")
        self.assertEqual(catalog.prepare(path, self.work / "unused")["status"], "ready")
        self.assertFalse((self.work / "unused").exists())
        with safe_open(path, framework="numpy") as file:
            self.assertEqual(file.metadata()["checkpoint_key"], "params")
            self.assertEqual(file.get_slice("conv_first.weight").get_dtype(), "F32")
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "Invalid cache entry"):
            catalog.prepare(source, cache)
        self.assertEqual(list(cache.glob("prepare-*")), [])

    def test_standalone_component_profiles(self):
        def signature(count, anchors, prefix, dtype):
            tensors = {name: (shape, dtype) for name, shape in anchors.items()}
            tensors.update({f"{prefix}{index}": ((1,), dtype) for index in range(count - len(tensors))})
            return tensors

        count, clip_anchors = catalog.sd_signatures(4)["clip"]
        clip = signature(count, clip_anchors, "text_model.fixture.", "F16")
        self.assertEqual(catalog.component_profile(clip), catalog.Match("clip_l", "transformers_clip"))
        clip["text_model.embeddings.token_embedding.weight"] = ((49408, 1280), "F16")
        self.assertIsNone(catalog.component_profile(clip))

        t5 = signature(219, {
            "shared.weight": (32128, 4096), "encoder.final_layer_norm.weight": (4096,),
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": (32, 64),
            "encoder.block.23.layer.1.DenseReluDense.wi_0.weight": (10240, 4096),
        }, "encoder.fixture.", "F16")
        self.assertEqual(catalog.component_profile(t5), catalog.Match("t5_xxl", "transformers_t5", ready=True))
        t5["encoder.embed_tokens.weight"] = t5["shared.weight"]
        self.assertEqual(catalog.component_profile(t5), catalog.Match("t5_xxl", "transformers_t5"))
        t5["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = ((32, 64), "F8_E4M3")
        self.assertIsNone(catalog.component_profile(t5))

        vae = signature(138, {
            "decoder.conv_in.weight": (512, 16, 3, 3), "decoder.conv_out.weight": (3, 128, 3, 3),
            "decoder.mid.attn_1.q.weight": (512, 512, 1, 1),
        }, "decoder.fixture.", "F32")
        self.assertEqual(catalog.component_profile(vae), catalog.Match("flux1_vae", "flux1_autoencoder", ready=True))
        vae["encoder.conv_in.weight"] = ((128, 3, 3, 3), "F32")
        self.assertEqual(catalog.component_profile(vae), catalog.Match("flux1_vae", "flux1_autoencoder"))
        vae["decoder.conv_in.weight"] = ((512, 32, 3, 3), "F32")
        self.assertIsNone(catalog.component_profile(vae))

    def test_scan_distinguishes_unknown_broken_and_partial_files(self):
        import numpy as np
        from safetensors.numpy import save_file

        folder = self.work / "weights"
        folder.mkdir()
        save_file({"unknown.weight": np.zeros((2, 3), dtype=np.float16)}, folder / "unknown.safetensors")
        (folder / "broken.safetensors").write_bytes(b"invalid")
        (folder / "growing.safetensors.part").touch()
        (folder / "other.gguf").touch()
        (folder / "notes.txt").touch()
        inventory = catalog.scan(self.work)
        self.assertEqual(inventory["counts"], {"unsupported": 2, "unreadable": 1, "partial_download": 1})
        rows = {row["path"]: row for row in inventory["entries"]}
        self.assertEqual(rows["weights/unknown.safetensors"]["dtypes"], {"F16": 1})
        self.assertEqual(rows["weights/broken.safetensors"]["status"], "unreadable")
        self.assertNotIn("error", rows["weights/growing.safetensors.part"])


if __name__ == "__main__":
    unittest.main()
