"""Use the pinned GVHMR SimpleVO implementation for moving cameras."""

import importlib
from importlib.machinery import PathFinder
import importlib.util
import sys

import numpy as np


def load_pycolmap(site):
    if site is None:
        try:
            return importlib.import_module("pycolmap")
        except ModuleNotFoundError as error:
            if error.name != "pycolmap":
                raise
            raise RuntimeError("Moving-camera GVHMR needs PyCOLMAP. Point --pycolmap-site to its local site-packages directory.") from error
    directory = site.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("--pycolmap-site must be a site-packages directory")
    # Load only PyCOLMAP from the shared CyTools runtime. Its wheel finds the
    # adjacent pycolmap.libs directory without exposing the rest of that runtime.
    spec = PathFinder.find_spec("pycolmap", [str(directory)])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"PyCOLMAP is missing from {directory}")
    if spec.name in sys.modules:
        module = sys.modules[spec.name]
        if getattr(module, "__file__", None) != spec.origin:
            raise RuntimeError("PyCOLMAP was already loaded from another Python environment")
        return module
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


def estimate_rotations(video, frames):
    name = "hmr4d.utils.preproc.relpose.simple_vo"
    stub = sys.modules.get(name)
    # The fixed-camera bridge registers an in-memory SimpleVO stub. The person
    # feature extractors already ran, so load the real odometry module here.
    if stub is not None and getattr(stub, "__file__", None) is None:
        del sys.modules[name]
    simple_vo = importlib.import_module(name).SimpleVO
    trajectory = np.asarray(simple_vo(str(video), scale=0.5, step=8, method="sift", f_mm=None).compute())
    if trajectory.shape != (frames, 4, 4) or not np.isfinite(trajectory).all():
        raise ValueError("SimpleVO did not provide a finite pose for each canonical video frame")
    rotation = trajectory[:, :3, :3]
    if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-3)):
        raise ValueError("SimpleVO returned an invalid world-to-camera rotation")
    return np.ascontiguousarray(rotation)


def max_rotation_degrees(rotation):
    precise = np.asarray(rotation, dtype=np.float64)
    relative = precise @ precise[0].T
    cosine = (np.trace(relative, axis1=1, axis2=2) - 1) / 2
    skew = relative - relative.transpose(0, 2, 1)
    sine = np.linalg.norm(skew[:, (2, 0, 1), (1, 2, 0)], axis=1) / 2
    return float(np.rad2deg(np.arctan2(sine, cosine)).max())
