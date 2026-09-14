"""ScanNet++ DSLR loader: undistorted images + real COLMAP ground-truth poses.

DSLR frames (unlike the iPhone stream) have accurate, bundle-adjusted COLMAP
poses as genuine ground truth, and no rough initial pose of their own -- the
"noisy initial guess" for these has to come from an actual predictor (e.g.
Depth Anything 3), not a stand-in. This module only handles the real,
verifiable input: reading the camera model + per-frame ground-truth poses,
and undistorting the fisheye DSLR images so they're plain pinhole for
downstream use.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import cv2
import numpy as np

from dataset import _ScratchZip, read_colmap_images_txt


def read_colmap_cameras_txt(path: str):
    """Minimal reader for COLMAP's public `cameras.txt` format. Returns
    {camera_id: {"model": str, "width": int, "height": int, "params": [float]}}."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            out[cam_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(x) for x in parts[4:]],
            }
    return out


def _undistort_maps_fisheye(camera: dict, balance: float = 0.0):
    """OPENCV_FISHEYE model (fx,fy,cx,cy,k1,k2,k3,k4) -> (map1, map2, new_K)."""
    fx, fy, cx, cy, k1, k2, k3, k4 = camera["params"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)
    W, H = camera["width"], camera["height"]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (W, H), np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (W, H), cv2.CV_16SC2)
    return map1, map2, new_K


def load_dslr_scene(scene_id: str, names: list[str], *,
                     dataset_root: str = "/data/zitong/scannetpp_val", max_size: int | None = None):
    """Load and undistort a specific list of DSLR frame names (e.g.
    "DSC04255.JPG") for one scene, plus their COLMAP ground-truth w2c poses.

    Returns (images (N,H,W,3) uint8 undistorted, K (3,3) float64 shared
    post-undistortion intrinsics, w2c_gt (N,4,4) float64).
    """
    scene_dir = os.path.join(dataset_root, "data", scene_id)
    colmap_zip = os.path.join(scene_dir, "dslr", "colmap.zip")

    with _ScratchZip(colmap_zip) as tmpdir:
        cameras = read_colmap_cameras_txt(os.path.join(tmpdir, "colmap", "cameras.txt"))
        gt_w2c_by_name = read_colmap_images_txt(os.path.join(tmpdir, "colmap", "images.txt"))

    camera = next(iter(cameras.values()))  # ScanNet++ DSLR: one shared camera model
    assert camera["model"] == "OPENCV_FISHEYE", f"unexpected DSLR camera model {camera['model']}"
    map1, map2, new_K = _undistort_maps_fisheye(camera)

    images_zip = os.path.join(scene_dir, "dslr", "resized_images.zip")
    images = []
    w2c_gt = []
    with zipfile.ZipFile(images_zip) as zf:
        namelist = {Path(n).name: n for n in zf.namelist()}
        for name in names:
            if name not in gt_w2c_by_name:
                raise KeyError(f"{name} has no COLMAP ground-truth pose in this scene")
            with zf.open(namelist[name]) as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            undist = cv2.remap(img_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
            if max_size is not None:
                h, w = undist.shape[:2]
                ratio = min(1.0, max_size / max(h, w))
                undist = cv2.resize(undist, (round(w * ratio), round(h * ratio)),
                                   interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
            w2c_gt.append(gt_w2c_by_name[name])

    if max_size is not None:
        new_K[0] *= images[0].shape[1] / camera["width"]
        new_K[1] *= images[0].shape[0] / camera["height"]
    return np.stack(images, axis=0), new_K.astype(np.float64), np.stack(w2c_gt, axis=0)


def list_dslr_frames(scene_id: str, n_views: int = 12, *,
                      dataset_root: str = "/data/zitong/scannetpp_val", seed: int = 0):
    """Pick `n_views` registered DSLR frame names for a scene, evenly spread
    through capture order (COLMAP image names sort to capture order for
    ScanNet++ DSLR)."""
    scene_dir = os.path.join(dataset_root, "data", scene_id)
    colmap_zip = os.path.join(scene_dir, "dslr", "colmap.zip")
    with _ScratchZip(colmap_zip) as tmpdir:
        gt_w2c_by_name = read_colmap_images_txt(os.path.join(tmpdir, "colmap", "images.txt"))
    names = sorted(gt_w2c_by_name.keys())
    if len(names) > n_views:
        idx = np.linspace(0, len(names) - 1, n_views).round().astype(int)
        idx = sorted(set(idx.tolist()))
        names = [names[i] for i in idx]
    return names
