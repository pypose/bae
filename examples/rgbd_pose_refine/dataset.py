"""ScanNet++ iPhone RGB-D loader for the raw official dataset layout at
`/data/zitong/scannetpp_val` (NOT the DA3-BENCH preprocessed copy used by
`da3/`'s own dataset code, which expects a different `merge_dslr_iphone`
layout that isn't present in this raw download).

Per-scene sources (all confirmed by direct inspection of the raw dataset,
see README):
  - RGB: already-extracted frames at `<frames_root>/<scene_id>/frame_%06d.jpg`
    (no extraction needed), OR the full per-scene video at
    `<video_root>/<scene_id>.mkv` as a fallback (all frames, decoded via
    `cv2.VideoCapture`).
  - Depth: `<scene>/iphone/depth.zip` -> `depth.bin`, a sequence of frames
    each a 4-byte little-endian length prefix + an LZ4-block-compressed (with
    raw-deflate zlib fallback) 192x256 uint16-millimeter frame -- this is the
    official ScanNet++ toolkit format (`scannetpp/iphone/prepare_iphone_data.py
    ::extract_depth`), verified against this dataset copy. Streamed directly
    out of the zip via `zipfile.ZipFile.open()` without ever writing
    `depth.bin` (~575MB/scene) to disk.
  - Initial pose + intrinsics: `<scene>/iphone/pose_intrinsic_imu.zip` ->
    `pose_intrinsic_imu.json`, keyed `"frame_%06d"`, one entry per depth frame.
    ARKit `pose` is camera-to-world in ARKit's right-up-back convention and is
    converted to OpenCV's right-down-forward convention by right-multiplying
    diag(1,-1,-1,1) (flips the local Y/Z camera axes) -- verify this against
    the COLMAP pseudo-GT on a few frames before trusting it broadly (see
    README "Known risks").
  - Pseudo-GT for evaluation: `<scene>/iphone/colmap.zip` ->
    `colmap/{cameras,images}.txt`, standard COLMAP OPENCV-model text format,
    parsed with the existing `depth_anything_3.utils.read_write_model`
    utilities (not a new parser).

The two small zips (colmap, pose_intrinsic_imu) are fully extracted to a
scratch directory, parsed, then deleted (`shutil.rmtree`); depth.zip is never
extracted to disk at all. Decoded results are cached to one compact
`<cache_root>/<scene_id>.pt` per scene so repeat runs skip re-decoding.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path

import cv2
import lz4.block
import numpy as np
import torch
import zlib

DEFAULT_DATASET_ROOT = "/data/zitong/scannetpp_val"
DEFAULT_FRAMES_ROOT = "/data/zitong/scannetpp_iphone_frames"
DEFAULT_VIDEO_ROOT = "/data/zitong/scannetpp_iphone_256"

_DEPTH_H, _DEPTH_W = 192, 256
_ARKIT_TO_OPENCV = torch.tensor([[1., 0, 0, 0], [0, -1., 0, 0],
                                  [0, 0, -1., 0], [0, 0, 0, 1.]], dtype=torch.float64)


def _frame_idx_from_name(name: str) -> int:
    stem = Path(name).stem  # "frame_000001"
    return int(stem.split("_")[-1])


def _decode_depth_frame(payload: bytes) -> np.ndarray:
    """One `depth.bin` chunk -> (192,256) uint16 mm. LZ4-block primary codec,
    raw-deflate zlib fallback, per the official toolkit's own decoder."""
    try:
        raw = lz4.block.decompress(payload, uncompressed_size=_DEPTH_H * _DEPTH_W * 2)
    except Exception:
        raw = zlib.decompress(payload, wbits=-zlib.MAX_WBITS)
    return np.frombuffer(raw, dtype=np.uint16).reshape(_DEPTH_H, _DEPTH_W)


def iter_depth_frames(depth_zip_path: str, wanted_indices):
    """Yields (frame_idx, depth_mm (192,256) uint16) for indices in
    `wanted_indices`, streaming directly out of the zip. Never writes
    `depth.bin` to disk: `ZipFile.open()` exposes the zip's own DEFLATE
    decompression as a plain sequential byte stream, which is exactly what
    the length-prefix framing needs (only sequential reads, no seeking into
    the logical depth.bin stream are required)."""
    wanted = set(wanted_indices)
    if not wanted:
        return
    max_idx = max(wanted)
    with zipfile.ZipFile(depth_zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith("depth.bin"))
        with zf.open(name) as f:
            idx = 0
            while idx <= max_idx:
                hdr = f.read(4)
                if len(hdr) < 4:
                    break
                n = struct.unpack("<I", hdr)[0]
                payload = f.read(n)
                if idx in wanted:
                    yield idx, _decode_depth_frame(payload)
                idx += 1


class _ScratchZip:
    """Extract a small zip to a temp dir; guarantees cleanup on exit."""

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.tmpdir = None

    def __enter__(self) -> str:
        self.tmpdir = tempfile.mkdtemp(prefix="rgbd_pose_refine_")
        with zipfile.ZipFile(self.zip_path) as zf:
            zf.extractall(self.tmpdir)
        return self.tmpdir

    def __exit__(self, *exc):
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


def load_colmap_pseudo_gt(colmap_zip_path: str):
    """Parse `iphone/colmap.zip` via the existing DA3 COLMAP text-model
    reader. Returns {frame_name: (4,4) w2c float64 numpy}."""
    import sys
    da3_src = str(Path(__file__).resolve().parents[2] / "da3" / "src")
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
    from depth_anything_3.utils.read_write_model import read_images_text

    with _ScratchZip(colmap_zip_path) as tmpdir:
        images_txt = os.path.join(tmpdir, "colmap", "images.txt")
        images = read_images_text(images_txt)
        out = {}
        for img in images.values():
            ext = np.eye(4, dtype=np.float64)
            ext[:3, :3] = img.qvec2rotmat()
            ext[:3, 3] = img.tvec
            out[img.name] = ext
    return out


def load_pose_intrinsic_imu(zip_path: str):
    """Parse `iphone/pose_intrinsic_imu.zip`. Returns dict keyed by frame
    index (int) -> {"pose_arkit": (4,4) float64, "K": (3,3) float64}."""
    with _ScratchZip(zip_path) as tmpdir:
        json_path = os.path.join(tmpdir, "pose_intrinsic_imu.json")
        with open(json_path) as f:
            data = json.load(f)
    out = {}
    for key, entry in data.items():
        idx = _frame_idx_from_name(key)
        out[idx] = {
            "pose_arkit": np.asarray(entry["pose"], dtype=np.float64),
            "K": np.asarray(entry["intrinsic"], dtype=np.float64),
        }
    return out


def arkit_to_opencv_pose(c2w_arkit: torch.Tensor) -> torch.Tensor:
    """(...,4,4) ARKit c2w (right-up-back) -> OpenCV c2w (right-down-forward)
    by right-multiplying diag(1,-1,-1,1) (flips the camera's local Y/Z axes;
    translation is unaffected since this is a right-multiplication of the
    local frame, not a world-frame transform).

    This is the standard OpenGL/ARKit -> OpenCV convention flip (ARKit uses
    the same right-up-back convention as OpenGL) -- confirmed against the
    equivalent conversion in the dust3r ScanNet++ preprocessing pipeline.
    Still verify against the COLMAP pseudo-GT on a few frames for THIS
    dataset copy before trusting it broadly; see `verify_arkit_convention`.
    """
    M = _ARKIT_TO_OPENCV.to(c2w_arkit.dtype).to(c2w_arkit.device)
    return c2w_arkit @ M


def verify_arkit_convention(scene_dir: str, n_check: int = 5) -> float:
    """Sanity check: compare relative poses between a few frame pairs, as
    given by the ARKit-converted poses vs. the COLMAP pseudo-GT poses (for
    the subset of frames COLMAP registered). Returns the mean relative
    rotation-angle disagreement in degrees (should be small, a few degrees at
    most, if the axis convention is right; ~90-180 degrees would indicate a
    wrong flip)."""
    from guards import rel_rot_deg
    from geometry import affine_inv, as_44

    pim = load_pose_intrinsic_imu(os.path.join(scene_dir, "iphone", "pose_intrinsic_imu.zip"))
    gt = load_colmap_pseudo_gt(os.path.join(scene_dir, "iphone", "colmap.zip"))
    gt_idx = sorted(_frame_idx_from_name(n) for n in gt.keys())
    gt_idx = [i for i in gt_idx if i in pim][:max(n_check, 2)]
    if len(gt_idx) < 2:
        raise RuntimeError("not enough overlapping frames to verify convention")

    c2w_arkit = torch.stack([torch.from_numpy(pim[i]["pose_arkit"]) for i in gt_idx])
    c2w_cv = arkit_to_opencv_pose(c2w_arkit)
    w2c_arkit_cv = affine_inv(as_44(c2w_cv))

    name_by_idx = {_frame_idx_from_name(n): n for n in gt.keys()}
    w2c_colmap = torch.stack([torch.from_numpy(gt[name_by_idx[i]]) for i in gt_idx])

    idx_i = torch.arange(1, len(gt_idx))
    idx_j = torch.zeros(len(gt_idx) - 1, dtype=torch.long)
    err_arkit = rel_rot_deg(w2c_arkit_cv, idx_i, idx_j)
    err_colmap = rel_rot_deg(w2c_colmap, idx_i, idx_j)
    return float((err_arkit - err_colmap).abs().mean())


def load_scene(scene_id: str, *, dataset_root: str = DEFAULT_DATASET_ROOT,
               frames_root: str = DEFAULT_FRAMES_ROOT,
               video_root: str = DEFAULT_VIDEO_ROOT,
               cache_root: str = None, max_views: int = 60,
               force_reload: bool = False):
    """Load one scene's RGB, depth, initial pose+K, and COLMAP pseudo-GT
    (for whichever frames it registered), from the raw ScanNet++ layout.

    Returns a dict: images (N,H,W,3) uint8, depth (N,H,W) float32 meters,
    K_init (N,3,3) float64, w2c_init (N,4,4) float64, frame_names (N,) list,
    gt_w2c (dict: frame_name -> (4,4) float64, only for registered frames).
    """
    if cache_root is None:
        cache_root = str(Path(tempfile.gettempdir()) / "rgbd_pose_refine_cache")
    os.makedirs(cache_root, exist_ok=True)
    cache_path = os.path.join(cache_root, f"{scene_id}_mv{max_views}.pt")
    if os.path.exists(cache_path) and not force_reload:
        return torch.load(cache_path, weights_only=False)

    scene_dir = os.path.join(dataset_root, "data", scene_id)
    frame_dir = os.path.join(frames_root, scene_id)
    video_path = os.path.join(video_root, f"{scene_id}.mkv")

    if os.path.isdir(frame_dir):
        names = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
        indices = [_frame_idx_from_name(n) for n in names]
        if max_views and len(indices) > max_views:
            sel = np.linspace(0, len(indices) - 1, max_views).round().astype(int)
            sel = sorted(set(sel.tolist()))
            names = [names[i] for i in sel]
            indices = [indices[i] for i in sel]
        images = np.stack([cv2.cvtColor(cv2.imread(os.path.join(frame_dir, n)), cv2.COLOR_BGR2RGB)
                            for n in names], axis=0)
    elif os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        n_full = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_views and n_full > max_views:
            indices = np.linspace(0, n_full - 1, max_views).round().astype(int).tolist()
        else:
            indices = list(range(n_full))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {idx} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        images = np.stack(frames, axis=0)
    else:
        raise FileNotFoundError(f"no RGB source found for scene {scene_id} "
                                 f"(checked {frame_dir} and {video_path})")

    H, W = images.shape[1:3]

    pim = load_pose_intrinsic_imu(os.path.join(scene_dir, "iphone", "pose_intrinsic_imu.zip"))
    missing = [i for i in indices if i not in pim]
    if missing:
        raise RuntimeError(f"{len(missing)} frame indices missing from pose_intrinsic_imu.json, "
                            f"e.g. {missing[:5]}")

    depth_by_idx = dict(iter_depth_frames(
        os.path.join(scene_dir, "iphone", "depth.zip"), indices))
    depth = np.stack([depth_by_idx[i].astype(np.float32) / 1000.0 for i in indices], axis=0)
    depth = torch.from_numpy(depth)[:, None]
    depth = torch.nn.functional.interpolate(depth, size=(H, W), mode="nearest")[:, 0].numpy()

    c2w_arkit = torch.stack([torch.from_numpy(pim[i]["pose_arkit"]) for i in indices])
    c2w_cv = arkit_to_opencv_pose(c2w_arkit)
    from geometry import affine_inv, as_44
    w2c_init = affine_inv(as_44(c2w_cv)).numpy()
    K_init = np.stack([pim[i]["K"] for i in indices], axis=0)

    gt_w2c = load_colmap_pseudo_gt(os.path.join(scene_dir, "iphone", "colmap.zip"))
    frame_names = [f"frame_{i:06d}.jpg" for i in indices]

    scene = {
        "scene_id": scene_id,
        "images": images,
        "depth": depth,
        "K_init": K_init,
        "w2c_init": w2c_init,
        "frame_names": frame_names,
        "gt_w2c": gt_w2c,
    }
    torch.save(scene, cache_path)
    return scene
