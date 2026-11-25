"""The DSLR path needs only its own image and COLMAP archives."""
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dslr import list_dslr_frames, load_dslr_scene


def test_dslr_archives_preserve_frame_order_and_colmap_poses(tmp_path):
    folder = tmp_path / "data" / "scene" / "dslr"
    folder.mkdir(parents=True)
    # Each pose line is followed by an empty POINTS2D line. The second camera
    # has a 90-degree rotation about Z, exercising quaternion conversion too.
    q = np.sqrt(0.5)
    poses = (
        "# COLMAP images\n"
        "1 1 0 0 0 1 2 3 1 first.JPG\n\n"
        f"2 {q} 0 0 {q} 4 5 6 1 second.JPG\n\n"
    )
    with zipfile.ZipFile(folder / "colmap.zip", "w") as archive:
        archive.writestr("colmap/images.txt", poses)
        archive.writestr("colmap/cameras.txt",
                         "1 OPENCV_FISHEYE 16 12 12 12 8 6 0 0 0 0\n")
    with zipfile.ZipFile(folder / "resized_images.zip", "w") as archive:
        for name, color in [("first.JPG", (20, 80, 160)),
                            ("second.JPG", (100, 40, 10))]:
            image = np.full((12, 16, 3), color, dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            assert ok
            archive.writestr("images/" + name, encoded.tobytes())

    assert list_dslr_frames("scene", dataset_root=tmp_path) == ["first.JPG", "second.JPG"]
    assert list_dslr_frames("scene", n_views=1, dataset_root=tmp_path) == ["first.JPG"]
    images, K, gt = load_dslr_scene("scene", ["second.JPG", "first.JPG"],
                                    dataset_root=tmp_path)
    assert images.shape == (2, 12, 16, 3)
    assert images.dtype == np.uint8
    # OpenCV decodes BGR; the public loader returns RGB in requested order.
    np.testing.assert_allclose(images[:, 6, 8], [[10, 40, 100], [160, 80, 20]], atol=2)
    np.testing.assert_allclose(gt[:, :3, 3], [[4, 5, 6], [1, 2, 3]])
    np.testing.assert_allclose(gt[0, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_array_equal(gt[1, :3, :3], np.eye(3))

    small, small_K, small_gt = load_dslr_scene(
        "scene", ["second.JPG", "first.JPG"], dataset_root=tmp_path, max_size=8)
    assert small.shape == (2, 6, 8, 3)
    np.testing.assert_allclose(small_K[:2], K[:2] * 0.5)
    np.testing.assert_array_equal(small_K[2], K[2])
    np.testing.assert_array_equal(small_gt, gt)
