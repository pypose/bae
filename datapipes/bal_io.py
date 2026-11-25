import os

import torch

DTYPE = torch.float64


def _rotvec_to_quat_xyzw(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half_theta = 0.5 * theta
    cos_half = torch.cos(half_theta)
    sin_half = torch.sin(half_theta)

    eps = 1e-12
    scale = torch.where(theta > eps, sin_half / theta, 0.5 - (theta * theta) / 48.0)
    xyz = rotvec * scale
    return torch.cat([xyz, cos_half], dim=-1)


def read_bal_data(file_name: str, use_quat: bool = True, *, dtype: torch.dtype = DTYPE) -> dict:
    """
    Read a Bundle Adjustment in the Large dataset problem text file.
    Floating-point tensors use ``dtype`` (default: torch.float64); indices use torch.int64.

    Format:
      <num_cameras> <num_points> <num_observations>
      <camera_index> <point_index> <x> <y>    (repeated n_observations)
      <camera parameters>                    (n_cameras * 9 lines)
      <point parameters>                     (n_points * 3 lines)

    Each camera has 9 parameters: Rodrigues rotvec (3), translation (3), f, k1, k2.
    This loader outputs either:
      - use_quat=True:  [tx, ty, tz, qx, qy, qz, qw, f, k1, k2] (10)
      - use_quat=False: [tx, ty, tz, rx, ry, rz, f, k1, k2]  (9)
    """
    with open(file_name, "r") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())

        camera_indices = torch.empty(n_observations, dtype=torch.int64)
        point_indices = torch.empty(n_observations, dtype=torch.int64)
        points_2d = torch.empty((n_observations, 2), dtype=dtype)

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i, 0] = float(x)
            points_2d[i, 1] = float(y)

        camera_params = torch.empty(n_cameras * 9, dtype=dtype)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, 9))

        points_3d = torch.empty(n_points * 3, dtype=dtype)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, 3))

    if use_quat:
        q = _rotvec_to_quat_xyzw(camera_params[:, :3])
        camera_params = torch.cat([camera_params[:, 3:6], q, camera_params[:, 6:]], dim=1)
    else:
        camera_params = torch.cat([camera_params[:, 3:6], camera_params[:, :3], camera_params[:, 6:]], dim=1)

    return {
        "problem_name": os.path.splitext(os.path.basename(file_name))[0],
        "camera_params": camera_params.to(dtype),
        "points_3d": points_3d,
        "points_2d": points_2d,
        "camera_index_of_observations": camera_indices,
        "point_index_of_observations": point_indices,
    }
