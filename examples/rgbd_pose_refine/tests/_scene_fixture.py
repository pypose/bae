"""Textured-plane fixture for numerical correctness tests only."""
import torch
import pypose as pp
from geometry import affine_inv, unproject_pixels


def make_synthetic_scene(n_views=6, H=96, W=96, fov_deg=50.0, plane_z=5.0,
                           baseline=0.3, rot_perturb_deg=1.5, trans_perturb=0.05,
                           seed=0, device="cpu", dtype=torch.float64):
    """N cameras on a small dolly rig facing a smooth-textured frontal plane.

    Returns (images uint8 (N,H,W,3), depth (N,H,W), K (3,3), w2c_gt (N,4,4),
    w2c_init (N,4,4) with a known perturbation on views 1..N-1).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    fx = fy = W / (2.0 * torch.tan(torch.tensor(fov_deg * torch.pi / 360.0)))
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device)

    # GT extrinsics: small lateral dolly + small random rotation per view.
    t_gt = torch.stack([torch.tensor([i * baseline, 0.0, 0.0]) for i in range(n_views)]).to(dtype)
    rvecs = torch.randn(n_views, 3, generator=g).to(dtype) * (2.0 * torch.pi / 180.0)
    rvecs[0] = 0.0
    R_gt = pp.so3(rvecs).Exp().matrix()
    c2w_gt = torch.eye(4, dtype=dtype).repeat(n_views, 1, 1)
    c2w_gt[:, :3, :3] = R_gt
    c2w_gt[:, :3, 3] = t_gt
    c2w_gt = c2w_gt.to(device)

    # Pixel grid, pixel centers.
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dtype, device=device),
                             torch.arange(W, dtype=dtype, device=device), indexing="ij")
    uv = torch.stack([xs + 0.5, ys + 0.5], dim=-1).reshape(-1, 2)   # (HW,2)
    dirs_cam = unproject_pixels(uv, torch.ones(uv.shape[0], dtype=dtype, device=device), K)  # z=1 ray

    images = torch.zeros(n_views, H, W, 3, dtype=torch.uint8, device=device)
    depth = torch.zeros(n_views, H, W, dtype=dtype, device=device)
    for i in range(n_views):
        Rw = c2w_gt[i, :3, :3]
        ow = c2w_gt[i, :3, 3]
        dirs_w = torch.einsum("ij,mj->mi", Rw, dirs_cam)           # (HW,3)
        t = (plane_z - ow[2]) / dirs_w[:, 2].clamp(min=1e-6)
        Xw = ow[None, :] + t[:, None] * dirs_w
        # Period (3.0) chosen well above the world-space displacement a
        # perturbed pose can induce at this depth/rotation scale, so the
        # optimizer cannot alias onto a neighboring texture cycle.
        period = 3.0
        tex = (0.5 + 0.25 * torch.sin(2 * torch.pi * Xw[:, 0] / period)
               + 0.25 * torch.sin(2 * torch.pi * Xw[:, 1] / period))
        tex = tex.clamp(0.0, 1.0)
        img = (tex * 255.0).round().to(torch.uint8)
        images[i] = img.reshape(H, W)[..., None].expand(H, W, 3)
        depth[i] = t.reshape(H, W)

    w2c_gt = affine_inv(c2w_gt)

    # Perturb views 1..N-1's initial guess.
    d_rvec = torch.randn(n_views, 3, generator=g).to(dtype) * (rot_perturb_deg * torch.pi / 180.0)
    d_t = torch.randn(n_views, 3, generator=g).to(dtype) * trans_perturb
    d_rvec[0] = 0.0
    d_t[0] = 0.0
    dR = pp.so3(d_rvec).Exp().matrix().to(device)
    c2w_init = c2w_gt.clone()
    c2w_init[:, :3, :3] = c2w_gt[:, :3, :3] @ dR
    c2w_init[:, :3, 3] = c2w_gt[:, :3, 3] + d_t.to(device)
    w2c_init = affine_inv(c2w_init)

    return images, depth, K, w2c_gt, w2c_init
