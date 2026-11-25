"""Verify that memory batching preserves global structure visibility."""
import sys
from pathlib import Path
import torch
import pypose as pp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from photometric import zbuffer_visible_all_views


def test_batched_visibility_matches_all_views_with_occlusion_and_empty_views():
    dtype=torch.float64
    poses=pp.identity_SE3(11,dtype=dtype).tensor()
    poses[:,0]=torch.linspace(-1,1,11,dtype=dtype)
    poses[-1,0]=100 # completely empty last batch/view
    K=torch.tensor([[30.,0,16],[0,30,16],[0,0,1]],dtype=dtype).repeat(11,1,1)
    # Several depths in each projected bin, plus out-of-frustum points.
    generator=torch.Generator().manual_seed(3)
    points=torch.randn(250,3,generator=generator,dtype=dtype)
    points[:,2]=points[:,2].abs()+2
    points=torch.cat([points,points*2,torch.tensor([[0.,0.,-1.]],dtype=dtype)])
    full=zbuffer_visible_all_views(points,poses,K,32,32,view_batch_size=None)
    chunks=zbuffer_visible_all_views(points,poses,K,32,32,view_batch_size=3)
    assert full is not None and chunks is not None
    for key in full:
        torch.testing.assert_close(full[key],chunks[key],rtol=0,atol=0)
    assert not (chunks['v_idx']==10).any()


def test_gradients_match_symmetric_subpixel_image_derivative():
    from geometry import image_gradients,sample_at
    im=torch.tensor([[[0.,.3,.1,.8],[.4,.8,.2,.9],[.7,.1,.6,.3],[0.,.4,.9,1.]]],dtype=torch.float64)
    gx,gy=image_gradients(im)
    uv=torch.tensor([[1.5,1.5],[2.5,2.5]],dtype=torch.float64)
    eps=1e-5
    for axis,g in [(0,gx),(1,gy)]:
        shift=torch.zeros_like(uv);shift[:,axis]=eps
        numerical=(sample_at(im[0],uv+shift,4,4)-sample_at(im[0],uv-shift,4,4))/(2*eps)
        torch.testing.assert_close(sample_at(g[0],uv,4,4),numerical,rtol=1e-7,atol=1e-9)


def test_photo_correspondences_exclude_points_behind_measured_surface():
    from photometric import build_photo_correspondences
    from geometry import image_gradients
    poses=pp.identity_SE3(2,dtype=torch.float64).tensor()
    gray=torch.rand(2,12,12,dtype=torch.float64)
    depth=torch.ones_like(gray);depth[0]=2
    K=torch.tensor([[10.,0,6],[0,10,6],[0,0,1]],dtype=torch.float64).repeat(2,1,1)
    gx,gy=image_gradients(gray);pairs=torch.tensor([[0,1]])
    assert build_photo_correspondences(poses,pairs,gray,depth,K,gx,gy,n_samples=64,max_depth_diff=None) is not None
    assert build_photo_correspondences(poses,pairs,gray,depth,K,gx,gy,n_samples=64,max_depth_diff=.07) is None


def test_irls_refreshes_lm_loss_after_weight_changes(monkeypatch):
    import pytest
    import photometric
    from _scene_fixture import make_synthetic_scene
    if not torch.cuda.is_available():pytest.skip('sparse LM needs CUDA')
    checks=[]
    class AuditedLM(photometric.LM):
        def step(self,inp):
            expected=self.model.loss(inp,None)
            torch.testing.assert_close(self.loss.tensor(),expected.tensor())
            checks.append(1)
            result=super().step(inp)
            # Mimic a stale cache from a differently weighted previous step.
            self.loss=self.loss+12345
            return result
    monkeypatch.setattr(photometric,'LM',AuditedLM)
    images,depth,K,gt,initial=make_synthetic_scene(n_views=3,H=32,W=32,device='cuda',dtype=torch.float64)
    K=K[None].expand(3,-1,-1).contiguous()
    photometric.refine_odometry(initial,depth,K,images,n_neighbors=2,n_samples=64,
         pyramid_levels=1,n_relin=(1,),n_inner=3,huber_delta=1.5,verbose=False)
    assert len(checks)==3


def test_composed_pose_sparse_jacobian_matches_left_retraction():
    """Nonidentity poses exercise ambient-to-tangent conversion through Inv/Act."""
    from bae.autograd.graph import jacobian
    from photometric import RGBDPhotoModel
    from geometry import project_to_pixels
    from bae.utils.pypose_ambient_grad import pypose_ambient_grad_enabled
    assert pypose_ambient_grad_enabled()
    dtype = torch.float64
    c2w = pp.se3(torch.tensor([[.1, -.2, .05, .1, -.05, .2],
                               [.4, .1, -.1, -.15, .2, .05]], dtype=dtype)).Exp().tensor()
    model = RGBDPhotoModel(c2w)
    points = torch.tensor([[.1, .2, 3.], [-.3, .1, 4.], [.5, -.2, 2.]], dtype=dtype)
    ki = torch.tensor([[100., 0, 48], [0, 100, 48], [0, 0, 1]], dtype=dtype).repeat(3, 1, 1)
    i, j = torch.tensor([0, 1, 0]), torch.tensor([1, 0, 1])
    uv, _ = project_to_pixels(pp.SE3(c2w[j]).Inv().Act(pp.SE3(c2w[i]).Act(points)), ki)
    inp = (i, j, points, ki, torch.zeros(3, dtype=dtype), torch.ones(3, dtype=dtype),
           torch.tensor([.1, -.05, .08], dtype=dtype), torch.tensor([.03, .1, -.04], dtype=dtype),
           uv, torch.ones(3, 1, dtype=dtype))
    analytic = jacobian(model(*inp), [model.pose_rest])[0].to_dense()
    numerical = torch.empty(3, 6, dtype=dtype)
    eps = 1e-6
    with torch.no_grad():
        for k in range(6):
            tangent = torch.zeros(1, 6, dtype=dtype); tangent[0, k] = eps
            model.pose_rest.copy_((pp.se3(tangent).Exp() * pp.SE3(c2w[1:])).tensor())
            plus = model(*inp).clone()
            model.pose_rest.copy_((pp.se3(-tangent).Exp() * pp.SE3(c2w[1:])).tensor())
            minus = model(*inp).clone()
            numerical[:, k] = ((plus - minus) / (2 * eps)).flatten()
        model.pose_rest.copy_(c2w[1:])
    torch.testing.assert_close(analytic, numerical, atol=1e-7, rtol=1e-6)
