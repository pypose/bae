from time import perf_counter
import argparse

import pypose as pp
import torch
import torch.nn as nn
from pypose.autograd.function import psjac

from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.utils.pysolvers import PCG

TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
# other options:
# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"

DEVICE = "cuda"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7


@psjac
def project(points, camera_params):
    projection = pp.SE3(camera_params[..., :7]).Act(points)
    projection = -projection[..., :2] / projection[..., [2]]

    f = camera_params[..., [-3]]
    k1 = camera_params[..., [-2]]
    k2 = camera_params[..., [-1]]

    n = torch.sum(projection**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    return projection * r * f


class Residual(nn.Module):
    def __init__(self, camera_params, points):
        super().__init__()
        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points = pp.Parameter(points, sjac=True)
        self.pose.trim_SE3_grad = True

    def forward(self, observes, cidx, pidx):
        points_proj = project(self.points[pidx], self.pose[cidx])
        return points_proj - observes


def least_square_error(camera_params, points, cidx, pidx, observes):
    model = Residual(camera_params, points)
    loss = model(observes, cidx, pidx)
    return torch.sum(loss**2, dim=-1).mean()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bundle adjustment with BAE")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=DEVICE)
    parser.add_argument("--dataset", default=TARGET_DATASET)
    parser.add_argument("--problem", default=TARGET_PROBLEM)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    device = torch.device(args.device)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            parser.error("MPS is unavailable; use an Apple Silicon Mac with MPS-enabled PyTorch")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device mps or --device cpu")
    dataset = get_problem(
        args.problem, args.dataset,
        dtype=torch.float32 if device.type == "mps" else torch.float64,
    )
    print(f"Fetched {args.problem} from {args.dataset}")

    dataset = {
        key: value.to(device)
        for key, value in dataset.items()
        if isinstance(value, torch.Tensor)
    }
    input = {
        "observes": dataset["points_2d"],
        "cidx": dataset["camera_index_of_observations"],
        "pidx": dataset["point_index_of_observations"],
    }

    model = Residual(
        dataset["camera_params"][:, :NUM_CAMERA_PARAMS].clone(),
        dataset["points_3d"].clone(),
    ).to(device)
    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = LM(
        model, strategy=strategy, solver=solver, reject=30,
        matrix_free_normal=device.type in ("cpu", "mps"),
    )

    print('Loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())

    print("Initial loss", optimizer.model.loss(input, None).item())

    _synchronize(device)
    start = perf_counter()
    for idx in range(args.iterations):
        loss = optimizer.step(input)
        print("Iteration", idx, "loss", loss.item(), "time", perf_counter() - start)

    _synchronize(device)
    end = perf_counter()
    print("Time", end - start)

    print('Ending loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())


if __name__ == "__main__":
    main()
