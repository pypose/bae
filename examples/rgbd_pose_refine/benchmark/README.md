# Benchmarks and reproduction

These scripts use the core refinement modules in the parent directory.
Run them from the repository root:

| Script | Purpose |
|---|---|
| `showcase_prepare.py` | Cache DA3 predictions from scene-wide DSLR photographs |
| `showcase_benchmark.py` | Run BAE D, E, D+E and optional Open3D reference comparisons |
| `validate_split.py` | Schedule and resume a complete split across GPUs |
| `showcase_report.py` | Summarize showcase runs |
| `validation_report.py` | Audit split results against the separate DA3 evaluator |

```bash
python examples/rgbd_pose_refine/benchmark/showcase_benchmark.py \
  --scenes 7831862f02 --variants bae_d bae_e bae_de
```

This command consumes `outputs/rgbd_showcase/7831862f02/prediction.npz`.
Preparation needs DA3 inference dependencies. BAE-only refinement of an existing
cache does not import the reference implementation. Reference comparisons and
the independent validation audit require the separate `da3/` checkout.

See [SHOWCASE.md](../SHOWCASE.md) for installation and preparation commands and
[VALIDATION.md](../VALIDATION.md) for the split protocol. Outputs stay under
ignored `outputs/`; published reports stay under `../docs/`.
