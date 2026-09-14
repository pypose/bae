# Measured DSLR refinement results

Six ScanNet++ scenes were tested with the nested checkpoint, and two scene runs were repeated with the `DA3-GIANT-1.1` checkpoint used in the earlier DA3 experiments. This is a demonstration study, **not the full 50-scene validation benchmark**. Five scene/model combinations include all actual Open3D D, E and D+E references. Missing reference runs are marked —.

All numbers below are **raw optimizer AUC@5 percentages**, before the common guard. Higher is better. JSON and CSV store AUC as fractions. Every row uses identical inputs and the same scene-wide frames across methods. Frame sets differ between model runs, so the two checkpoints are not a controlled model bake-off.

| Model | Scene | Views | Initial | bae D | bae E | bae D+E | Open3D D | Open3D E | Open3D D+E |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NESTED-GIANT-LARGE | 09c1414f1b | 144 | 12.18 | 24.58 | 13.13 | 24.97 | 11.25 | 8.35 | 9.68 |
| NESTED-GIANT-LARGE | 0d2ee665be | 139 | 44.82 | 46.37 | 46.07 | 46.71 | 41.71 | 39.51 | 39.06 |
| NESTED-GIANT-LARGE | 3f15a9266d | 144 | 4.86 | 21.80 | 5.02 | 22.63 | 27.20 | 4.68 | 29.45 |
| NESTED-GIANT-LARGE | 6115eddb86 | 144 | 3.49 | 2.27 | 3.98 | 2.35 | — | — | — |
| NESTED-GIANT-LARGE | 7831862f02 | 144 | 57.14 | 49.88 | 59.40 | 50.69 | — | — | — |
| NESTED-GIANT-LARGE | cc5237fd77 | 144 | 4.42 | 5.93 | 4.55 | 6.07 | — | — | — |
| GIANT-1.1 | 3f15a9266d | 156 | 2.15 | 11.88 | 2.39 | 14.02 | 27.28 | 3.18 | 31.55 |
| GIANT-1.1 | 7831862f02 | 156 | 61.21 | 50.84 | 64.36 | 51.94 | 56.58 | 62.44 | 57.90 |

## What the results support

- The native solver improves pose accuracy on several real scene-wide DSLR predictions. In the selected bookshelf office, the nested-model result goes from **4.86 to 22.63** with bae D+E. The actual Open3D D+E reference reaches **29.45** on the same arrays. The direction agrees; the magnitude does not match.
- The compact home office has a smaller native D+E gain (**44.82 to 46.71**). The apartment improves numerically (**12.18 to 24.97**) but retains conspicuous reconstruction artifacts, so it was excluded from the main visual gallery.
- The lounge demonstrates that chaining is not always beneficial: native E alone improves accuracy, while D and D+E regress. Raw regressions are preserved in this report.
- The existing photometric guard rejected **39/39** evaluated proposals, including **24** proposals whose raw AUC@5 improved. Thus the guarded results return the initial poses. The main gallery intentionally shows labeled raw optimizer output; it is **not** a successful demonstration of the current guard.
- GIANT-1.1 does not remove the gap to the reference on the office. These measurements do not establish parity with Forms D/E across scenes.

## Coverage and memory

| Checkpoint | Scene | Selected / registered | Largest successful PyTorch allocation (GiB) | Next failed view count |
|---|---|---:|---:|---:|
| DA3NESTED-GIANT-LARGE | 09c1414f1b | 144 / 2391 | 22.72 | cap reached |
| DA3NESTED-GIANT-LARGE | 0d2ee665be | 139 / 189 | 22.29 | 144 |
| DA3NESTED-GIANT-LARGE | 3f15a9266d | 144 / 906 | 22.72 | 150 |
| DA3NESTED-GIANT-LARGE | 6115eddb86 | 144 / 597 | 22.72 | 150 |
| DA3NESTED-GIANT-LARGE | 7831862f02 | 144 / 380 | 22.72 | 150 |
| DA3NESTED-GIANT-LARGE | cc5237fd77 | 144 / 1353 | 22.72 | cap reached |
| DA3-GIANT-1.1 | 3f15a9266d | 156 / 906 | 22.47 | 162 |
| DA3-GIANT-1.1 | 7831862f02 | 156 / 380 | 22.47 | 162 |

Each successful prediction attends globally across the selected scene-wide frame set. The search is bracketed to eight views; it does not claim an exact maximum. The 139-frame run shared GPU capacity with another job during its initial search. The cap remains a user setting.

## Reproducibility and limitations

- Inputs: actual undistorted DSLR frames; full selected filenames and memory attempts are in [results.json](results.json). No COLMAP conditioning, synthetic noise, inpainting, or generated room imagery.
- Native D and Open3D D solve different problems; native E and Open3D E fuse different structures. All reference calls use the actual functions under `da3/`, with raw D+E composition and common post-hoc guarding as described in [SHOWCASE.md](../../SHOWCASE.md).
- Pose AUC uses every unordered frame pair. It is not a ground-truth surface metric and does not guarantee a visually clean mesh. No claim of map-to-LiDAR accuracy is made.
- The showcased scenes and views were selected after examining results for visual clarity. This is not an unbiased estimate of generalization. All eight tested scene/model combinations are included above.
- Wall times and PyTorch allocation peaks are retained in JSON. Workloads sometimes overlapped and the formulations use different sampling/iteration schedules, so these times are not a controlled speed comparison.
- Six initial scene runs plus two matching-checkpoint runs; seven example tests pass. Browser checks cover desktop/mobile layout, loaded assets, scene selection, slider state and timeline playback.

[Portable gallery](index.html) · [Machine-readable results](results.json) · [CSV](results.csv)
