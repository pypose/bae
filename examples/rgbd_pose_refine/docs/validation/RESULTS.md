# Full DSLR validation

Coverage: **50/50 scenes** from `nvs_sem_val.txt`; 604,206 unordered camera pairs. Checkpoint: `depth-anything/DA3-GIANT-1.1` at `72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19`.

All methods receive the same globally attended, undistorted DSLR prediction for each scene. `BAE_USE_PYPOSE_AMBIENT_GRAD=1` is enabled. No scene is selected using its AUC. Raw and guarded outcomes are separate. All table AUCs are percentages; JSON stores fractions.

## Macro AUC (each scene has equal weight)

| Method | AUC@3 | AUC@5 | AUC@10 | AUC@20 | AUC@30 | Improved / regressed at 5° |
|---|---:|---:|---:|---:|---:|---:|
| da3 | 15.682 | 29.038 | 49.776 | 69.128 | 77.851 | 0 / 0 |
| bae_d_raw | 23.066 | 36.825 | 56.806 | 74.036 | 81.457 | 44 / 6 |
| bae_d | 16.280 | 29.460 | 49.996 | 69.236 | 77.923 | 3 / 0 |
| bae_e_raw | 17.326 | 30.394 | 50.605 | 69.561 | 78.143 | 45 / 5 |
| bae_e | 15.996 | 29.254 | 49.885 | 69.183 | 77.888 | 5 / 0 |
| bae_de_raw | 23.114 | 36.890 | 56.872 | 74.059 | 81.478 | 44 / 6 |
| bae_de | 16.314 | 29.481 | 50.004 | 69.240 | 77.925 | 3 / 0 |
| form_d_raw | 20.597 | 35.871 | 56.190 | 72.344 | 79.470 | 36 / 14 |
| form_d | 15.152 | 28.541 | 49.438 | 68.937 | 77.713 | 0 / 3 |
| form_e_raw | 17.930 | 29.819 | 48.906 | 67.926 | 76.799 | 24 / 26 |
| form_e | 15.778 | 29.049 | 49.750 | 69.102 | 77.827 | 1 / 2 |
| form_de_raw | 22.619 | 37.419 | 57.107 | 72.698 | 79.653 | 36 / 14 |
| form_de | 15.767 | 29.020 | 49.694 | 69.060 | 77.792 | 2 / 2 |

## Does native refinement match the reference?

| Raw stage | bae − reference AUC@5 (pp) | Largest scene gap (pp) | Within 1 pp |
|---|---:|---:|---:|
| D | +0.954 | 23.159 | 3/50 |
| E | +0.575 | 10.487 | 20/50 |
| DE | -0.529 | 20.370 | 5/50 |

Native D minimizes a joint photometric objective; Open3D D performs dense pairwise odometry followed by PGO. Native E rebuilds a voxel-averaged structure; reference E uses TSDF fusion and rigid color-map optimization. These are different formulations, so numerical equality is not guaranteed by correct gradients.

## Guard decisions

Improvement and regression below refer to ground-truth AUC@5, which the guard does not receive.

| Method | Accepted | Rejected AUC improvements | Rejected AUC regressions |
|---|---:|---:|---:|
| bae_d | 3/50 | 41 | 6 |
| bae_e | 5/50 | 40 | 5 |
| bae_de | 3/50 | 41 | 6 |
| form_d | 3/50 | 36 | 11 |
| form_e | 3/50 | 23 | 24 |
| form_de | 4/50 | 34 | 12 |

## Metric and historical-number audit

- Every saved result was rescored with `da3/eval_pose_auc_scannetpp.py::pair_errors_idx` and its trapezoidal `error_auc`. Maximum difference from the example evaluator across all scenes, methods and thresholds: **8.1614817e-05 AUC**.
- Main tables score all methods in float64. The original CLI uses float32; that path was independently run for every result. Maximum float64/float32 AUC difference: **0.001629 percentage points**. Both sets of per-scene scores are in JSON.
- The local historical JSON covers `nvs_test.txt`. The mounted dataset is `nvs_sem_val.txt`: **0 shared scenes**. Its baseline macro AUC@5 is 39.771%, but it is not a reproduction target for this different split and frame budget.
- The reference arms here call the actual DA3 Form D/E functions on the same cached arrays as bae. They use the showcase protocol: shared self-calibration, graph baseline filter 0.1 / rotation 1°, D texture gate and internal guard disabled, then a common 512-pair wide guard. D+E composes raw D then E and guards the whole proposal. This differs from the historical CLI defaults.
- Parallel odometry check: 28 real DSLR edges gave identical success flags, transforms and information matrices with one versus eight edge workers at one OpenMP thread. Against the original eight-OpenMP-thread serial controls on two scenes, the largest D/E/D+E AUC difference over all five thresholds was 0.09320 percentage points. Changing floating-point reduction order is not bitwise identical; the measured controls are included in JSON.
- Camera ground truth is used only for evaluation. No optimizer receives COLMAP poses or calibrated intrinsics. Fixed DSLR calibration is used only to undistort the input photographs.
- Per-scene input filenames, memory search attempts, checkpoint revision, configuration and metrics are in JSON. Full pair errors are saved beside each prediction as `evaluation_errors.npz`.

## Pooled AUC (each camera pair has equal weight)

| Method | AUC@3 | AUC@5 | AUC@10 | AUC@20 | AUC@30 |
|---|---:|---:|---:|---:|---:|
| da3 | 15.704 | 29.081 | 49.826 | 69.161 | 77.874 |
| bae_d_raw | 23.091 | 36.864 | 56.842 | 74.058 | 81.472 |
| bae_d | 16.302 | 29.504 | 50.046 | 69.270 | 77.945 |
| bae_e_raw | 17.350 | 30.438 | 50.655 | 69.594 | 78.166 |
| bae_e | 16.018 | 29.297 | 49.935 | 69.216 | 77.911 |
| bae_de_raw | 23.141 | 36.930 | 56.909 | 74.082 | 81.493 |
| bae_de | 16.336 | 29.524 | 50.054 | 69.273 | 77.947 |
| form_d_raw | 20.612 | 35.905 | 56.239 | 72.393 | 79.509 |
| form_d | 15.174 | 28.584 | 49.488 | 68.970 | 77.736 |
| form_e_raw | 17.948 | 29.856 | 48.954 | 67.958 | 76.822 |
| form_e | 15.800 | 29.092 | 49.800 | 69.135 | 77.849 |
| form_de_raw | 22.634 | 37.453 | 57.157 | 72.748 | 79.694 |
| form_de | 15.789 | 29.063 | 49.744 | 69.093 | 77.814 |

## Per-scene raw AUC@5

| Scene | Views | Initial | bae D | bae E | bae D+E | Reference D | Reference E | Reference D+E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7b6477cb95 | 156 | 57.093 | 72.755 | 60.369 | 72.598 | 67.155 | 67.757 | 71.128 |
| c50d2d1d42 | 156 | 37.463 | 68.326 | 40.504 | 68.990 | 58.720 | 44.592 | 64.075 |
| cc5237fd77 | 156 | 9.706 | 12.648 | 9.813 | 12.782 | 26.319 | 6.387 | 29.771 |
| acd95847c5 | 156 | 14.797 | 27.509 | 16.120 | 27.929 | 12.992 | 14.348 | 13.005 |
| fb5a96b1a2 | 156 | 68.619 | 62.618 | 72.025 | 63.816 | 63.683 | 70.022 | 63.060 |
| a24f64f7fb | 156 | 8.311 | 17.229 | 8.487 | 17.695 | 13.921 | 7.093 | 16.645 |
| 1ada7a0617 | 156 | 42.565 | 71.441 | 45.827 | 71.105 | 67.424 | 56.314 | 73.364 |
| 5eb31827b7 | 151 | 1.312 | 8.641 | 1.325 | 8.625 | 7.656 | 1.420 | 6.415 |
| 3e8bba0176 | 156 | 32.474 | 49.522 | 34.488 | 50.176 | 56.000 | 43.867 | 62.521 |
| 3f15a9266d | 156 | 2.147 | 11.883 | 2.389 | 14.006 | 27.293 | 3.182 | 31.512 |
| 21d970d8de | 156 | 69.334 | 72.132 | 72.039 | 71.865 | 70.053 | 70.958 | 69.506 |
| 5748ce6f01 | 156 | 0.711 | 1.588 | 0.604 | 1.447 | 2.174 | 0.722 | 1.383 |
| c4c04e6d6c | 156 | 43.250 | 56.057 | 45.758 | 56.647 | 52.633 | 49.208 | 50.868 |
| 7831862f02 | 156 | 61.205 | 50.837 | 64.359 | 51.940 | 56.584 | 62.437 | 57.966 |
| bde1e479ad | 156 | 61.275 | 69.112 | 63.187 | 69.041 | 63.754 | 60.088 | 63.008 |
| 38d58a7a31 | 156 | 56.912 | 73.374 | 60.374 | 73.775 | 66.947 | 59.562 | 65.547 |
| 5ee7c22ba0 | 156 | 6.896 | 11.922 | 5.782 | 9.364 | 13.635 | 5.435 | 9.257 |
| f9f95681fd | 156 | 5.862 | 15.436 | 6.322 | 15.735 | 20.244 | 7.974 | 25.565 |
| 3864514494 | 156 | 49.707 | 55.785 | 51.669 | 55.190 | 47.874 | 53.567 | 48.362 |
| 40aec5fffa | 156 | 45.853 | 64.826 | 49.777 | 65.177 | 58.208 | 49.276 | 60.010 |
| 13c3e046d7 | 156 | 2.404 | 3.425 | 2.513 | 3.512 | 7.378 | 1.518 | 8.407 |
| e398684d27 | 156 | 17.957 | 39.795 | 20.351 | 40.624 | 37.164 | 20.657 | 42.481 |
| a8bf42d646 | 156 | 7.583 | 12.919 | 7.558 | 13.023 | 36.078 | 5.731 | 33.392 |
| 45b0dac5e3 | 156 | 55.736 | 60.676 | 55.925 | 61.333 | 46.673 | 49.644 | 47.685 |
| 31a2c91c43 | 156 | 42.217 | 42.777 | 43.309 | 41.574 | 34.402 | 41.635 | 48.792 |
| e7af285f7d | 156 | 7.013 | 21.764 | 7.852 | 22.130 | 28.046 | 6.338 | 27.564 |
| 286b55a2bf | 156 | 30.685 | 43.622 | 30.995 | 41.068 | 33.984 | 30.021 | 37.376 |
| 7bc286c1b6 | 156 | 49.163 | 55.835 | 51.667 | 54.027 | 40.355 | 51.608 | 37.509 |
| f3685d06a9 | 156 | 47.663 | 51.559 | 48.398 | 51.468 | 42.715 | 47.613 | 38.952 |
| b0a08200c9 | 156 | 47.113 | 70.001 | 53.251 | 69.249 | 51.056 | 56.576 | 58.985 |
| 825d228aec | 156 | 3.587 | 8.196 | 3.502 | 8.686 | 11.298 | 3.191 | 12.925 |
| a980334473 | 156 | 60.353 | 60.185 | 63.344 | 61.407 | 52.374 | 59.899 | 55.914 |
| f2dc06b1d2 | 156 | 11.247 | 24.332 | 12.133 | 21.833 | 20.216 | 11.356 | 23.532 |
| 5942004064 | 156 | 8.493 | 8.008 | 8.732 | 8.280 | 13.742 | 3.929 | 14.593 |
| 25f3b7a318 | 156 | 4.168 | 7.937 | 4.469 | 7.725 | 16.488 | 4.137 | 16.899 |
| bcd2436daf | 156 | 50.387 | 52.811 | 51.170 | 52.352 | 46.720 | 45.699 | 49.152 |
| f3d64c30f8 | 156 | 56.710 | 65.208 | 58.894 | 65.939 | 49.335 | 53.401 | 53.031 |
| 0d2ee665be | 159 | 44.502 | 46.376 | 45.695 | 47.455 | 39.179 | 36.637 | 35.573 |
| 3db0a1c8f3 | 156 | 12.483 | 14.779 | 13.407 | 14.688 | 6.524 | 9.548 | 8.390 |
| ac48a9b736 | 156 | 2.972 | 6.693 | 3.057 | 6.784 | 5.527 | 3.212 | 6.416 |
| c5439f4607 | 156 | 62.457 | 70.803 | 64.536 | 71.249 | 64.813 | 64.051 | 67.469 |
| 578511c8a9 | 156 | 1.276 | 2.127 | 1.257 | 2.052 | 4.417 | 0.673 | 3.340 |
| d755b3d9d8 | 156 | 30.200 | 42.844 | 32.488 | 44.699 | 50.491 | 40.720 | 53.143 |
| 99fa5c25e1 | 156 | 61.245 | 63.805 | 62.103 | 62.952 | 61.640 | 59.210 | 57.183 |
| 09c1414f1b | 156 | 8.008 | 24.482 | 8.188 | 24.902 | 23.121 | 9.842 | 25.731 |
| 5f99900f09 | 156 | 2.457 | 9.323 | 2.558 | 9.878 | 28.058 | 1.588 | 28.761 |
| 9071e139d9 | 156 | 4.312 | 10.560 | 4.505 | 11.239 | 25.358 | 4.565 | 30.586 |
| 6115eddb86 | 156 | 3.431 | 1.761 | 3.849 | 1.785 | 8.053 | 3.112 | 9.326 |
| 27dd4da69e | 156 | 30.785 | 25.452 | 31.770 | 24.718 | 25.041 | 22.976 | 21.730 |
| c49a8c6cff | 156 | 9.800 | 19.554 | 10.995 | 19.961 | 30.035 | 7.664 | 33.160 |
