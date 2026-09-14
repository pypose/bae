"""Audit a complete split using DA3's evaluator and report paired method gaps."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'da3'))
from eval_pose_auc_scannetpp import build_pairs, pair_errors_idx, error_auc
from eval import pair_errors as native_pair_errors
from dslr import list_dslr_frames

THRESHOLDS = (3, 5, 10, 20, 30)
METHODS = ['bae_d', 'bae_e', 'bae_de', 'form_d', 'form_e', 'form_de']
KEYS = ['da3'] + [k for m in METHODS for k in (m+'_raw', m)]


def report(root, dest, device):
    manifest = json.loads((root/'manifest.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    for filename, expected in provenance['source_sha256'].items():
        actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Validation source changed during run: {filename}')
    scenes = manifest['config']['scenes']
    complete = [s for s in scenes if manifest['status'][s]['state'] == 'complete']
    dest.mkdir(parents=True, exist_ok=True)
    pooled = {k: [] for k in KEYS}
    records = []
    torch.set_num_threads(8)
    for scene in complete:
        folder = root/scene
        source = json.loads((folder/'metrics.json').read_text())
        inp = json.loads((folder/'input.json').read_text())
        log = (folder/'run.log').read_text()
        assert inp['model'] == manifest['config']['model']
        assert inp['checkpoint_revision'] == manifest['config']['revision']
        assert inp['process_res'] == 504 and inp['input_pose_or_intrinsics_conditioning'] is False
        if records:
            assert source['config'] == records[0]['config'], f'{scene}: inconsistent optimization configuration'
        with (folder/'prediction.npz').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == source['prediction_sha256'], f'{scene}: prediction changed'
        with np.load(folder/'prediction.npz') as pred, np.load(folder/'refined.npz') as refined:
            gt = torch.as_tensor(pred['gt'], dtype=torch.float64, device=device)
            names = pred['names'].tolist()
            assert names == inp['selected_frame_names']
            assert len(names) == len(set(names)) == source['views'] == inp['selected_views']
            registered = list_dslr_frames(scene, n_views=100000,
                dataset_root=str(Path(manifest['config']['split']).parents[1]))
            expected = [registered[i] for i in np.linspace(0, len(registered)-1, len(names)).round().astype(int)]
            assert names == expected, f'{scene}: inputs do not match the scene-wide sampling protocol'
            assert inp['registered_views'] == len(registered)
            assert source['config']['ambient_grad'].lower() in ('1', 'true', 'yes', 'on')
            i, j = build_pairs(len(gt)); i, j = i.to(device), j.to(device)
            record = dict(scene=scene, views=len(gt), pairs=len(i), input=inp,
                          config=source['config'], prediction_sha256=source['prediction_sha256'], variants={})
            record['diagnostics'] = dict(linear_solver_failures=log.count('Linear solver failed.'),
                                         tracebacks=log.count('Traceback (most recent call last)'),
                                         scene_wide_filenames_verified_against_colmap=True)
            errors_by_variant = {}
            for key in KEYS:
                w = torch.as_tensor(refined[key], dtype=torch.float64, device=device)
                assert bool(torch.isfinite(w).all()), (scene, key)
                # DA3's explicit-pair evaluator is the primary metric for this report.
                errors = pair_errors_idx(w, gt, i, j)
                cli_errors = pair_errors_idx(w.float(), gt.float(), i, j)
                old_errors = native_pair_errors(w, gt)
                auc = error_auc(errors, THRESHOLDS)
                cli_auc = error_auc(cli_errors, THRESHOLDS)
                delta = {f'auc@{t}': auc[f'auc@{t}']-source['variants'][key][f'auc@{t}'] for t in THRESHOLDS}
                record['variants'][key] = dict(auc=auc, native_metric_delta=delta,
                    original_cli_float32_auc=cli_auc,
                    float64_minus_float32_auc={f'auc@{t}': auc[f'auc@{t}']-cli_auc[f'auc@{t}'] for t in THRESHOLDS},
                    max_pair_error_difference_deg=float(np.max(np.abs(errors-old_errors))),
                    accepted=source['variants'][key].get('accepted'),
                    seconds=source['variants'][key].get('seconds'))
                errors_by_variant[key] = errors
                pooled[key].append(errors)
            # Persist the actual pair errors so pooled scores are independently reproducible.
            np.savez_compressed(folder/'evaluation_errors.npz', i=i.cpu().numpy(), j=j.cpu().numpy(),
                                names=np.array(names), **errors_by_variant)
            records.append(record)
        print(f'Audited {len(records)}/{len(complete)}: {scene}', flush=True)
    summary = {}
    for key in KEYS:
        if not records:
            continue
        summary[key] = dict(
            macro_auc={f'auc@{t}': float(np.mean([r['variants'][key]['auc'][f'auc@{t}'] for r in records])) for t in THRESHOLDS},
            pooled_auc=error_auc(np.concatenate(pooled[key]), THRESHOLDS),
            improved=sum(r['variants'][key]['auc']['auc@5'] > r['variants']['da3']['auc']['auc@5']+1e-8 for r in records),
            regressed=sum(r['variants'][key]['auc']['auc@5'] < r['variants']['da3']['auc']['auc@5']-1e-8 for r in records),
            accepted=sum(r['variants'][key]['accepted'] is True for r in records))
    comparisons = {}
    guard_audit = {}
    for method in METHODS:
        guard_audit[method] = dict(
            accepted=sum(r['variants'][method]['accepted'] is True for r in records),
            rejected_improvements=sum(r['variants'][method]['accepted'] is False and
                r['variants'][method+'_raw']['auc']['auc@5'] > r['variants']['da3']['auc']['auc@5']+1e-8 for r in records),
            rejected_regressions=sum(r['variants'][method]['accepted'] is False and
                r['variants'][method+'_raw']['auc']['auc@5'] < r['variants']['da3']['auc']['auc@5']-1e-8 for r in records))
    for stage in ('d', 'e', 'de'):
        a, b = 'bae_'+stage+'_raw', 'form_'+stage+'_raw'
        diff = np.array([r['variants'][a]['auc']['auc@5']-r['variants'][b]['auc']['auc@5'] for r in records])
        if len(diff):
            comparisons[stage] = dict(mean_auc5_gap=float(diff.mean()),
                median_auc5_gap=float(np.median(diff)), max_abs_auc5_gap=float(np.abs(diff).max()),
                within_1_percentage_point=int((np.abs(diff) <= .01).sum()),
                bae_higher=int((diff > 1e-8).sum()), reference_higher=int((diff < -1e-8).sum()))
    max_metric_delta = max((abs(v) for r in records for m in r['variants'].values()
                           for v in m['native_metric_delta'].values()), default=0)
    max_precision_delta = max((abs(v) for r in records for m in r['variants'].values()
                              for v in m['float64_minus_float32_auc'].values()), default=0)
    historical_path = HERE.parents[1]/'da3/da3_pose_auc_nvs_test_GIANT11.json'
    historical = json.loads(historical_path.read_text())
    historical_comparison = dict(path=str(historical_path), split=historical['split'],
        macro_auc=historical['macro_auc'], shared_scenes=sorted(set(scenes)&set(historical['per_scene'])),
        directly_comparable=False,
        reason='Different scene split and frame counts; historical nvs_test has up to 627 views, this run uses mounted nvs_sem_val with adaptive GPU limits.')
    serial_audit = json.loads((root/'serial_control_audit.json').read_text())
    edge_audit = json.loads((root/'parallel_edge_audit.json').read_text())
    max_serial_delta = max(abs(v) for scene in serial_audit.values() for method in scene.values()
                           for v in method['auc_difference'].values())
    result = dict(complete=len(complete)==len(scenes), expected_scenes=len(scenes), completed_scenes=len(complete),
        manifest=manifest, provenance=provenance, n_pairs=sum(r['pairs'] for r in records), summary=summary,
        paired_comparisons=comparisons, guard_audit=guard_audit, max_native_vs_da3_auc_difference=max_metric_delta,
        max_float64_vs_original_cli_float32_auc_difference=max_precision_delta,
        historical_comparison=historical_comparison, serial_control_audit=serial_audit,
        parallel_edge_audit=edge_audit, per_scene=records)
    (dest/'results.json').write_text(json.dumps(result, indent=2))
    with (dest/'results.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['scene', 'views', 'method', 'accepted', *[f'auc@{t}' for t in THRESHOLDS]])
        for r in records:
            for k, v in r['variants'].items():
                writer.writerow([r['scene'], r['views'], k, v['accepted'], *[v['auc'][f'auc@{t}'] for t in THRESHOLDS]])
    lines = ['# Full DSLR validation', '',
        f'Coverage: **{len(complete)}/{len(scenes)} scenes** from `{Path(manifest["config"]["split"]).name}`; {result["n_pairs"]:,} unordered camera pairs. '
        f'Checkpoint: `{manifest["config"]["model"]}` at `{manifest["config"]["revision"]}`.', '',
        'All methods receive the same globally attended, undistorted DSLR prediction for each scene. '
        '`BAE_USE_PYPOSE_AMBIENT_GRAD=1` is enabled. No scene is selected using its AUC. '
        'Raw and guarded outcomes are separate. All table AUCs are percentages; JSON stores fractions.', '',
        '## Macro AUC (each scene has equal weight)', '',
        '| Method | AUC@3 | AUC@5 | AUC@10 | AUC@20 | AUC@30 | Improved / regressed at 5° |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k, v in summary.items():
        lines.append('| '+k+' | '+' | '.join(f'{v["macro_auc"][f"auc@{t}"]*100:.3f}' for t in THRESHOLDS)+f' | {v["improved"]} / {v["regressed"]} |')
    lines += ['', '## Does native refinement match the reference?', '',
        '| Raw stage | bae − reference AUC@5 (pp) | Largest scene gap (pp) | Within 1 pp |',
        '|---|---:|---:|---:|']
    for k, v in comparisons.items():
        lines.append(f'| {k.upper()} | {v["mean_auc5_gap"]*100:+.3f} | {v["max_abs_auc5_gap"]*100:.3f} | {v["within_1_percentage_point"]}/{len(complete)} |')
    lines += ['', 'Native D minimizes a joint photometric objective; Open3D D performs dense pairwise odometry followed by PGO. '
        'Native E rebuilds a voxel-averaged structure; reference E uses TSDF fusion and rigid color-map optimization. '
        'These are different formulations, so numerical equality is not guaranteed by correct gradients.', '',
        '## Guard decisions', '',
        'Improvement and regression below refer to ground-truth AUC@5, which the guard does not receive.', '',
        '| Method | Accepted | Rejected AUC improvements | Rejected AUC regressions |',
        '|---|---:|---:|---:|']
    for k, v in guard_audit.items():
        lines.append(f'| {k} | {v["accepted"]}/{len(complete)} | {v["rejected_improvements"]} | {v["rejected_regressions"]} |')
    lines += ['', '## Metric and historical-number audit', '',
        f'- Every saved result was rescored with `da3/eval_pose_auc_scannetpp.py::pair_errors_idx` and its trapezoidal `error_auc`. '
        f'Maximum difference from the example evaluator across all scenes, methods and thresholds: **{max_metric_delta:.9g} AUC**.',
        f'- Main tables score all methods in float64. The original CLI uses float32; that path was independently run for every result. '
        f'Maximum float64/float32 AUC difference: **{max_precision_delta*100:.6f} percentage points**. Both sets of per-scene scores are in JSON.',
        '- The local historical JSON covers `nvs_test.txt`. The mounted dataset is `nvs_sem_val.txt`: '
        f'**{len(historical_comparison["shared_scenes"])} shared scenes**. Its baseline macro AUC@5 is '
        f'{historical["macro_auc"]["auc@5"]*100:.3f}%, but it is not a reproduction target for this different split and frame budget.',
        '- The reference arms here call the actual DA3 Form D/E functions on the same cached arrays as bae. '
        'They use the showcase protocol: shared self-calibration, graph baseline filter 0.1 / rotation 1°, '
        'D texture gate and internal guard disabled, then a common 512-pair wide guard. '
        'D+E composes raw D then E and guards the whole proposal. This differs from the historical CLI defaults.',
        f'- Parallel odometry check: {edge_audit["pairs"]} real DSLR edges gave identical success flags, transforms and information matrices '
        f'with one versus eight edge workers at one OpenMP thread. Against the original eight-OpenMP-thread serial controls on two scenes, '
        f'the largest D/E/D+E AUC difference over all five thresholds was {max_serial_delta*100:.5f} percentage points. '
        'Changing floating-point reduction order is not bitwise identical; the measured controls are included in JSON.',
        '- Camera ground truth is used only for evaluation. No optimizer receives COLMAP poses or calibrated intrinsics. '
        'Fixed DSLR calibration is used only to undistort the input photographs.',
        '- Per-scene input filenames, memory search attempts, checkpoint revision, configuration and metrics are in JSON. '
        'Full pair errors are saved beside each prediction as `evaluation_errors.npz`.', '',
        '## Pooled AUC (each camera pair has equal weight)', '',
        '| Method | AUC@3 | AUC@5 | AUC@10 | AUC@20 | AUC@30 |',
        '|---|---:|---:|---:|---:|---:|']
    for k, v in summary.items():
        lines.append('| '+k+' | '+' | '.join(f'{v["pooled_auc"][f"auc@{t}"]*100:.3f}' for t in THRESHOLDS)+' |')
    lines += ['', '## Per-scene raw AUC@5', '',
        '| Scene | Views | Initial | bae D | bae E | bae D+E | Reference D | Reference E | Reference D+E |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        keys = ['da3']+[k+'_raw' for k in METHODS]
        lines.append(f'| {r["scene"]} | {r["views"]} | '+' | '.join(f'{r["variants"][k]["auc"]["auc@5"]*100:.3f}' for k in keys)+' |')
    (dest/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: result[k] for k in ('complete', 'completed_scenes', 'n_pairs', 'paired_comparisons', 'max_native_vs_da3_auc_difference')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/rgbd_validation_giant11'))
    p.add_argument('--dest', type=Path, default=HERE/'docs/validation')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    report(a.root, a.dest, a.device)
