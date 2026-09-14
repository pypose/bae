"""Render authentic multiview tours, detail comparisons and solver portraits.

Each 11-second chapter has a four-second synchronized 3D camera move, four
seconds of saved optimizer states, and a three-second second-view detail.
Only the presentation camera interpolates; optimizer poses never do.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from showcase_render import BG, font, homogeneous, look_at
from showcase_textured import TexturedRenderer

SIZE = (924, 693)
FPS = 24
MINT, ORANGE, MUTED = '#71efce', '#ffbd8c', '#9aaabc'


def compose(left, right, story, coverage, auc, phase, caption, progress=0):
    out = Image.new('RGB', (1920, 1080), BG)
    d = ImageDraw.Draw(out)
    d.text((28, 17), 'bae', font=font(40, True), fill=MINT)
    d.text((130, 30), 'ROOMS, REFINED', font=font(18, True), fill='white')
    d.text((28, 78), story['headline'], font=font(46, True), fill='white')
    d.text((28, 140), 'DA3 / INITIAL', font=font(18, True), fill=ORANGE)
    d.text((972, 140), phase, font=font(18, True), fill=MINT)
    out.paste(left, (24, 178))
    out.paste(right, (972, 178))
    d.text((28, 901), caption, font=font(25), fill='white')
    d.text((28, 950), coverage, font=font(18), fill=MUTED)
    d.text((1400, 22), f'Pose AUC@5  {auc}', font=font(24, True), fill=MINT)
    d.text((28, 990), 'Raw optimizer output · guard rejects this proposal · fixed DA3 depth', font=font(17), fill=MUTED)
    d.text((28, 1021), 'Undistorted DSLR · identical rendering on both sides · no synthetic pose noise', font=font(17), fill=MUTED)
    d.rectangle((24, 1060, 24 + round(1872 * progress), 1064), fill=MINT)
    return out


def render_scene(root, scene, story, dataset_root):
    out = root / scene
    dest = out / 'renders'
    data = np.load(out / 'prediction.npz')
    ref = dict(np.load(out / 'refined.npz'))
    manifest = json.loads((dest / 'gallery.json').read_text())
    metrics = json.loads((out / 'metrics.json').read_text())
    assert metrics['config']['ambient_grad'] == '1'
    assert not metrics['variants']['bae_de']['accepted'], 'Update chapter guard label for accepted runs'
    initial, final = homogeneous(data['w2c']), homogeneous(ref['bae_de_raw'])
    c2w = np.linalg.inv(initial)
    up = -c2w[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    renderer = TexturedRenderer(data, ref['K'], out / 'textures', SIZE, dataset_root)
    view_id = story['view']
    center, forward = c2w[view_id, :3, 3], c2w[view_id, :3, 2]
    view = look_at(center, center + forward, up)
    coverage = f'{story["title"]} / {scene}   ·   {manifest["views"]} views across {manifest["registered"]} registered photographs'
    auc = manifest['auc']
    pair = [renderer.render(w, view, exclude=view_id) for w in (initial, final)]

    # Individual fixed-camera images support direct visual method comparison.
    formulations = []
    for name, label in [('bae_d', 'bae D'), ('bae_e', 'bae E'), ('bae_de', 'bae D+E'),
                        ('form_d', 'Open3D D'), ('form_e', 'Open3D E'), ('form_de', 'Open3D D+E')]:
        raw = name + '_raw'
        filename = f'method_{name}.webp'
        renderer.render(ref[raw], view, exclude=view_id).save(dest / filename, quality=95)
        formulations.append(dict(label=label, image=f'renders/{filename}',
                                 auc=round(metrics['variants'][raw]['auc@5'] * 100, 2)))

    detail_id = story['detail_view']
    dc = c2w[detail_id]
    detail_view = look_at(dc[:3, 3], dc[:3, 3] + dc[:3, 2], up)
    detail = [renderer.render(w, detail_view, fov=story['detail_fov'], exclude=detail_id)
              for w in (initial, final)]
    for label, im in zip(('before', 'after'), detail):
        im.save(dest / f'detail_{label}.webp', quality=97)
    compose(*detail, story, coverage, auc, 'BAE D+E / RAW RESULT', story['detail_title'], 1).save(dest / 'detail_comparison.jpg', quality=97)

    writer = imageio.get_writer(dest / 'tour.mp4', fps=FPS, codec='libx264', macro_block_size=2,
                                ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p'])
    actual_paths = manifest['frames']
    total = 11 * FPS
    camera_path, state_indices = [], []
    try:
        # Move a small fraction of the scene's median depth sideways, around
        # one fixed target. The exact same camera is used for both maps.
        depth = data['depth'][view_id]
        radius = float(np.median(depth[np.isfinite(depth) & (depth > 0)]))
        target = center + forward * radius
        right = np.linalg.inv(view)[:3, 0]
        for k in range(4 * FPS):
            t = k / (4 * FPS - 1)
            offset = .035 * radius * np.sin(2 * np.pi * t)
            moving = look_at(center + right * offset, target, up)
            camera_path.append(moving.tolist())
            images = [renderer.render(w, moving, exclude=view_id) for w in (initial, final)]
            writer.append_data(np.asarray(compose(*images, story, coverage, auc,
                'BAE D+E / RAW RESULT', '01 / Explore the geometry · synchronized 3D camera move', k / total)))
        for k in range(4 * FPS):
            j = round(k / (4 * FPS - 1) * (len(actual_paths) - 1))
            state_indices.append(j)
            current = Image.open(out / actual_paths[j]).convert('RGB').resize(SIZE, Image.Resampling.LANCZOS)
            phase = 'BAE / INITIAL' if j == 0 else 'BAE D+E / RAW RESULT' if j == len(actual_paths)-1 else f'BAE / SAVED STATE {j}'
            writer.append_data(np.asarray(compose(pair[0], current, story, coverage, auc,
                phase, '02 / Follow the optimization · actual saved solver states', (4 * FPS + k) / total)))
        frame = compose(*detail, story, coverage, auc, 'BAE D+E / RAW RESULT',
                        '03 / ' + story['detail_title'], 1)
        for _ in range(3 * FPS):
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    compose(*pair, story, coverage, auc, 'BAE D+E / RAW RESULT', story['description'], 1).save(dest / 'tour_poster.jpg', quality=97)
    manifest.update(title=story['title'], description=story['description'],
                    run_label='GIANT 1.1 · ambient gradients',
                    tour='renders/tour.mp4', tour_poster='renders/tour_poster.jpg',
                    detail=dict(title=story['detail_title'], description=story['detail_description'],
                                before='renders/detail_before.webp', after='renders/detail_after.webp'),
                    formulations=formulations)
    (dest / 'gallery.json').write_text(json.dumps(manifest))
    provenance = dict(scene=scene, story=story, width=1920, height=1080, fps=FPS, frames=total,
                      variant='bae_de_raw', ambient_grad='1', camera_path=camera_path,
                      state_indices=state_indices, optimization_pose_interpolation=False,
                      excluded_source_views=[view_id, detail_id], depth_unchanged=True,
                      initial_and_final_share_render_camera=True,
                      prediction_sha256=metrics['prediction_sha256'])
    (dest / 'story.json').write_text(json.dumps(provenance, indent=2))
    print(dest / 'tour.mp4', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/rgbd_validation_giant11'))
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--dataset-root', default='/data/zitong/scannetpp_val')
    args = parser.parse_args()
    stories = json.loads(Path(__file__).with_suffix('.json').read_text())
    for scene in args.scenes or stories:
        render_scene(args.out, scene, stories[scene], args.dataset_root)
