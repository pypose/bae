# Rendering and presentation

These tools consume cached predictions, refined poses, and actual optimization
histories. They do not run the refinement solver. Run commands from the repository
root; see [SHOWCASE.md](../SHOWCASE.md) for rendering dependencies and examples.

| Script | Purpose |
|---|---|
| `showcase_render.py` | Main render CLI: previews, comparisons, and optimization videos |
| `showcase_mesh.py`, `showcase_textured.py` | Optional mesh and textured rendering backends |
| `showcase_stories.py`, `showcase_stories.json` | Camera tours and selected detail views |
| `showcase_video.py` | Video panels with camera trajectories and pose AUC |
| `showcase_manifest.py` | Gallery metadata and camera plots |
| `showcase_gallery.py` | Assemble a portable HTML gallery |

The default renderer uses point splats. The other backends serve different
presentation needs and remain selectable with `--renderer`. Pose metrics live
in `../eval.py`; NumPy rendering and Torch benchmarking share the AUC calculation.

Intermediate renders stay under `outputs/`. Published galleries and videos remain
in `../docs/showcase/` so existing asset links keep working. Regenerate them with
these scripts rather than editing the generated HTML or metrics by hand.
