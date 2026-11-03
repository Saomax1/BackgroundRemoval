# Architecture

Cutout is a personal, fully offline Windows app that removes the background from
one photo or illustration at a time and saves a transparent PNG at exactly the
input resolution. This document is the map for everyone building on the engine.

## Stack (and why)

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Best ecosystem for image models; PyInstaller freezes it into a normal Windows app. |
| Inference | ONNX Runtime GPU, CUDA execution provider | Runs exported models without shipping PyTorch (~2.5 GB saved). CUDA 12 + cuDNN come as pip wheels (`onnxruntime-gpu[cuda,cudnn]`) and are loaded by `onnxruntime.preload_dlls()`, so the user installs nothing but the app. Falls back to CPU if CUDA fails. |
| Image ops | NumPy + OpenCV (headless) + Pillow | OpenCV for fast filters and 16-bit PNG/TIFF, Pillow for EXIF, ICC and palette formats. Headless OpenCV avoids clashing with Qt. |
| UI (next) | PySide6 (Qt 6) | Native-feeling Windows UI, a fast `QGraphicsView` canvas for brushing at full resolution, Qt threads for keeping the UI responsive. |
| Installer (next) | PyInstaller one-folder build + Inno Setup | One `Setup.exe`, Start-menu shortcut, uninstaller. Models are copied into `{app}\models`. Expected size about 2.5 to 3.5 GB, within the 5 GB budget. |

## Models

The engine picks the best *installed* model for the chosen subject, so the app
runs with any subset of these files. Registry: `src/cutout/core/models.py`.

| Subject | Model (priority order) | Net size | Source |
|---|---|---|---|
| Person | **BiRefNet HR-matting** (`birefnet-hr-matting`) | 2048 | Exported from `ZhengPeng7/BiRefNet_HR-matting`. Trained for true alpha mattes at high resolution, so hair and soft edges come out as real transparency rather than a hard cut. |
| Person | BiRefNet general (`birefnet-general`) | 1024 | Official ONNX from the BiRefNet GitHub release. |
| Anime | **ToonOut** (`toonout`) | 1024 | BiRefNet fine-tuned on anime characters; much better on line art, spiky hair and flat-shaded edges than photo-trained models. Exported from its `.pth`. |
| Anime | ISNet anime-seg (`isnet-anime`) | 1024 | Ready-made ONNX from `skytnt/anime-seg`. Small and fast fallback. |

* `scripts/fetch_models.py` downloads the ready-made ONNX files and prints the
  SHA-256 hashes to pin in the registry.
* `scripts/export_birefnet_onnx.py` turns the PyTorch-only checkpoints into ONNX
  (needs torch, run once on the build machine). **Not yet run end-to-end**: the
  sandbox this was written in could not reach Hugging Face or GitHub releases.
  The first build-machine run must validate each export with
  `python -m cutout sample.jpg --model <id>`.
* Choosing person vs anime is a UI toggle for now. Auto-detection (a tiny
  classifier) is a possible later improvement.

## Pipeline

```
load_image ──► Segmenter.predict ──► refine_alpha ──► (user edits alpha) ──► compose_rgba ──► save_png
 imageio.py      segmenter.py          refine.py          editing (next)        refine.py        imageio.py
 full res        net res (1-2K)        full res           full res               full res
```

1. **Load** (`imageio.load_image`): EXIF orientation is applied once, so the
   pixel grid is what the user sees. 8-bit and 16-bit inputs both kept. ICC
   profile kept. Any existing alpha is dropped.
2. **Segment** (`segmenter.OnnxSegmenter`): resize to the net's size (stretch for
   BiRefNet, letterbox for ISNet), normalise, run on GPU, return a coarse matte.
3. **Refine** (`refine.refine_alpha`): bicubic upsample to the exact input size,
   then a guided filter steered by the full-res photo snaps the edge back onto
   real image detail. Applied only in the uncertain band around the edge, so
   solid subject and solid background are untouched.
4. **Compose** (`refine.compose_rgba`): Blur-Fusion foreground estimation removes
   background colour bleeding into semi-transparent hair. Fully opaque pixels are
   bit-exact copies of the input. Output dtype matches input (8 or 16-bit).
5. **Save** (`imageio.save_png`): atomic write, ICC embedded on the 8-bit path.

Measured on a 24 MP photo on a 4-core CPU sandbox (no GPU there): refine about
3 s, compose about 1 s, PNG encode a few seconds for worst-case noise images.
GPU inference time is not measured yet (no GPU or real models in the sandbox).

## Module boundaries

```
src/cutout/
  core/            headless engine. No Qt imports, ever.
    imageio.py     load/save, the "no quality loss" guarantees
    models.py      ModelSpec registry, model selection, models dir
    runtime.py     ONNX Runtime sessions, CUDA/CPU selection
    segmenter.py   Segmenter protocol + OnnxSegmenter (pre/post-processing)
    refine.py      upsample, guided filter, foreground estimation, compose
    engine.py      CutoutEngine: the public API the UI and CLI use
  cli.py           command-line front end (dev, benchmarks, installer smoke test)
  editing/         (next) touch-up tools, pure NumPy on the alpha matte
  ui/              PySide6 editor window (see "Editor window" below)
packaging/         (next) PyInstaller spec, Inno Setup script
scripts/           build-time model download and export
```

### Public engine API

```python
from cutout import CutoutEngine, Subject

engine = CutoutEngine(device="auto")      # models from %CUTOUT_MODELS_DIR% or <exe dir>\models
engine.warmup(Subject.PERSON)              # optional: load + first run at app start
result = engine.remove_background(path, Subject.ANIME)
result.alpha        # float32 HxW in [0, 1], exact input size: the thing editors change
result.image        # LoadedImage (rgb uint8/uint16, icc_profile)
result.save_png(out)
# after edits:
rgba = compose_rgba(result.image, edited_alpha)
save_png(out, rgba, result.image.icc_profile)
```

### Contracts for the follow-on work

* **Editing (brush and click-to-select).** Work only on `result.alpha` (float32,
  full resolution). Brush = paint 0 or 1 with a soft round stamp; keep an undo
  stack of changed tiles, not whole copies (a 24 MP matte is 96 MB). Click-to-
  select should be a separate prompt model (for example SAM 2 or EfficientSAM via
  ONNX) implementing a sibling of the `Segmenter` protocol that takes click
  points, added to `models.py` like any other model. Its mask should be refined
  with `refine.guided_refine` before merging, so clicked regions get the same
  edge quality. Never resize the matte.
* **UI.** Run every engine call in a worker thread (`QThreadPool`), never on the
  UI thread. Preview by drawing the original image with `alpha` as a mask (a
  checkerboard behind it); call `compose_rgba` only for export, since
  decontamination on a large photo takes about a second. Show
  `result.device` so a CPU fallback is visible.
* **Installer.** Freeze with PyInstaller (one-folder), include the `nvidia/*`
  DLL folders from site-packages, put models in `{app}\models`. The smoke test
  after install is `cutout.exe --list-models` plus one run on a bundled sample.

## Editor window

`python -m cutout.ui [image]` (or `cutout-gui`, or `from cutout.ui import main`
in the packaged `Cutout.exe`). Files in `src/cutout/ui`:

| File | Role |
|---|---|
| `app.py` | `main(argv)`, dark Fusion theme, Windows taskbar id. |
| `main_window.py` | Open / drag-drop, Person/Anime switch, auto removal on open, Save PNG, menus, status bar (size, model, GPU/CPU, timing, pixel opacity, zoom). |
| `document.py` | `CutoutDocument`: image, `alpha`, `QUndoStack`. All matte edits go through it. |
| `canvas.py` | `CutoutCanvas`: `QGraphicsView` with screen-space checkerboard (or white/black), wheel zoom at cursor, drag/Space/middle-button pan, "show original" compare. |
| `tools.py` | `CanvasTool` base + `ToolContext`: the hook the touch-up tools plug into. |
| `workers.py` | `Task` on `QThreadPool`; results come back as signals on the UI thread. |

Behaviour worth knowing:

* Opening an image starts the automatic removal right away. Switching the
  subject or "Redo Auto" re-runs it as one undoable step.
* The preview is an 8-bit premultiplied copy of the image with the matte as
  alpha. Only changed rectangles are recomputed, so brush strokes stay live on
  large photos. Edge colour cleanup runs only on export.
* Save composes at full resolution and bit depth on a worker thread, writes
  `<name>_cutout.png` by default, and refuses to overwrite the source image.
* While a task runs a busy overlay covers the canvas and conflicting actions
  are disabled, so the matte cannot change under a running save.

### Adding a touch-up tool

```python
class BrushTool(CanvasTool):
    id, label = "brush", "Brush"
    def mouse_press(self, pos, event):      # pos: QPointF in image pixels
        ...edit self.ctx.document.alpha in place...
        self.ctx.document.notify_alpha_changed(dirty_qrect)   # live redraw
        return True
    def mouse_release(self, pos, event):
        self.ctx.document.push_patch(rect, after, before=before, text="Brush")  # one undo step
        return True

window.register_tool(BrushTool(), shortcut="B")
```

Long work (a click-to-select model) goes through
`ctx.run_in_background(fn, *args, on_done=..., on_error=...)`. Tools that set
`needs_alpha = True` (the default) are disabled until a cutout exists. Ctrl or
Alt + wheel is offered to the tool first (for brush size); plain wheel zooms.

## Development

```
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[gpu,ui,dev]"       # or ".[cpu,ui,dev]" without an NVIDIA GPU
python scripts/fetch_models.py
python -m cutout photo.jpg --subject person
python -m cutout.ui                   # the editor window
pytest
```

Tests run without models or a GPU: they use a fake segmenter and tiny
generated ONNX graphs, so CI (Windows and Linux, CPU) exercises the real
ONNX Runtime plumbing.
