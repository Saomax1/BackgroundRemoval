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
| UI | PySide6 (Qt 6) | Native-feeling Windows UI, a fast `QGraphicsView` canvas for brushing at full resolution, Qt threads for keeping the UI responsive. |
| Installer | PyInstaller one-folder build + Inno Setup 6.5.2+ | A small `Setup.exe` on GitHub Releases, desktop and Start-menu shortcuts, uninstaller. The GPU runtime and models are separate *packs* the app downloads on first start. Built, smoke-tested and published by GitHub Actions; see `packaging/README.md`. |
| Updates | `cutout.update` + a PowerShell helper | Help > Check for Updates downloads only the packs whose content hash changed (usually just the app, tens of MB), checks every byte, then swaps files after the app exits with automatic rollback. Offline unless the user asks. |

## Models

The engine picks the best *installed* model for the chosen subject, so the app
runs with any subset of these files. Registry: `src/cutout/core/models.py`.

| Subject | Model (priority order) | Net size | Source |
|---|---|---|---|
| Person | **BiRefNet HR-matting** (`birefnet-hr-matting`) | 2048 | Exported from `ZhengPeng7/BiRefNet_HR-matting`. Trained for true alpha mattes at high resolution, so hair and soft edges come out as real transparency rather than a hard cut. |
| Person | BiRefNet general (`birefnet-general`) | 1024 | Official ONNX from the BiRefNet GitHub release. |
| Anime | **ToonOut** (`toonout`) | 1024 | BiRefNet fine-tuned on anime characters; much better on line art, spiky hair and flat-shaded edges than photo-trained models. Exported from its `.pth`. |
| Anime | ISNet anime-seg (`isnet-anime`) | 1024 | Ready-made ONNX from `skytnt/anime-seg`. Small and fast fallback. |

Click-to-select uses a separate *promptable* model, picked the same way (best installed first):

| Model (priority order) | Files | Approx. size | Source |
|---|---|---|---|
| **SAM 2.1 Hiera-L** (`sam2.1-hiera-large`) | `.encoder.onnx` + `.decoder.onnx` | ~900 MB | `facebook/sam2.1-hiera-large` |
| SAM 2.1 Hiera-B+ (`sam2.1-hiera-base-plus`) | same | ~330 MB | `facebook/sam2.1-hiera-base-plus` |
| SAM 2.1 Hiera-S (`sam2.1-hiera-small`) | same | ~190 MB | `facebook/sam2.1-hiera-small` |

The encoder runs once per image, the decoder on every click. Click-to-select runs
on the **CPU** by default (`CutoutEngine(select_device=...)`, or the
`CUTOUT_SELECT_DEVICE` environment variable: `cpu`, `auto`, `cuda`), so it never
competes with the automatic-removal model for graphics memory (see "CPU and GPU in tandem").
`scripts/export_sam2_onnx.py <id> --check photo.jpg` builds the pair and compares
it with the PyTorch predictor (**not yet run end-to-end**, same sandbox limit).
If the installer is tight on space, ship B+ only; it is close to L for this use.

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
 imageio.py      segmenter.py          refine.py          editing/              refine.py        imageio.py
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
5. **Save** (`imageio.save_png`): lossless PNG encoded on all cores (`png.py`),
   atomic write, ICC profile embedded for 8 and 16-bit.

### CPU and GPU in tandem

The GPU runs only the network (at 1-2K). Everything at the photo's own resolution
is CPU work, and it is arranged so the CPU is never idle while the GPU runs and
never uses just one core:

* **Overlap.** `remove_background` starts `refine.prepare_image` on a CPU thread
  before the model runs: the guide image and its half of the guided filter, and the
  photo on Blur-Fusion's coarse grids. Neither depends on the matte, so they are
  ready when the GPU finishes.
* **All cores.** Every per-pixel full-resolution step is cut into row strips and run
  on a thread pool (`parallel.map_rows`; NumPy, OpenCV and zlib release the GIL).
  Only per-pixel steps are split, so the output is bit-for-bit what the
  single-threaded code gives (`tests/test_parallel.py` checks this). Neighbourhood
  filters stay whole-image OpenCV calls, which are already multi-threaded.
* **PNG on all cores.** `png.encode_png` filters and deflates strips in parallel the
  way `pigz` does (each strip primed with the previous 32 KB, joined with sync
  flushes, Adler-32 combined). Pixels are exact; files are about the size Pillow made.
* **Save prepared ahead.** As soon as a cutout is shown, the window composes and
  encodes the PNG in the background. Save then only writes the bytes. Any edit
  (tracked by `CutoutDocument.revision`) makes that copy stale and Save composes afresh.
* **Not done on purpose:** splitting one network between CPU and GPU. Copying
  activations over PCIe costs more than the CPU adds; the GPU is 20-50x faster at it.

`CUTOUT_CPU_THREADS` limits the thread count (`1` = single-threaded). The debug
report has a "Timings" section with every step of the last cutouts and saves.

Measured on a 24 MP photo on a 4-core CPU sandbox (no GPU there), CPU work after
the model: before 7.1 s (refine 0.6, compose 1.7, PNG 4.7), now 2.7 s (refine
0.35, compose 0.5, PNG 1.8) with 0.25 s more done while the model runs, and
the compose + PNG part is done before the user presses Save. Peak RAM went down
slightly. More cores scale these further.
GPU inference time is not measured yet (no GPU or real models in the sandbox).

## Module boundaries

```
src/cutout/
  core/            headless engine. No Qt imports, ever.
    imageio.py     load/save, the "no quality loss" guarantees
    png.py         lossless PNG encoder that uses every core
    parallel.py    row-strip thread pool for full-resolution per-pixel work
    models.py      ModelSpec registry, model selection, models dir
    runtime.py     ONNX Runtime sessions, CUDA/CPU selection
    segmenter.py   Segmenter protocol + OnnxSegmenter (pre/post-processing)
    refine.py      upsample, guided filter, foreground estimation, compose
    engine.py      CutoutEngine: the public API the UI and CLI use
    prompt_segmenter.py  PromptSegmenter protocol + OnnxSamSegmenter (click-to-select model)
    debug.py       log files, debug mode, crash detection, memory snapshots, the debug report
  cli.py           command-line front end (dev, benchmarks, installer smoke test)
  editing/         touch-up tools, pure NumPy/OpenCV on the alpha matte, no Qt
    matte.py       MatteEditor: brush strokes, merging masks, tile-based undo (Patch, EditHistory)
    brush.py       Brush settings, soft round stamp, dab spacing
    select.py      ClickSelector: clicks -> full-res, edge-refined Selection
    rect.py        Rect: the dirty area every edit returns, for partial repaints
  ui/              PySide6 editor window (see "Editor window" below)
    update_ui.py   Help > Check for Updates, the update and finish-setup dialog
  update/          updater, no Qt: packs.py (how the install splits), manifest.py
                   (cutout-release.json), client.py (check, download, apply), apply_update.ps1
packaging/         PyInstaller spec, Inno Setup script, model staging, release packs (packaging/README.md)
scripts/           build-time model download, export and checks
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
result.notice       # None, or a message to show (e.g. "ran on the CPU because the GPU was full")
engine.release_models()  # unload everything, e.g. before a memory-hungry step
```

### GPU and RAM use

- One automatic-removal model is loaded at a time. Switching subject (or model)
  drops the old session before the new one is created; the 2048 px person model
  and ToonOut together don't fit on a typical 8-12 GB card.
- Sessions use `arena_extend_strategy=kSameAsRequested` and every run asks ORT
  to shrink the GPU arena afterwards (`memory.enable_memory_arena_shrinkage`),
  so activations are not parked in VRAM between runs. The CPU arena is off for
  the same reason in RAM.
- cuDNN uses `HEURISTIC` algorithm search with a capped workspace. `EXHAUSTIVE`
  tries algorithms that want GB-sized scratch buffers and was enough on its own
  to run out of VRAM.
- On an out-of-memory error the engine unloads every model (including
  click-to-select), retries on the GPU, and with `device="auto"` then runs that
  one image on the CPU and sets `result.notice`. The CPU session isn't cached,
  so the next image tries the GPU again. Otherwise it raises `GpuOutOfMemory`
  with a readable message. Never recover inside the `except` block: the live
  traceback keeps the failed session (and its VRAM) alive.
- Output resolution is unaffected by all of this: models run at their fixed size.

### Contracts for the follow-on work

* **Editing (brush and click-to-select).** Work only on `result.alpha` (float32,
  full resolution). Brush = paint 0 or 1 with a soft round stamp; keep an undo
  stack of changed tiles, not whole copies (a 24 MP matte is 96 MB). Click-to-
  select should be a separate prompt model (for example SAM 2 or EfficientSAM via
  ONNX) implementing a sibling of the `Segmenter` protocol that takes click
  points, added to `models.py` like any other model. Its mask should be refined
  with `refine.guided_refine` before merging, so clicked regions get the same
  edge quality. Never resize the matte.
* **Editing API (implemented).**

  ```python
  from cutout.editing import Brush, BrushMode, MaskOp, MatteEditor

  editor = MatteEditor(result.alpha, result.image)   # edits result.alpha in place
  dirty = editor.begin_stroke(x, y, Brush(BrushMode.ERASE, radius=30, hardness=0.6))
  dirty = editor.continue_stroke(x, y, pressure)      # Rect to repaint
  patch = editor.end_stroke()                         # one undo step (or None)
  editor.undo(); editor.redo()

  selector = engine.click_selector(result)            # ModelNotInstalled if no SAM model
  selector.prepare()                                  # worker thread, once per image
  sel = selector.add_click(x, y, positive=True)       # worker thread; sel.preview for an overlay
  patch = editor.apply_mask(sel, MaskOp.ADD)          # or MaskOp.SUBTRACT
  ```

  * Coordinates are image pixels in Qt scene convention (pixel `i` spans
    `[i, i+1)`), so a `QPointF` from the canvas goes straight in.
  * Brush modes: KEEP (alpha to 1), ERASE (alpha to 0), REFINE (re-fit the edge
    to the photo with the guided filter, for hair). Within one stroke, dabs
    combine with `max` into a coverage buffer, so passing twice over the same
    spot never hardens soft edges and opacity is a true cap.
  * Undo stores the 256x256 tiles an edit touched, swapped in place on undo and
    redo. `EditHistory` is capped at 1 GB / 200 steps. A UI with its own
    `QUndoStack` passes `history=None` and uses `patch.rect`, `patch.before`,
    `patch.after`.
  * Brush calls are cheap enough for the UI thread (about 1 ms per mouse move,
    3 ms for REFINE, at 24 MP on a 4-core CPU). `prepare()` and `add_click()`
    run models and must go to a worker; `add_click` spent 0.55 s on a selection
    covering a quarter of a 24 MP photo, mostly the full-res guided filter.
  * A click mask is rebuilt at full resolution only over its bounding area:
    the 256x256 logits are sampled at every full-res pixel centre, thresholded,
    given a narrow soft band and snapped to image edges with `guided_refine`.
    The previous click's logits are fed back to the decoder, as SAM intends.
* **UI.** Run every engine call in a worker thread (`QThreadPool`), never on the
  UI thread. Preview by drawing the original image with `alpha` as a mask (a
  checkerboard behind it); call `compose_rgba` only for export, since
  decontamination on a large photo takes about a second. Show
  `result.device` so a CPU fallback is visible.
* **Installer.** Freeze with PyInstaller (one-folder): `Cutout.exe` (window,
  calls `cutout.ui.main(argv)`) and `cutout-cli.exe` (console). Every DLL from
  the `nvidia-*` wheels is copied flat into `_internal\nvidia_dlls`, which
  `runtime.py` adds to the DLL search path before ORT loads CUDA. Models go in
  `{app}\models`. Setup carries only the app, libraries and Qt; the app
  downloads the rest on first start (`cutout.update`), so the window must open
  even with no models or CUDA DLLs installed. The smoke test after install is `cutout-cli.exe --diagnose`,
  `--list-models` and one run per subject on a generated sample. New models only
  need a registry entry: `scripts/build_models.py` builds and checks every entry.

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
