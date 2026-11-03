# Windows installer

One `Cutout-Setup-<version>.exe` that installs everything: the app, the CUDA 12
runtime and cuDNN DLLs, and the AI models. After install the app is fully
offline. The only thing the PC needs is an NVIDIA display driver (without one
the app still runs, on the CPU, and the installer says so).

## Getting a build

GitHub Actions builds it: workflow **windows installer**
(`.github/workflows/build-installer.yml`). It runs on every push to `main`, on
pull requests that touch packaging, and on demand (Actions tab, "Run workflow").
The finished `Setup.exe` is under **Artifacts** on the run page (it downloads as
a zip; unzip, then run the exe). Windows SmartScreen will warn because the exe is
not code-signed: click "More info" then "Run anyway".

## What the pipeline does

1. **models** job (Linux, cached): `scripts/build_models.py` downloads the
   ready-made ONNX models, exports the PyTorch-only ones
   (`scripts/export_birefnet_onnx.py`, toolchain pinned in
   `requirements-export.txt`), then runs every model through the real pipeline
   on a test image. A model that fails to download, export or run is left out
   and recorded in `manifest.json`; the build fails only if a subject (person or
   anime) ends up with no working model. The cache key is a hash of the model
   recipe files, so this slow job only reruns when they change.
2. **prompt-models** job (Linux, cached): `scripts/build_prompt_models.py`
   exports the click-to-select model (SAM 2.1 Hiera-B+) and checks that one
   click selects a test shape. If it fails, the installer still builds, just
   without click-to-select.
3. **installer** job (Windows):
   * `pip install -e .[gpu,ui,build]` pulls `onnxruntime-gpu` plus the
     `nvidia-*` CUDA/cuDNN wheels (CUDA 13 since onnxruntime-gpu 1.2x; the spec
     finds DLLs anywhere under `site-packages/nvidia`, whatever the layout).
   * `pyinstaller packaging/cutout.spec` makes `dist/Cutout/` with `Cutout.exe`
     (window) and `cutout-cli.exe` (console), and copies every CUDA/cuDNN DLL
     flat into `_internal/nvidia_dlls/`. At startup `cutout.core.runtime` puts
     that folder on the DLL search path and preloads it.
   * `packaging/stage_models.py` copies the working models into
     `dist/Cutout/models`. If they exceed the budget (2.6 GB) it drops redundant
     fallbacks, largest first; the best model for each subject always stays.
   * Inno Setup 6.5.2+ compiles `packaging/cutout.iss`. A single Setup.exe can
     hold about 4 GB, and the job fails if it would be bigger.
   * Smoke test: silent install on the runner, then `cutout-cli --diagnose`
     must show the CUDA provider DLL loading from the bundle (no GPU needed for
     that), `--list-models`, and one person and one anime cutout on CPU.

## Install behaviour

* Installs per user to `%LOCALAPPDATA%\Programs\Cutout` with no admin prompt
  (the first page offers "all users" instead).
* Start-menu entry, desktop shortcut (ticked by default), uninstaller.
* Upgrading replaces `models\` and `_internal\` wholesale.

## Building locally (Windows)

```
pip install -e ".[gpu,ui,build]" -r packaging/requirements-export.txt
python scripts/build_models.py --models-dir build/models
pyinstaller packaging/cutout.spec --noconfirm
python packaging/stage_models.py build/models dist/Cutout/models
"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" packaging\cutout.iss
```

## Checking the GPU on a user's PC

`"%LOCALAPPDATA%\Programs\Cutout\cutout-cli.exe" --diagnose` prints the ONNX
Runtime providers and whether the bundled CUDA DLLs load.
