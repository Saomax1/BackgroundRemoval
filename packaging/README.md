# Windows installer and updates

`Cutout-Setup-<version>.exe` (a few hundred MB) installs the app. The first time
Cutout starts it downloads the rest once: the CUDA/cuDNN DLLs and the AI models,
about 3 GB. After that the app is fully offline. The only thing the PC needs is
an NVIDIA display driver (without one the app still runs, on the CPU, and the
installer says so).

## Getting a build

Download `Cutout-Setup-<version>.exe` from the latest release:
<https://github.com/Saomax1/BackgroundRemoval/releases/latest> (no GitHub sign-in
needed). Windows SmartScreen warns the first time because the exe is not
code-signed: click "More info" then "Run anyway". Updates installed from inside
the app don't show that warning.

GitHub Actions builds it: workflow **windows installer**
(`.github/workflows/build-installer.yml`). Every push to `main` publishes a
release, versioned `<major>.<minor>.<build number>` (`packaging/set_version.py`;
change major/minor in `src/cutout/__init__.py`). Pull requests that touch
packaging build and test the same way but publish nothing; their Setup.exe is
under **Artifacts** on the run page, for testing only.

## Updates

**Help > Check for Updates** looks at the latest release on GitHub. Nothing goes
online unless the user clicks it or ticks **Help > Check for Updates at Startup**
(off by default). If a newer version exists it shows the release notes and the
download size, and **Install Update** does the rest: Cutout restarts on the new
version a moment later. Settings (Qt keeps them in the registry) and logs
(`%LOCALAPPDATA%\Cutout\logs`) are outside the install folder and never touched.

How it stays small: `packaging/make_release.py` splits the app folder into
*packs* (`src/cutout/update/packs.py`):

| Pack | What | Changes |
|---|---|---|
| `app` | `Cutout.exe`, `cutout-cli.exe` (all of Cutout's Python code is inside them), `_internal/cutout` | every release, tens of MB |
| `libs` | Python, NumPy, OpenCV, ONNX Runtime | when a dependency updates |
| `qt` | PySide6 / Qt | rarely |
| `gpu-runtime` | CUDA + cuDNN DLLs, ONNX Runtime's CUDA provider | rarely, ~1 GB |
| `model-<name>` | one per AI model file | when that model is rebuilt |

Each release has `cutout-release.json`, listing every pack's content hash, files
and download parts. The app downloads only packs whose hash differs from what is
installed. A pack that did not change keeps pointing at the release that first
uploaded it, so the big packs are uploaded once. Such a reused pack gets no local
zip in the build, so the CI install test downloads it from the published release
and checks its SHA-256, and `make_release.py` refuses to finish if any download
in the manifest is missing or the wrong size. **Don't delete old releases**:
newer ones may still download their models from them. Packs over 1.9 GB are split
into parts, because GitHub caps release files at 2 GB.

Safety: downloads resume after a dropped connection; every part is SHA-256
checked and every unpacked pack is hashed against the manifest before anything in
the install folder changes. Windows won't let a running program replace its own
files, so the swap is done by `src/cutout/update/apply_update.ps1` after the app
exits: it moves each old file into a backup folder, moves the new ones in, and
restarts Cutout. If any step fails it puts every old file back, and Cutout says
on the next start that the update was not installed.

The command line does the same: `cutout-cli --version`, `--check-updates`,
`--update` (`--manifest FILE --from-dir DIR` installs from local files, which is
what the CI smoke test does).

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
   * `packaging/make_release.py` zips the packs into `dist/release`, writes the
     manifest (reusing the previous release's downloads for unchanged packs) and
     stages the Setup input (app, libs, Qt) in `dist/setup`.
   * Inno Setup 6.5.2+ compiles `packaging/cutout.iss`. The job fails if
     Setup.exe would be over GitHub's 2 GB file limit.
   * Smoke test: silent install on the runner; `cutout-cli --update` finishes the
     install from the local packs (same code path as the first-start download);
     `cutout-cli --diagnose` must show the CUDA provider DLL loading from the
     install (no GPU needed for that), `--list-models`, and one person and one
     anime cutout on CPU. Then it builds a newer version with one changed file,
     checks that only the `app` pack is offered, applies it and cuts out again.
   * On `main`: publishes the release (as a draft first, manifest last, so the
     app never sees a half-uploaded release).

## Install behaviour

* Installs per user to `%LOCALAPPDATA%\Programs\Cutout` with no admin prompt.
  Always per user, so the updater can write to the folder.
* Start-menu entry, desktop shortcut (ticked by default), uninstaller (which also
  removes what the app downloaded).
* Running a newer Setup over an install keeps the downloaded models and GPU files.
  The next start re-checks every file by hash (a few seconds) and downloads only
  what is missing or outdated.

## Building locally (Windows)

```
pip install -e ".[gpu,ui,build]" -r packaging/requirements-export.txt
python scripts/build_models.py --models-dir build/models
pyinstaller packaging/cutout.spec --noconfirm
python packaging/stage_models.py build/models dist/Cutout/models
python packaging/make_release.py dist/Cutout dist/release --version 0.1.0 --base-url https://example.invalid --setup-dir dist/setup
"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" packaging\cutout.iss
dist\installer\Cutout-Setup-0.1.0.exe
"%LOCALAPPDATA%\Programs\Cutout\cutout-cli.exe" --update --manifest "%LOCALAPPDATA%\Programs\Cutout\release.json" --from-dir dist\release
```

## Checking the GPU on a user's PC

`"%LOCALAPPDATA%\Programs\Cutout\cutout-cli.exe" --diagnose` prints the ONNX
Runtime providers and whether the bundled CUDA DLLs load.
