# PyInstaller spec for the Windows build (one-folder). Run from the repo root:
#     pyinstaller packaging/cutout.spec --noconfirm
# Produces dist/Cutout/ with Cutout.exe (windowed app) and cutout-cli.exe (console),
# sharing one _internal/ folder. Models are added later by packaging/stage_models.py.
# -*- mode: python -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
PKG = ROOT / "packaging"
CUDA_DEST = "nvidia_dlls"  # must match cutout.core.runtime.FROZEN_CUDA_DIR


def nvidia_dlls():
    """Every DLL from the nvidia-* wheels that onnxruntime-gpu[cuda,cudnn] installed, copied flat."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("nvidia-* CUDA wheels not installed: pip install -e .[gpu,build]")
    found = {}
    for root in spec.submodule_search_locations:
        for dll in sorted(Path(root).glob("*/bin/*.dll")):
            found.setdefault(dll.name.lower(), dll)
    if not any(n.startswith("cudnn64_") for n in found) or not any(n.startswith("cudart64_") for n in found):
        raise SystemExit(f"CUDA runtime or cuDNN DLLs missing; found only {sorted(found)}")
    print(f"bundling {len(found)} CUDA/cuDNN DLLs, {sum(p.stat().st_size for p in found.values()) / 1e9:.2f} GB")
    return [(str(p), CUDA_DEST) for p in found.values()]


hidden = collect_submodules("cutout")
datas = collect_data_files("cutout")  # UI icons/styles shipped as package data
excludes = ["nvidia", "torch", "torchvision", "tkinter", "matplotlib", "IPython", "pytest"]


def analysis(script):
    return Analysis(
        [str(PKG / script)],
        pathex=[str(ROOT / "src")],
        # ORT's provider DLLs (onnxruntime_providers_cuda.dll etc.) are loaded at runtime,
        # not imported, so collect them explicitly in case the contrib hook misses one.
        binaries=collect_dynamic_libs("onnxruntime") + (nvidia_dlls() if script == "entry_cli.py" else []),
        datas=datas,
        hiddenimports=hidden,
        excludes=excludes,
        noarchive=False,
    )


gui_a = analysis("entry_gui.py")
cli_a = analysis("entry_cli.py")

gui_exe = EXE(
    PYZ(gui_a.pure),
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name="Cutout",
    console=False,
    upx=False,
    icon=str(PKG / "cutout.ico") if (PKG / "cutout.ico").exists() else None,
)
cli_exe = EXE(
    PYZ(cli_a.pure),
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name="cutout-cli",
    console=True,
    upx=False,
)

coll = COLLECT(
    gui_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_exe,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=False,
    name="Cutout",
)
