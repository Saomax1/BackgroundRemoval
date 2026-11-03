"""ONNX Runtime session creation and device selection.

The shipped build uses ``onnxruntime-gpu`` with the CUDA execution provider.
CUDA/cuDNN DLLs come from the ``nvidia-*`` pip wheels and are loaded with
``onnxruntime.preload_dlls()`` so nothing needs to be installed system-wide.
If CUDA cannot start (no NVIDIA driver, old driver), ``device="auto"`` falls
back to CPU and says so; ``device="cuda"`` raises instead.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import debug

log = logging.getLogger(__name__)

_DLLS_PRELOADED = False
_DLL_DIR_HANDLES: list = []  # keep add_dll_directory handles alive for the process lifetime

# Folder (inside the PyInstaller bundle) that the installer build copies every
# CUDA/cuDNN DLL from the ``nvidia-*`` wheels into. Kept flat on purpose: cuDNN 9
# loads its engine sub-libraries from the folder that holds ``cudnn64_9.dll``.
FROZEN_CUDA_DIR = "nvidia_dlls"


class DeviceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionInfo:
    model_path: Path
    provider: str  # the execution provider actually running the graph

    @property
    def on_gpu(self) -> bool:
        return self.provider != "CPUExecutionProvider"


def _ort():
    import onnxruntime as ort  # imported lazily so the UI can start before the heavy DLLs load

    return ort


def bundled_cuda_dir() -> Path | None:
    """The CUDA DLL folder of an installed (frozen) build, or ``None`` when running from source.

    In a source checkout the DLLs stay in ``site-packages/nvidia/*`` and
    ``onnxruntime.preload_dlls()`` finds them there by itself.
    """
    base = getattr(sys, "_MEIPASS", None)
    if not getattr(sys, "frozen", False) or base is None:
        return None
    path = Path(base) / FROZEN_CUDA_DIR
    return path if path.is_dir() else None


def _expose_dll_dir(path: Path) -> None:
    """Let Windows find DLLs in ``path`` for every loader style ORT may use."""
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
    add = getattr(os, "add_dll_directory", None)  # Windows only
    if add is not None:
        try:
            _DLL_DIR_HANDLES.append(add(str(path)))
        except OSError as exc:
            log.debug("add_dll_directory(%s) failed: %s", path, exc)


def _preload_cuda_dlls() -> None:
    global _DLLS_PRELOADED
    if _DLLS_PRELOADED:
        return
    _DLLS_PRELOADED = True
    bundled = bundled_cuda_dir()
    if bundled is not None:
        _expose_dll_dir(bundled)
    ort = _ort()
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        if bundled is not None:
            # PyInstaller already ships the MSVC runtime next to the exe.
            preload(msvc=False, directory=str(bundled))
        else:
            preload()
    except Exception as exc:  # missing wheels just means CUDA is unavailable
        log.debug("onnxruntime.preload_dlls failed: %s", exc)


def diagnose() -> list[str]:
    """Human-readable lines describing the GPU setup, for ``cutout --diagnose`` and bug reports."""
    lines = []
    bundled = bundled_cuda_dir()
    lines.append(f"frozen build: {bool(getattr(sys, 'frozen', False))}")
    lines.append(f"bundled CUDA folder: {bundled or '(not frozen, using site-packages)'}")
    _preload_cuda_dlls()
    try:
        ort = _ort()
    except Exception as exc:
        return lines + [f"onnxruntime failed to import: {exc}"]
    lines.append(f"onnxruntime {ort.__version__}, device {ort.get_device()}")
    lines.append(f"providers: {', '.join(ort.get_available_providers())}")
    if bundled is not None and sys.platform == "win32":
        import ctypes

        dlls = sorted(bundled.glob("*.dll"))
        bad = []
        for dll in dlls:
            try:
                ctypes.WinDLL(str(dll))
            except OSError as exc:
                bad.append(f"{dll.name}: {exc}")
        lines.append(f"CUDA DLLs loadable: {len(dlls) - len(bad)}/{len(dlls)}")
        lines.extend(f"  cannot load {b}" for b in bad)
    lines.append(f"CUDA provider DLL loads: {'yes' if cuda_provider_loads() else 'no'}")
    return lines


def cuda_provider_loads() -> bool:
    """True if ORT's CUDA provider DLL and all its CUDA/cuDNN dependencies resolve.

    This needs no GPU (the driver is only touched on first use), so it is how the
    installer CI proves the bundle is complete on a runner without an NVIDIA card.
    """
    if sys.platform != "win32":
        return "CUDAExecutionProvider" in _ort().get_available_providers()
    import ctypes

    _preload_cuda_dlls()
    capi = Path(_ort().__file__).resolve().parent / "capi"
    for name in ("onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"):
        try:
            ctypes.WinDLL(str(capi / name))
        except OSError as exc:
            log.debug("cannot load %s: %s", name, exc)
            return False
    return True


def available_devices() -> list[str]:
    ort = _ort()
    providers = ort.get_available_providers()
    devices = ["cpu"]
    if "CUDAExecutionProvider" in providers:
        devices.insert(0, "cuda")
    return devices


def create_session(model_path: Path, device: str = "auto"):
    """Return ``(InferenceSession, SessionInfo)`` for an ONNX file.

    ``device`` is ``"auto"`` (CUDA if it works, else CPU), ``"cuda"`` or ``"cpu"``.
    """
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda or cpu, not {device!r}")
    ort = _ort()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3

    providers: list = ["CPUExecutionProvider"]
    want_cuda = device in ("auto", "cuda")
    if want_cuda:
        _preload_cuda_dlls()
        if "CUDAExecutionProvider" in ort.get_available_providers():
            cuda_opts = {
                # Big conv nets at fixed input size: let cuDNN benchmark once, then reuse.
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "arena_extend_strategy": "kSameAsRequested",
            }
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        elif device == "cuda":
            raise DeviceUnavailable("onnxruntime was built without CUDA support")

    opts.log_severity_level = debug.ort_log_severity()  # info in debug mode, errors otherwise
    debug.log_memory(f"before loading {Path(model_path).name}")
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    active = session.get_providers()[0]
    log.info("loaded %s on %s in %.1fs", Path(model_path).name, active, time.perf_counter() - t0)
    debug.log_memory(f"after loading {Path(model_path).name}")
    if want_cuda and active != "CUDAExecutionProvider":
        if device == "cuda":
            raise DeviceUnavailable(
                "CUDA execution provider failed to initialise (check the NVIDIA driver is up to date)"
            )
        log.warning("CUDA unavailable, running %s on CPU (this will be slow)", Path(model_path).name)
    return session, SessionInfo(model_path=Path(model_path), provider=active)
