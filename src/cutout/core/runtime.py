"""ONNX Runtime session creation and device selection.

The shipped build uses ``onnxruntime-gpu`` with the CUDA execution provider.
CUDA/cuDNN DLLs come from the ``nvidia-*`` pip wheels and are loaded with
``onnxruntime.preload_dlls()`` so nothing needs to be installed system-wide.
If CUDA cannot start (no NVIDIA driver, old driver), ``device="auto"`` falls
back to CPU and says so; ``device="cuda"`` raises instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
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


class GpuOutOfMemory(RuntimeError):
    """The graphics card (or, after the CPU fallback, the PC) ran out of memory for a model."""


# Fragments ORT / CUDA / cuDNN put in the message when an allocation fails.
_OOM_MARKERS = (
    "failed to allocate memory",
    "bfcarena",
    "out of memory",
    "cudaerrormemoryallocation",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "std::bad_alloc",
)


def is_out_of_memory(exc: BaseException) -> bool:
    """True if ``exc`` is an allocation failure from ONNX Runtime or the CUDA libraries."""
    if isinstance(exc, (MemoryError, GpuOutOfMemory)):
        return True
    text = str(exc).lower()
    return any(m in text for m in _OOM_MARKERS)


@dataclass(frozen=True)
class SessionInfo:
    model_path: Path
    provider: str  # the execution provider actually running the graph
    # The cached optimised graph this session was loaded from (``None``: built from the model).
    graph_cache: Path | None = None

    @property
    def on_gpu(self) -> bool:
        return self.provider != "CPUExecutionProvider"


def run_options(info: SessionInfo):
    """``RunOptions`` that hand the memory a run used back to the system once it finishes.

    Without this ONNX Runtime keeps every activation buffer of the biggest run in its
    arena for the life of the session, so a 2048 px BiRefNet run parks several GB of
    VRAM that the next model then cannot get.
    """
    ort = _ort()
    ro = ort.RunOptions()
    if info.on_gpu:
        ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    return ro


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


def graph_cache_root() -> Path | None:
    """Folder for optimised model graphs, or ``None`` if caching is off.

    ``%LOCALAPPDATA%\\Cutout\\graph-cache`` on Windows. ``CUTOUT_GRAPH_CACHE=0`` turns it off and
    any other value is used as the folder.
    """
    env = os.environ.get("CUTOUT_GRAPH_CACHE", "").strip()
    if env == "0":
        return None
    if env:
        return Path(env)
    return debug.log_dir().parent / "graph-cache"


def _graph_cache_dir(model_path: Path, providers: list) -> Path | None:
    """The cache folder for this exact model file, ONNX Runtime build, device and settings.

    Any change (a model update, a new ONNX Runtime, other provider options, another CPU)
    gives a new key, so a stale graph is never loaded.
    """
    root = graph_cache_root()
    if root is None:
        return None
    try:
        st = Path(model_path).stat()
    except OSError:
        return None
    ort = _ort()
    key = json.dumps({
        "model": str(Path(model_path).resolve()), "size": st.st_size, "mtime": st.st_mtime_ns,
        "ort": ort.__version__, "providers": repr(providers), "level": "all", "format": 1,
        "cpu": platform.processor() or platform.machine(),
    }, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return root / f"{Path(model_path).stem}-{digest}"


def _session_options(save_to: Path | None = None, optimize: bool = True):
    ort = _ort()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    # The CPU arena never gives memory back, and the person model's input and any
    # CPU-placed nodes are large; plain malloc/free keeps RAM use to what is in use now.
    opts.enable_cpu_mem_arena = False
    opts.log_severity_level = debug.ort_log_severity()  # info in debug mode, errors otherwise
    if save_to is not None:
        # Weights go to a side file: a 2048 px model is too big for one protobuf file.
        opts.optimized_model_filepath = str(save_to / GRAPH_FILE)
        opts.add_session_config_entry("session.optimized_model_external_initializers_file_name", WEIGHTS_FILE)
        opts.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    return opts


def _provider_name(p) -> str:
    return p[0] if isinstance(p, tuple) else p


GRAPH_FILE = "model.onnx"
WEIGHTS_FILE = "weights.bin"


def forget_graph_cache(info: SessionInfo) -> None:
    """Delete the cached graph a session came from (it failed to run); the next load rebuilds it."""
    if info.graph_cache is not None:
        log.warning("dropping the cached graph %s", info.graph_cache.name)
        shutil.rmtree(info.graph_cache, ignore_errors=True)


def _store_graph(tmp: Path, final: Path) -> None:
    """Move a freshly saved graph into place and remove older graphs of the same model."""
    try:
        stem = final.name.rsplit("-", 1)[0]
        same_model = re.compile(re.escape(stem) + r"-[0-9a-f]{16}(\.tmp\d+)?")
        for old in final.parent.iterdir():
            # Older graphs of this model and saves a crash left half done.
            if old.is_dir() and old not in (final, tmp) and same_model.fullmatch(old.name):
                shutil.rmtree(old, ignore_errors=True)
        tmp.rename(final)  # a folder rename is atomic: a half-written graph is never used
        log.info("saved the optimised graph of %s for faster starts", final.name)
    except OSError as exc:  # e.g. another copy of the app saved it first
        log.debug("could not store the optimised graph %s: %s", final.name, exc)
        shutil.rmtree(tmp, ignore_errors=True)


def create_session(model_path: Path, device: str = "auto", graph_cache: bool = False):
    """Return ``(InferenceSession, SessionInfo)`` for an ONNX file.

    ``device`` is ``"auto"`` (CUDA if it works, else CPU), ``"cuda"`` or ``"cpu"``.

    With ``graph_cache``, the graph ONNX Runtime builds from the model (fused, constant-folded)
    is saved on the first load and reused by later ones, which then skip that work: it was most
    of every start's load time. The cached graph is the same graph the session would build, so
    the output does not change. A cached graph that fails to load is deleted and rebuilt.
    """
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"device must be auto, cuda or cpu, not {device!r}")
    ort = _ort()

    providers: list = ["CPUExecutionProvider"]
    want_cuda = device in ("auto", "cuda")
    if want_cuda:
        _preload_cuda_dlls()
        if "CUDAExecutionProvider" in ort.get_available_providers():
            cuda_opts = {
                # EXHAUSTIVE benchmarking tries algorithms that need GB-sized workspaces and
                # keeps the winners' workspace around; that alone was enough to run a
                # 2048 px model out of VRAM. HEURISTIC picks well with no trial allocations.
                "cudnn_conv_algo_search": "HEURISTIC",
                "cudnn_conv_use_max_workspace": "0",
                # Grow the arena by exactly what is asked, so shrinkage can return it.
                "arena_extend_strategy": "kSameAsRequested",
            }
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        elif device == "cuda":
            raise DeviceUnavailable("onnxruntime was built without CUDA support")

    name = Path(model_path).name
    cache_dir = _graph_cache_dir(Path(model_path), providers) if graph_cache else None
    debug.log_memory(f"before loading {name}")
    t0 = time.perf_counter()
    session = None
    source = None
    if cache_dir is not None and (cache_dir / GRAPH_FILE).is_file():
        try:
            session = ort.InferenceSession(
                str(cache_dir / GRAPH_FILE), sess_options=_session_options(optimize=False), providers=providers
            )
            source = cache_dir
            if session.get_providers()[0] != _provider_name(providers[0]):
                # Built for a device that did not start this time: not the graph to use here.
                log.info("cached graph for %s is for another device; loading the model", name)
                session, source = None, None
        except Exception as exc:
            if is_out_of_memory(exc):
                raise
            log.warning("cached graph for %s did not load (%s); rebuilding it", name, exc)
            shutil.rmtree(cache_dir, ignore_errors=True)
    if session is None:
        tmp = None
        if cache_dir is not None:
            tmp = cache_dir.with_name(f"{cache_dir.name}.tmp{os.getpid()}")
            try:
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
            except OSError as exc:
                log.debug("graph cache unavailable (%s)", exc)
                tmp = None
        try:
            session = ort.InferenceSession(str(model_path), sess_options=_session_options(tmp), providers=providers)
        except Exception as exc:
            if tmp is None or is_out_of_memory(exc):
                raise
            log.warning("saving the optimised graph of %s failed (%s); loading without it", name, exc)
            shutil.rmtree(tmp, ignore_errors=True)
            tmp = None
            session = ort.InferenceSession(str(model_path), sess_options=_session_options(), providers=providers)
        if tmp is not None:
            # Keep only a graph built for the device asked for (not a CPU fallback's).
            if (tmp / GRAPH_FILE).is_file() and session.get_providers()[0] == _provider_name(providers[0]):
                _store_graph(tmp, cache_dir)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
    active = session.get_providers()[0]
    log.info("loaded %s on %s in %.1fs%s", name, active, time.perf_counter() - t0,
             " (optimised graph from cache)" if source is not None else "")
    debug.log_memory(f"after loading {name}")
    if want_cuda and active != "CUDAExecutionProvider":
        if device == "cuda":
            raise DeviceUnavailable(
                "CUDA execution provider failed to initialise (check the NVIDIA driver is up to date)"
            )
        log.warning("CUDA unavailable, running %s on CPU (this will be slow)", name)
    return session, SessionInfo(model_path=Path(model_path), provider=active, graph_cache=source)
