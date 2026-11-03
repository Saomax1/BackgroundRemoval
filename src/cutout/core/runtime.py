"""ONNX Runtime session creation and device selection.

The shipped build uses ``onnxruntime-gpu`` with the CUDA execution provider.
CUDA/cuDNN DLLs come from the ``nvidia-*`` pip wheels and are loaded with
``onnxruntime.preload_dlls()`` so nothing needs to be installed system-wide.
If CUDA cannot start (no NVIDIA driver, old driver), ``device="auto"`` falls
back to CPU and says so; ``device="cuda"`` raises instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_DLLS_PRELOADED = False


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


def _preload_cuda_dlls() -> None:
    global _DLLS_PRELOADED
    if _DLLS_PRELOADED:
        return
    _DLLS_PRELOADED = True
    ort = _ort()
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as exc:  # missing wheels just means CUDA is unavailable
        log.debug("onnxruntime.preload_dlls failed: %s", exc)


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

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    active = session.get_providers()[0]
    if want_cuda and active != "CUDAExecutionProvider":
        if device == "cuda":
            raise DeviceUnavailable(
                "CUDA execution provider failed to initialise (check the NVIDIA driver is up to date)"
            )
        log.warning("CUDA unavailable, running %s on CPU (this will be slow)", Path(model_path).name)
    return session, SessionInfo(model_path=Path(model_path), provider=active)
