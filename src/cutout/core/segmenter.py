"""Segmenters: turn an RGB image into a coarse alpha matte.

A segmenter only has to be right about *what* is foreground. It works at the
network's own resolution; ``refine.py`` takes care of bringing the matte back to
the full input resolution with sharp edges. Anything implementing the
``Segmenter`` protocol can be plugged into the engine (tests use a fake one;
a future click-to-select model can reuse the same shape).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .models import ModelSpec, ResizeMode
from .runtime import SessionInfo, create_session, run_options


class Segmenter(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """``rgb``: HxWx3 uint8. Returns float32 alpha in [0, 1] covering the whole image,
        at any resolution (the engine stretches it back to the exact input size)."""
        ...


class OnnxSegmenter:
    def __init__(self, spec: ModelSpec, model_file: Path, device: str = "auto"):
        self.spec = spec
        self.session, self.info = create_session(model_file, device)
        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        self._input_dtype = np.float16 if "float16" in inp.type else np.float32
        self._run_opts = run_options(self.info)

    @property
    def name(self) -> str:
        return self.spec.id

    @property
    def session_info(self) -> SessionInfo:
        return self.info

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return self._predict(rgb, self._run_opts)

    def _predict(self, rgb: np.ndarray, opts) -> np.ndarray:
        tensor, crop = preprocess(rgb, self.spec)
        outputs = self.session.run(None, {self._input_name: tensor.astype(self._input_dtype)}, opts)
        return postprocess(outputs[self.spec.output_index], crop, self.spec)

    def warmup(self, cancel: threading.Event | None = None) -> None:
        """Run once on a blank image so the session's slow first run happens before the user's.

        Raises ``WarmupCancelled`` if ``cancel`` is set (then call ``cancel_warmup()`` to stop
        a run already in progress).
        """
        s = self.spec.input_size
        opts = run_options(self.info)  # its own options, so cancelling never touches a user run
        self._warm_opts = opts  # publish before checking ``cancel``: no window to miss a cancel
        try:
            if cancel is not None and cancel.is_set():
                raise WarmupCancelled(self.spec.id)
            self._predict(np.zeros((s, s, 3), np.uint8), opts)
        except WarmupCancelled:
            raise
        except Exception as exc:
            if opts.terminate:
                raise WarmupCancelled(self.spec.id) from exc
            raise
        finally:
            self._warm_opts = None

    def cancel_warmup(self) -> None:
        """Stop a warm-up running on another thread (ONNX Runtime checks between nodes)."""
        opts = getattr(self, "_warm_opts", None)
        if opts is not None:
            opts.terminate = True


class WarmupCancelled(RuntimeError):
    """A background warm-up was stopped to let a cutout use the GPU first."""


def preprocess(rgb: np.ndarray, spec: ModelSpec) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return the NCHW float32 tensor and the (top, left, height, width) of the image inside it."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("segmenter input must be HxWx3 uint8")
    s = spec.input_size
    h, w = rgb.shape[:2]
    if spec.resize_mode is ResizeMode.STRETCH:
        resized = _resize(rgb, s, s)
        canvas = resized
        crop = (0, 0, s, s)
    else:
        scale = s / max(h, w)
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        top, left = (s - nh) // 2, (s - nw) // 2
        canvas = np.zeros((s, s, 3), np.uint8)
        canvas[top : top + nh, left : left + nw] = _resize(rgb, nw, nh)
        crop = (top, left, nh, nw)

    x = canvas.astype(np.float32) / 255.0
    x = (x - np.asarray(spec.mean, np.float32)) / np.asarray(spec.std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), crop


def postprocess(raw: np.ndarray, crop: tuple[int, int, int, int], spec: ModelSpec) -> np.ndarray:
    pred = np.asarray(raw, dtype=np.float32)
    pred = pred.reshape(pred.shape[-2], pred.shape[-1])  # (1,1,H,W) / (1,H,W) / (H,W)
    if spec.activation == "sigmoid" or (
        spec.activation == "auto" and (pred.min() < 0.0 or pred.max() > 1.0)
    ):
        pred = 1.0 / (1.0 + np.exp(-pred))
    top, left, ch, cw = crop
    if pred.shape != (spec.input_size, spec.input_size):
        # Some exports emit a different output size; map the crop proportionally.
        fy, fx = pred.shape[0] / spec.input_size, pred.shape[1] / spec.input_size
        top, left = round(top * fy), round(left * fx)
        ch, cw = max(1, round(ch * fy)), max(1, round(cw * fx))
    return np.clip(pred[top : top + ch, left : left + cw], 0.0, 1.0)


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    shrinking = width * height < img.shape[0] * img.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, (width, height), interpolation=interp)
