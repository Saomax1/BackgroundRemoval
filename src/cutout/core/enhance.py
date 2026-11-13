"""Enhance: make a small or blurry picture bigger and sharper before the cutout.

Two optional models, both run through ONNX Runtime like the cutout models:

* **Upscaler** (Real-ESRGAN, ``realesrgan-x4plus`` for photos, ``...-anime-6b`` for
  drawings). A 4x network, run in overlapping tiles so any picture fits in graphics
  memory. Each tile is padded with real neighbouring pixels and only its middle is
  kept, so there are no seams. 2x and 1x ("clean up only") average 2x2 or 4x4 blocks
  of the 4x output, inside each tile, so the big 4x picture is never held in memory.
* **Face restorer** (GFPGAN 1.4, photos only). Every face YuNet finds on the upscaled
  picture is aligned to the 512x512 face template from its five landmarks, repaired,
  warped back and blended in with a feathered mask. ``face_strength`` mixes the repaired
  face with the upscaled one (1 = fully repaired), which keeps the person's likeness.

Bit depth and colour profile are kept: a 16-bit picture is upscaled in floating point
and comes back 16-bit. The output is ``scale`` times the input size, capped at
``MAX_OUTPUT_PIXELS`` (a bigger request is lowered to the largest scale that fits).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from .models import EnhanceModelSpec
from .runtime import SessionInfo, create_session, run_options

log = logging.getLogger(__name__)

#: Scales the user can pick. 1 = same size, only cleaned up.
SCALES = (1, 2, 4)
#: Largest picture Enhance makes (36 MP, e.g. 6000 x 6000). The cutout then runs on it
#: at full resolution, and its RAM use grows with the pixel count.
MAX_OUTPUT_PIXELS = 36_000_000
#: Input tile side and the overlap kept around each tile (both in input pixels).
TILE = 256
TILE_PAD = 24

# Where the eyes, nose and mouth corners sit in GFPGAN's 512x512 face (FFHQ alignment).
FACE_TEMPLATE_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
# Eye distance in that template, used to skip faces that are already big and sharp.
_TEMPLATE_EYES = float(np.linalg.norm(FACE_TEMPLATE_512[1] - FACE_TEMPLATE_512[0]))

Progress = Callable[[int, int], None]  # (steps done, steps in total)


class EnhanceTooLarge(ValueError):
    """The picture is already too big to enhance at any scale."""


class Upscaler(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def scale(self) -> int: ...

    def upscale(self, tile: np.ndarray) -> np.ndarray:
        """``tile``: HxWx3 float32 RGB in [0, 1]. Returns (H*scale)x(W*scale)x3 float32."""
        ...


class FaceRestorer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def face_size(self) -> int: ...

    def restore(self, face: np.ndarray) -> np.ndarray:
        """``face``: SxSx3 float32 RGB in [0, 1], aligned. Returns the repaired face, same shape."""
        ...


class _OnnxModel:
    """One ONNX session, opened lazily, with the first input/output names read from the graph."""

    def __init__(self, spec: EnhanceModelSpec, model_file: Path, device: str = "auto"):
        self.spec = spec
        self.session, self.info = create_session(Path(model_file), device)
        inp = self.session.get_inputs()[0]
        self._input_name = inp.name
        self._dtype = np.float16 if "float16" in inp.type else np.float32
        self._extra = {}
        # Some face exports take a second input (e.g. a fidelity weight); feed it a default.
        for extra in self.session.get_inputs()[1:]:
            shape = [d if isinstance(d, int) else 1 for d in extra.shape]
            self._extra[extra.name] = np.full(shape, 1.0, np.float64 if "double" in extra.type else np.float32)
        self._run_opts = run_options(self.info)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.spec.id

    @property
    def session_info(self) -> SessionInfo:
        return self.info

    def _run(self, nchw: np.ndarray) -> np.ndarray:
        feed = {self._input_name: nchw.astype(self._dtype), **self._extra}
        with self._lock:
            out = self.session.run(None, feed, self._run_opts)[0]
        return np.asarray(out, np.float32)


class OnnxUpscaler(_OnnxModel):
    """Real-ESRGAN (RRDBNet): RGB in [0, 1] in, RGB in [0, 1] out, ``spec.scale`` times bigger."""

    @property
    def scale(self) -> int:
        return self.spec.scale

    def upscale(self, tile: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(tile.transpose(2, 0, 1)[None])
        out = self._run(x)[0].transpose(1, 2, 0)
        return np.clip(out, 0.0, 1.0)


class OnnxFaceRestorer(_OnnxModel):
    """GFPGAN: an aligned RGB face in [-1, 1] in, the repaired face in [-1, 1] out."""

    @property
    def face_size(self) -> int:
        return self.spec.face_size

    def restore(self, face: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray((face * 2.0 - 1.0).transpose(2, 0, 1)[None])
        out = self._run(x)[0].transpose(1, 2, 0)
        return np.clip((out + 1.0) * 0.5, 0.0, 1.0)


# ------------------------------------------------------------------------- scale


def pick_scale(width: int, height: int, wanted: int, max_pixels: int = MAX_OUTPUT_PIXELS) -> int:
    """The biggest of ``SCALES`` up to ``wanted`` whose output fits in ``max_pixels``.

    Raises ``EnhanceTooLarge`` if even 1x does not fit.
    """
    if wanted not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, not {wanted!r}")
    for s in sorted((s for s in SCALES if s <= wanted), reverse=True):
        if width * s * height * s <= max_pixels:
            return s
    raise EnhanceTooLarge(
        f"This picture is {width} x {height}, which is already too big to enhance "
        f"(the limit is {max_pixels / 1e6:.0f} megapixels).")


# ------------------------------------------------------------------- upscaling


def tiles(height: int, width: int, tile: int = TILE) -> list[tuple[int, int, int, int]]:
    """``(y0, y1, x0, x1)`` boxes covering the image, about ``tile`` square, evenly sized."""
    ny, nx = max(1, math.ceil(height / tile)), max(1, math.ceil(width / tile))
    ys = [round(i * height / ny) for i in range(ny + 1)]
    xs = [round(i * width / nx) for i in range(nx + 1)]
    return [(ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(ny) for j in range(nx)]


def _box_down(a: np.ndarray, k: int) -> np.ndarray:
    """Average ``k`` x ``k`` blocks (exact box filter; ``a``'s sides are multiples of ``k``)."""
    if k == 1:
        return a
    h, w, c = a.shape
    return a.reshape(h // k, k, w // k, k, c).mean(axis=(1, 3), dtype=np.float32)


def upscale_image(
    rgb: np.ndarray,
    upscaler: Upscaler,
    scale: int,
    tile: int = TILE,
    pad: int = TILE_PAD,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray:
    """Upscale ``rgb`` (HxWx3 uint8 or uint16) by ``scale`` (1, 2 or 4) with a 4x network.

    Returns the same dtype at ``scale`` times the size. Tiles overlap by ``pad`` input
    pixels on every side that has a neighbour; only each tile's own area is written.
    """
    net = upscaler.scale
    if net % scale:
        raise ValueError(f"a {net}x upscaler can't make {scale}x")
    k = net // scale
    h, w = rgb.shape[:2]
    maxv = 65535.0 if rgb.dtype == np.uint16 else 255.0
    out = np.empty((h * scale, w * scale, 3), rgb.dtype)
    boxes = tiles(h, w, tile)
    for n, (y0, y1, x0, x1) in enumerate(boxes):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("enhance cancelled")
        py0, py1 = max(0, y0 - pad), min(h, y1 + pad)
        px0, px1 = max(0, x0 - pad), min(w, x1 + pad)
        src = rgb[py0:py1, px0:px1].astype(np.float32) / np.float32(maxv)
        big = upscaler.upscale(src)
        if big.shape[:2] != ((py1 - py0) * net, (px1 - px0) * net):
            raise RuntimeError(f"{upscaler.name} returned {big.shape[:2]} for a "
                               f"{py1 - py0}x{px1 - px0} tile; expected {net}x")
        core = big[(y0 - py0) * net:(y1 - py0) * net, (x0 - px0) * net:(x1 - px0) * net]
        small = _box_down(core, k)
        out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = np.clip(
            np.rint(small * np.float32(maxv)), 0, maxv).astype(rgb.dtype)
        if progress is not None:
            progress(n + 1, len(boxes))
    return out


# ------------------------------------------------------------- face restoration


@dataclass
class FaceFix:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 of the area that changed
    eye_distance: float


def restore_faces(
    rgb: np.ndarray,
    faces: list,
    restorer: FaceRestorer,
    strength: float = 0.7,
    max_upsize: float = 2.0,
) -> list[FaceFix]:
    """Repair each face of ``faces`` (``head.Face`` with landmarks) in ``rgb``, in place.

    Faces whose eyes are more than ``max_upsize`` times further apart than in the model's
    template are skipped: the repaired face would be softer than the picture around it.
    """
    size = restorer.face_size
    template = FACE_TEMPLATE_512 * (size / 512.0)
    maxv = 65535.0 if rgb.dtype == np.uint16 else 255.0
    h, w = rgb.shape[:2]
    fixed: list[FaceFix] = []
    for face in faces:
        marks = getattr(face, "landmarks", None)
        if not marks:
            continue
        pts = np.asarray(marks, np.float32)
        eyes = float(np.linalg.norm(pts[1] - pts[0]))
        if eyes > max_upsize * _TEMPLATE_EYES * (size / 512.0) or eyes < 4:
            log.info("enhance: skipping a face (eye distance %.0f px)", eyes)
            continue
        m, _ = cv2.estimateAffinePartial2D(pts, template, method=cv2.LMEDS)
        if m is None:
            continue
        inv = cv2.invertAffineTransform(m)
        # Only the area the face lands on is touched: work on that crop of the picture.
        corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], np.float32) @ inv.T
        bx0, by0 = np.floor(corners.min(0)).astype(int) - 2
        bx1, by1 = np.ceil(corners.max(0)).astype(int) + 2
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
        if bx1 - bx0 < 8 or by1 - by0 < 8:
            continue
        crop = rgb[by0:by1, bx0:bx1].astype(np.float32) / np.float32(maxv)
        # crop pixel + (bx0, by0) = picture pixel, so crop -> face is m after that shift.
        shift = np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1]], np.float64)
        m_crop = m @ shift
        inv_crop = cv2.invertAffineTransform(m_crop)
        aligned = cv2.warpAffine(crop, m_crop, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        repaired = restorer.restore(aligned)
        back = cv2.warpAffine(repaired, inv_crop, (crop.shape[1], crop.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
        mask = _face_mask(size, inv_crop, crop.shape[:2], eyes)
        a = (mask * np.float32(strength))[..., None]
        blended = crop * (1.0 - a) + back * a
        rgb[by0:by1, bx0:bx1] = np.clip(np.rint(blended * np.float32(maxv)), 0, maxv).astype(rgb.dtype)
        fixed.append(FaceFix((bx0, by0, bx1, by1), eyes))
    return fixed


def _face_mask(size: int, inv: np.ndarray, shape: tuple[int, int], eyes: float) -> np.ndarray:
    """A soft-edged copy of the aligned square, warped back onto the picture."""
    square = np.zeros((size, size), np.float32)
    inset = max(2, size // 32)  # stay off the warp's reflected border
    square[inset:-inset, inset:-inset] = 1.0
    mask = cv2.warpAffine(square, inv, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
    k = max(3, int(eyes * 0.35) | 1)  # feather about a third of the eye distance
    mask = cv2.erode(mask, np.ones((k // 2 | 1, k // 2 | 1), np.uint8))
    return cv2.GaussianBlur(mask, (k, k), 0)
