"""Image loading and saving with zero quality loss.

Rules this module guarantees (the rest of the app relies on them):

* The pixel grid of the output is exactly the pixel grid of the input as it is
  *displayed*: EXIF orientation is applied once at load time, nothing is ever
  resized, cropped or re-sampled.
* Bit depth is preserved: 8-bit in -> 8-bit RGBA PNG out, 16-bit in -> 16-bit
  RGBA PNG out.
* The embedded ICC colour profile is carried through to the PNG (8-bit path).
* Paths may contain any Unicode characters (OpenCV's own ``imread`` cannot do
  this on Windows, so we always decode from bytes).
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jfif"}
)

_EXIF_ORIENTATION_TAG = 0x0112


@dataclass
class LoadedImage:
    """An RGB image exactly as the user sees it, plus what is needed to save it back.

    ``rgb`` is ``H x W x 3`` in RGB order, dtype ``uint8`` or ``uint16``.
    Any alpha channel in the source file is intentionally dropped: we compute a new one.
    """

    rgb: np.ndarray
    icc_profile: bytes | None = None
    source_path: Path | None = None

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def max_value(self) -> int:
        return 65535 if self.rgb.dtype == np.uint16 else 255

    def as_float(self) -> np.ndarray:
        """RGB as float32 in [0, 1]."""
        return self.rgb.astype(np.float32) / np.float32(self.max_value)

    def as_uint8(self) -> np.ndarray:
        """RGB as uint8 (what the segmentation models consume)."""
        if self.rgb.dtype == np.uint8:
            return self.rgb
        return np.round(self.rgb.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)


def load_image(path: str | os.PathLike) -> LoadedImage:
    path = Path(path)
    data = path.read_bytes()

    with Image.open(io.BytesIO(data)) as probe:
        icc = probe.info.get("icc_profile")
        orientation = probe.getexif().get(_EXIF_ORIENTATION_TAG, 1)
        mode = probe.mode
        fmt = probe.format

    # Pillow silently truncates 48/64-bit PNG and 16-bit TIFF to 8 bits, so let
    # OpenCV decode those formats and keep its result when it really is 16-bit.
    decoded = None
    if fmt in ("PNG", "TIFF"):
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is not None and decoded.dtype == np.uint16:
        rgb = _cv_to_rgb(decoded)
        rgb = _apply_orientation(rgb, orientation)
    else:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            rgb = np.asarray(im, dtype=np.uint8).copy()

    if mode == "CMYK":
        icc = None  # a CMYK profile would be wrong for the RGB output
    return LoadedImage(rgb=np.ascontiguousarray(rgb), icc_profile=icc, source_path=path)


def save_png(path: str | os.PathLike, rgba: np.ndarray, icc_profile: bytes | None = None) -> None:
    """Write an ``H x W x 4`` RGBA array (uint8 or uint16) as a PNG, atomically."""
    path = Path(path)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected HxWx4 RGBA, got shape {rgba.shape}")

    if rgba.dtype == np.uint8:
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG", icc_profile=icc_profile, compress_level=6)
        payload = buf.getvalue()
    elif rgba.dtype == np.uint16:
        # Pillow cannot write 16-bit RGBA; OpenCV can (ICC is not carried on this path).
        ok, enc = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        if not ok:
            raise OSError(f"PNG encoding failed for {path}")
        payload = enc.tobytes()
    else:
        raise TypeError(f"unsupported dtype {rgba.dtype}")

    _atomic_write(path, payload)


def _cv_to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    channels = arr.shape[2]
    if channels == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    if channels == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if channels == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"unsupported channel count {channels}")


def _apply_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation value (1-8) to an HxWxC array."""
    ops = {
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1, :],
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, k=-1),
        7: lambda a: np.swapaxes(a, 0, 1)[::-1, ::-1],
        8: lambda a: np.rot90(a, k=1),
    }
    op = ops.get(int(orientation))
    return np.ascontiguousarray(op(arr)) if op else arr


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".cutout-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
