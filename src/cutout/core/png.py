"""Lossless PNG encoding on every CPU core.

Pillow and OpenCV compress a PNG on one core, which for a 24 MP cutout takes
several seconds: by far the slowest CPU step of a save. This encoder does the
same work (adaptive per-row filters, zlib deflate at level 6, as libpng does) in horizontal strips
on all cores, the way ``pigz`` parallelises gzip:

* Each strip's rows are filtered independently (a PNG filter only looks at the row
  above, which is raw image data, so strips never wait on each other).
* Each strip is deflated on its own, primed with the last 32 KB of the previous
  strip as a dictionary so compression stays as good as one long stream, and ended
  with a sync flush so the pieces join into one valid deflate stream.
* The Adler-32 checksums of the pieces are combined into the one for the whole stream.

The pixels are stored exactly; any PNG reader decodes the same array the app
passed in. 8 and 16 bit RGBA are supported, and the ICC profile is embedded for both.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from . import parallel

SIGNATURE = b"\x89PNG\r\n\x1a\n"
LEVEL = 6  # what Pillow used before: a good size/speed balance
_WINDOW = 32 * 1024  # deflate's dictionary size
_IDAT_SIZE = 1 << 20
_STRIP_PIXELS = 512 * 1024
# libpng's choice for filtered rows; a few percent smaller than the default strategy.
_STRATEGY = zlib.Z_FILTERED
_ADLER_BASE = 65521
# |b| of each byte read as a signed int8: the filter-choice cost.
_ABS_INT8 = np.abs(np.arange(256, dtype=np.uint8).view(np.int8).astype(np.int16)).astype(np.uint8)


def encode_png(rgba: np.ndarray, icc_profile: bytes | None = None, level: int = LEVEL) -> bytes:
    """``H x W x 4`` RGBA or ``H x W x 3`` RGB (uint8 or uint16) -> PNG file bytes."""
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx4 RGBA or HxWx3 RGB, got shape {rgba.shape}")
    channels = rgba.shape[2]
    if rgba.dtype == np.uint8:
        depth, rows = 8, np.ascontiguousarray(rgba)
    elif rgba.dtype == np.uint16:
        depth, rows = 16, rgba.astype(">u2")  # PNG stores 16-bit samples big-endian
    else:
        raise TypeError(f"unsupported dtype {rgba.dtype}")
    h, w = rgba.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("cannot encode an empty image")
    raw = rows.reshape(h, -1).view(np.uint8)  # one row of bytes per scanline
    bpp = channels * depth // 8

    # Small strips bound the filter's temporary memory (about 20 bytes per input byte).
    parts = parallel.strips(h, w, max_pixels=_STRIP_PIXELS)

    def strip(y0: int, y1: int) -> tuple[bytes, int, int]:
        data = _filter_rows(raw, y0, y1, bpp).tobytes()
        if y0 > 0:
            # Prime with the end of the previous strip's filtered bytes: what a
            # single-stream encoder would still have in its window at this point.
            back = -(-_WINDOW // (raw.shape[1] + 1))
            prev = _filter_rows(raw, max(0, y0 - back), y0, bpp).tobytes()[-_WINDOW:]
            comp = zlib.compressobj(level, zlib.DEFLATED, -15, 9, _STRATEGY, prev)
        else:
            comp = zlib.compressobj(level, zlib.DEFLATED, -15, 9, _STRATEGY)
        end = zlib.Z_FINISH if y1 == h else zlib.Z_SYNC_FLUSH
        return comp.compress(data) + comp.flush(end), zlib.adler32(data), len(data)

    pieces = parallel.map_parts(strip, parts)

    adler = 1
    for _, a, n in pieces:
        adler = adler32_combine(adler, a, n)
    stream = b"".join([b"\x78\x9c", *(p for p, _, _ in pieces), struct.pack(">I", adler)])

    out = [SIGNATURE, _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, 6 if channels == 4 else 2, 0, 0, 0))]
    if icc_profile:
        out.append(_chunk(b"iCCP", b"ICC Profile\x00\x00" + zlib.compress(icc_profile)))
    for i in range(0, len(stream), _IDAT_SIZE):
        out.append(_chunk(b"IDAT", stream[i : i + _IDAT_SIZE]))
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _filter_rows(raw: np.ndarray, y0: int, y1: int, bpp: int) -> np.ndarray:
    """PNG-filter scanlines ``y0:y1``: per row, the filter with the smallest sum of
    absolute signed bytes (the heuristic libpng and Pillow use). Returns rows with
    their filter-type byte in front, as uint8 ``(y1 - y0) x (1 + rowbytes)``."""
    x = raw[y0:y1].astype(np.int16)
    up = np.zeros_like(x)
    if y0 > 0:
        up[0] = raw[y0 - 1]
    up[1:] = x[:-1]
    left = np.zeros_like(x)
    left[:, bpp:] = x[:, :-bpp]
    upleft = np.zeros_like(x)
    upleft[:, bpp:] = up[:, :-bpp]

    # Paeth predictor (PNG spec 9.4), vectorised.
    pa = np.abs(up - upleft)
    pb = np.abs(left - upleft)
    pc = np.abs(left + up - 2 * upleft)
    paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, upleft))
    del pa, pb, pc

    candidates = np.stack(
        [x, x - left, x - up, x - ((left + up) >> 1), x - paeth]
    ).astype(np.uint8)  # mod 256, as the spec defines
    del up, left, upleft, paeth
    cost = _ABS_INT8[candidates].sum(axis=2, dtype=np.int64)
    best = np.argmin(cost, axis=0)  # first minimum wins ties, like libpng (None < Sub < ...)

    n = y1 - y0
    out = np.empty((n, raw.shape[1] + 1), np.uint8)
    out[:, 0] = best
    out[:, 1:] = candidates[best, np.arange(n)]
    return out


def adler32_combine(adler1: int, adler2: int, len2: int) -> int:
    """Adler-32 of ``A + B`` from the checksums of ``A`` and ``B`` (zlib's ``adler32_combine``)."""
    rem = len2 % _ADLER_BASE
    sum1 = adler1 & 0xFFFF
    sum2 = (rem * sum1) % _ADLER_BASE
    sum1 += (adler2 & 0xFFFF) + _ADLER_BASE - 1
    sum2 += ((adler1 >> 16) & 0xFFFF) + ((adler2 >> 16) & 0xFFFF) + _ADLER_BASE - rem
    if sum1 >= _ADLER_BASE:
        sum1 -= _ADLER_BASE
    if sum1 >= _ADLER_BASE:
        sum1 -= _ADLER_BASE
    if sum2 >= (_ADLER_BASE << 1):
        sum2 -= _ADLER_BASE << 1
    if sum2 >= _ADLER_BASE:
        sum2 -= _ADLER_BASE
    return sum1 | (sum2 << 16)
