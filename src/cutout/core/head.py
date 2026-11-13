"""Head only: keep just the head (face, hair and anything on it), cut at the neck.

This runs *after* the automatic removal, on its matte, so the head keeps exactly the
edge quality of a normal cutout (soft hair, full resolution). Two steps:

1. **Find the face** with YuNet (OpenCV's face detector, a 230 KB ONNX file shipped
   inside the app, see ``data/``). It runs on the CPU in a few tens of milliseconds,
   so it never takes graphics memory from the cutout models. It is trained on photos
   but also finds most front-facing anime and cartoon faces; for those a lower
   confidence threshold is used. Faces off the cutout (in the background) are ignored.
2. **Cut at the neck.** For each face, a straight cut is placed a little below the
   chin, square to the face (it follows a tilted head). Of the matte above that cut,
   only the part *connected to the face* is kept: hair, hats, ears and buns stay;
   raised hands, the shoulders and the body go. The cut is feathered over a few
   pixels so it doesn't look like a scissor line.

Nothing is resized and nothing is cropped: the result is a matte of the same size,
with everything but the head(s) transparent.

When no face is found (a face turned away, a very stylised drawing), the caller can
pass a face box the user drew instead (``Face.from_box``).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import parallel
from .models import Subject

log = logging.getLogger(__name__)

FACE_MODEL = Path(__file__).with_name("data") / "face_detection_yunet_2023mar.onnx"

# The detector runs on a copy no larger than this (long side). Faces in a one-subject
# picture are big, so this costs nothing in accuracy and keeps detection ~50 ms.
DETECT_MAX_SIDE = 1280


class NoFaceFound(LookupError):
    """No face was found on the cutout, so the head can't be located automatically."""


@dataclass(frozen=True)
class Face:
    """A face box in image pixels, plus its tilt.

    ``roll`` is the angle of the eye line in radians (image coordinates, y down), so a
    head tilted towards its left shoulder has a positive roll. The box is axis-aligned
    and runs roughly from the brow to the chin.
    """

    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    roll: float = 0.0
    # Five points from the detector (subject's right eye, left eye, nose tip, right and
    # left mouth corner; i.e. image-left first), or ``None`` for a box the user drew.
    landmarks: tuple[tuple[float, float], ...] | None = field(default=None, compare=False)

    @staticmethod
    def from_box(x0: float, y0: float, x1: float, y1: float) -> "Face":
        """A face from a box the user drew (any corner order), upright."""
        return Face(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @property
    def size(self) -> tuple[float, float]:
        """(width across the face, height brow to chin), corrected for the tilt."""
        c, s = abs(math.cos(self.roll)), abs(math.sin(self.roll))
        # An axis-aligned box around a tilted face: undo the mixing of width and height.
        fw = max(min(self.w, self.h), min(max(self.w, self.h), self.w * c + self.h * s))
        fh = max(min(self.w, self.h), min(max(self.w, self.h), self.h * c + self.w * s))
        return fw, fh


@dataclass(frozen=True)
class HeadOptions:
    # Where the cut is, below the chin, as a fraction of the face height (brow to chin).
    # Under the face it is ``neck`` (a short piece of neck, so the head doesn't look sliced
    # off at the jaw); beside the neck it rises to ``side_cut``, so hair is cut about level
    # with the chin and raised shoulders (and the arms on them) stay off the head.
    neck: float = 0.2
    side_cut: float = 0.0
    # Half-width of the neck, in face widths, and how gradually the cut rises beside it.
    neck_half_width: float = 0.32
    shoulder_blend: float = 0.3
    # Below eye level nothing further than ``cheek_reach`` face widths from the middle of
    # the face is kept (hair hanging beside the face fits; shoulders pushed up beside the
    # head in a selfie, collars and hoods don't). The limit eases in from ``cheek_start``
    # face heights above the face centre (about eye level), so hair above is untouched.
    cheek_reach: float = 0.9
    cheek_start: float = 0.15
    # Softness of the cut, as a fraction of the face height (at least 1.5 px).
    feather: float = 0.03
    # How far from the face hair may reach, in face heights: sideways and upwards.
    reach_side: float = 3.0
    reach_up: float = 3.5
    # Arms raised beside the head touch the hair, so they are connected to it. A part that
    # sticks out of the head's core oval (``core_side`` x ``core_up`` face heights), rises
    # from the shoulder line to more than ``limb_reach`` face heights from the face centre
    # (above eye level) and is no thicker than an arm (``limb_width`` face heights) is taken
    # for an arm and dropped. Hair hangs down rather than rising, and big hair is thicker.
    core_side: float = 0.8
    core_up: float = 1.3
    limb_reach: float = 1.5
    limb_width: float = 0.6
    # Matte values above this count as "connected" (catches faint hair strands).
    min_alpha: float = 0.02
    # Detector confidence: photos, then drawings, then a last lower try.
    person_threshold: float = 0.6
    anime_threshold: float = 0.4
    retry_threshold: float = 0.25
    # Several people: keep every face at least this fraction of the largest face's size.
    min_relative_size: float = 0.45


@dataclass
class HeadCut:
    alpha: np.ndarray  # float32 HxW, same size as the input matte
    faces: list[Face] = field(default_factory=list)  # the faces whose heads were kept


class FaceFinder:
    """YuNet face detector. Thread-safe; the network is loaded on first use (a few ms)."""

    def __init__(self, model_path: str | Path = FACE_MODEL, opts: HeadOptions = HeadOptions()):
        self.model_path = Path(model_path)
        self.opts = opts
        self._lock = threading.Lock()
        self._det = None

    def _detector(self, w: int, h: int, threshold: float):
        if self._det is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"face detector missing: {self.model_path}")
            self._det = cv2.FaceDetectorYN.create(str(self.model_path), "", (w, h), threshold, 0.3, 5000)
        self._det.setInputSize((w, h))
        self._det.setScoreThreshold(threshold)
        return self._det

    def detect(self, rgb: np.ndarray, threshold: float) -> list[Face]:
        """Every face above ``threshold`` in an 8-bit RGB image, best first."""
        h, w = rgb.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        small = rgb
        if scale < 1.0:
            small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        bgr = np.ascontiguousarray(small[:, :, ::-1])
        with self._lock:
            det = self._detector(bgr.shape[1], bgr.shape[0], threshold)
            _, rows = det.detect(bgr)
        faces = []
        for r in rows if rows is not None else ():
            x, y, bw, bh = (float(v) / scale for v in r[:4])
            # Landmarks: right eye, left eye (the subject's), nose, mouth corners.
            rx, ry, lx, ly = (float(v) / scale for v in r[4:8])
            roll = 0.0
            if math.hypot(lx - rx, ly - ry) > 0.15 * bw:
                roll = math.atan2(ly - ry, lx - rx)
                roll = max(-math.radians(80), min(math.radians(80), roll))
            marks = tuple((float(r[4 + 2 * i]) / scale, float(r[5 + 2 * i]) / scale) for i in range(5))
            faces.append(Face(float(x), float(y), float(bw), float(bh), float(r[14]), float(roll), marks))
        faces.sort(key=lambda f: f.score, reverse=True)
        return faces

    def find(self, rgb: np.ndarray, alpha: np.ndarray | None = None,
             subject: Subject = Subject.PERSON) -> list[Face]:
        """The faces of the cut-out subject(s): on the matte, and not tiny next to the main one.

        Looks only inside the cutout's bounding box (so a small subject still gets enough
        detector pixels), first at the subject's usual confidence, then once lower.
        """
        o = self.opts
        x0 = y0 = 0
        crop = rgb
        if alpha is not None:
            ys, xs = np.nonzero(alpha[:: _step(alpha), :: _step(alpha)] > 0.5)
            if ys.size == 0:
                return []
            s = _step(alpha)
            h, w = alpha.shape
            bx0, bx1 = xs.min() * s, min(w, (xs.max() + 1) * s)
            by0, by1 = ys.min() * s, min(h, (ys.max() + 1) * s)
            m = int(0.05 * max(bx1 - bx0, by1 - by0)) + s
            x0, y0 = max(0, bx0 - m), max(0, by0 - m)
            crop = rgb[y0 : min(h, by1 + m), x0 : min(w, bx1 + m)]
        first = o.anime_threshold if Subject(subject) is Subject.ANIME else o.person_threshold
        for threshold in (first, min(first, o.retry_threshold)):
            faces = [Face(f.x + x0, f.y + y0, f.w, f.h, f.score, f.roll) for f in self.detect(crop, threshold)]
            if alpha is not None:
                faces = [f for f in faces if _on_matte(f, alpha)]
            if faces:
                biggest = max(f.w * f.h for f in faces)
                kept = [f for f in faces if f.w * f.h >= (o.min_relative_size ** 2) * biggest]
                log.info("head only: %d face(s) at confidence %.2f: %s", len(kept), threshold, describe(kept))
                return kept
            log.info("head only: no face on the cutout at confidence %.2f", threshold)
            if threshold <= o.retry_threshold:
                break
        return []


def describe(faces: list[Face]) -> str:
    """Short text for the log and debug report: box, confidence and tilt of each face."""
    return "; ".join(f"{f.w:.0f}x{f.h:.0f} at {f.x:.0f},{f.y:.0f} score {f.score:.2f} tilt "
                     f"{math.degrees(f.roll):.0f}°" for f in faces) or "none"


def _step(alpha: np.ndarray) -> int:
    """Subsampling step for quick whole-matte statistics (about 1 MP)."""
    return max(1, int(math.sqrt(alpha.size / 1_000_000)))


def _on_matte(face: Face, alpha: np.ndarray) -> bool:
    """Most of the middle of the face box is part of the cutout."""
    cx, cy = face.center
    h, w = alpha.shape
    x0, x1 = int(max(0, cx - face.w / 4)), int(min(w, cx + face.w / 4 + 1))
    y0, y1 = int(max(0, cy - face.h / 4)), int(min(h, cy + face.h / 4 + 1))
    if x1 <= x0 or y1 <= y0:
        return False
    return float(np.mean(alpha[y0:y1, x0:x1] > 0.5)) >= 0.5


def keep_head(alpha: np.ndarray, faces: list[Face], opts: HeadOptions = HeadOptions()) -> HeadCut:
    """A copy of ``alpha`` with everything but the heads of ``faces`` made transparent.

    Raises ``NoFaceFound`` if none of the faces sits on the cutout.
    """
    h, w = alpha.shape
    out = np.zeros_like(alpha, dtype=np.float32)
    kept: list[Face] = []
    for face in faces:
        roi = _head_roi(alpha, face, opts)
        if roi is None:
            continue
        (rx0, ry0, rx1, ry1), keep = roi
        region = out[ry0:ry1, rx0:rx1]
        np.maximum(region, alpha[ry0:ry1, rx0:rx1] * keep, out=region)
        kept.append(face)
    if not kept:
        raise NoFaceFound("none of the faces is on the cutout")
    return HeadCut(out, kept)


def _head_roi(alpha: np.ndarray, face: Face, o: HeadOptions):
    """(roi box, keep factor in [0, 1] over the box) for one face, or None if it isn't on the matte."""
    h, w = alpha.shape
    fw, fh = face.size
    cx, cy = face.center
    ex, ey = math.cos(face.roll), math.sin(face.roll)  # along the eye line
    dx, dy = -ey, ex  # down the face, chin-wards
    # Distance of the cut below the face centre: deepest under the chin (the neck), rising
    # to ``side_cut`` beside it with a smooth (cosine) shoulder.
    cut_mid, cut_side = fh / 2 + o.neck * fh, fh / 2 + o.side_cut * fh
    n0, n1 = o.neck_half_width * fw, (o.neck_half_width + o.shoulder_blend) * fw
    cut = max(cut_mid, cut_side)
    feather = max(1.5, o.feather * fh)
    cheek, cheek_top = o.cheek_reach * fw, o.cheek_start * fh
    side, up, low = o.reach_side * fh, o.reach_up * fh, cut + feather

    # Axis-aligned box around the (possibly tilted) search area.
    corners = [(cx + a * ex + b * dx, cy + a * ey + b * dy) for a in (-side, side) for b in (-up, low)]
    rx0 = max(0, int(math.floor(min(p[0] for p in corners))))
    rx1 = min(w, int(math.ceil(max(p[0] for p in corners))) + 1)
    ry0 = max(0, int(math.floor(min(p[1] for p in corners))))
    ry1 = min(h, int(math.ceil(max(p[1] for p in corners))) + 1)
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    a = alpha[ry0:ry1, rx0:rx1]
    rh, rw = a.shape

    ramp = np.empty((rh, rw), np.float32)  # 1 above the cut, fading to 0 across it
    binary = np.empty((rh, rw), np.uint8)  # matte above the cut, inside the search area
    zone = np.empty((rh, rw), np.uint8)  # 0 core oval, 1 elsewhere, 2 shoulder line, 4 far up
    cs, cu = o.core_side * fh, o.core_up * fh
    band = 0.12 * fh
    reach2 = (o.limb_reach * fh) ** 2
    xs = (np.arange(rx0, rx1, dtype=np.float32) + 0.5 - cx)[None, :]

    def strip(y0: int, y1: int) -> None:
        ys = (np.arange(ry0 + y0, ry0 + y1, dtype=np.float32) + 0.5 - cy)[:, None]
        along = xs * ex + ys * ey
        down = xs * dx + ys * dy
        t = np.clip((np.abs(along) - n0) / max(n1 - n0, 1e-6), 0.0, 1.0)
        depth = cut_side + (cut_mid - cut_side) * (0.5 + 0.5 * np.cos(np.pi * t))
        r = np.clip(0.5 - (down - depth) / feather, 0.0, 1.0)
        # Sideways limit below the eyes: widens smoothly to no limit higher up.
        lift = np.clip((down + cheek_top) / max(cheek_top, 1e-6), 0.0, 1.0)  # 0 above, 1 from centre down
        reach = cheek + (side - cheek) * (1.0 - lift)
        r = np.minimum(r, np.clip(0.5 - (np.abs(along) - reach) / feather, 0.0, 1.0))
        ramp[y0:y1] = r
        inside = (np.abs(along) <= side) & (down >= -up) & (r > 0)
        binary[y0:y1] = inside & (a[y0:y1] > o.min_alpha)
        vert = np.where(down < 0, cu, cut)
        outside = (along / cs) ** 2 + (down / vert) ** 2 > 1.0
        # Where the body is cut off: the shoulder line, and the sideways limit below the eyes.
        shoulder = ((down >= depth - band) & (np.abs(along) > n0)) | (
            (lift > 0) & (np.abs(along) >= reach - band))
        far_up = (down < 0) & (along * along + down * down > reach2)
        zone[y0:y1] = outside * (1 + 2 * shoulder + 4 * far_up).astype(np.uint8)

    parallel.map_rows(strip, rh, rw)

    n, labels = cv2.connectedComponents(binary, connectivity=8)
    # The face's own component(s): whatever the middle of the face box touches.
    sx0, sx1 = int(max(rx0, cx - fw / 4)) - rx0, int(min(rx1, cx + fw / 4 + 1)) - rx0
    sy0, sy1 = int(max(ry0, cy - fh / 4)) - ry0, int(min(ry1, cy + fh / 4 + 1)) - ry0
    seed = labels[max(0, sy0) : max(0, sy1), max(0, sx0) : max(0, sx1)]
    ids = np.unique(seed[seed > 0])
    if ids.size == 0:
        log.info("face at %.0f,%.0f is not on the cutout; skipped", cx, cy)
        return None
    if n - 1 == ids.size:
        comp = binary  # everything above the cut is the head: skip the lookup
    else:
        lut = np.zeros(n, np.uint8)
        lut[ids] = 1
        comp = lut[labels]
    del labels
    _drop_limbs(comp, zone, o.limb_width * fh / 2)
    del zone
    # Grow a little so the faint fringe just under min_alpha stays with the hair.
    r = max(2, round(0.01 * fh))
    comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    def finish(y0: int, y1: int) -> None:
        ramp[y0:y1] *= comp[y0:y1]

    parallel.map_rows(finish, rh, rw)
    return (rx0, ry0, rx1, ry1), ramp


def _drop_limbs(comp: np.ndarray, zone: np.ndarray, max_half_width: float) -> None:
    """Clear (in place) the arm-thin parts of ``comp`` outside the core oval that run from
    the shoulder line to far above it: arms raised beside the head."""
    outside = ((comp > 0) & (zone > 0)).view(np.uint8)
    n, labels = cv2.connectedComponents(outside, connectivity=8)
    if n <= 1:
        return
    flat = labels.ravel()
    z = zone.ravel()
    on_shoulder = np.bincount(flat, weights=(z & 2) > 0, minlength=n) > 0
    far = np.bincount(flat, weights=(z & 4) > 0, minlength=n) > 0
    limb = on_shoulder & far
    limb[0] = False
    if limb.any():
        # Thicker than an arm somewhere: a mass of hair, not a limb.
        inscribed = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        limb[np.unique(labels[inscribed > max_half_width])] = False
    if limb.any():
        log.info("dropped %d arm-like part(s) next to the head", int(limb.sum()))
        comp[limb[labels]] = 0


def head_only(
    rgb: np.ndarray,
    alpha: np.ndarray,
    subject: Subject = Subject.PERSON,
    faces: list[Face] | None = None,
    finder: FaceFinder | None = None,
    opts: HeadOptions = HeadOptions(),
) -> HeadCut:
    """Find the face(s) on the cutout (unless ``faces`` is given) and keep only the head(s).

    ``rgb`` is the 8-bit image (``LoadedImage.as_uint8()``), ``alpha`` its full-res matte.
    Raises ``NoFaceFound`` when no face is found on the cutout.
    """
    if faces is None:
        faces = (finder or FaceFinder(opts=opts)).find(rgb, alpha, subject)
        if not faces:
            raise NoFaceFound("no face found on the cutout")
    return keep_head(alpha, faces, opts)
