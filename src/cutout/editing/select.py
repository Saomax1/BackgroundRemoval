"""Click-to-select: positive and negative clicks -> a full-resolution, edge-refined mask.

    selector = engine.click_selector(result)       # or ClickSelector(prompt_segmenter, result.image)
    selector.prepare()                              # worker thread: runs the image encoder once
    sel = selector.add_click(x, y, positive=True)   # worker thread: decoder + full-res refine
    overlay(sel.preview)                            # low-res probability, cheap to draw stretched
    editor.apply_mask(sel, MaskOp.ADD)              # "keep this" (or MaskOp.SUBTRACT: "remove this")
    selector.clear()

The prompt model (SAM 2.1) sees the image at 1024x1024 and returns 256x256
logits. To meet the "no quality loss" rule the mask is rebuilt at the photo's
own resolution, only over the area it covers: the logits are upsampled exactly
onto the full-res pixel grid, thresholded, given a narrow soft band, and then
snapped to real image edges with the same guided filter the automatic cutout
uses (``refine.guided_refine``). Nothing outside that area is computed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..core.imageio import LoadedImage
from ..core.prompt_segmenter import ImageEmbedding, PromptSegmenter
from ..core.refine import RefineOptions, guided_refine
from .rect import Rect


@dataclass(frozen=True)
class SelectOptions:
    refine_edges: bool = True
    refine: RefineOptions = RefineOptions()
    # Width of the soft band given to the thresholded mask before the guided filter,
    # as a fraction of the long side (0.001 -> 4 px on a 4000 px photo).
    feather_frac: float = 0.001
    # Feed the previous mask back to the decoder on each click, as SAM is meant to be used.
    iterative: bool = True
    # Logit above which a low-res cell counts as part of the selection.
    threshold: float = 0.0


@dataclass(frozen=True)
class Click:
    x: float
    y: float
    positive: bool = True


@dataclass
class Selection:
    """The current selection. ``mask`` covers ``rect`` only (float32 in [0, 1], full resolution)."""

    mask: np.ndarray
    rect: Rect
    score: float
    preview: np.ndarray  # float32 r x r probability for the whole image (stretched), for a quick overlay
    logits: np.ndarray = field(repr=False, default=None)  # low-res logits, fed back on the next click

    @property
    def empty(self) -> bool:
        return self.rect.empty

    def full_mask(self, height: int, width: int) -> np.ndarray:
        out = np.zeros((height, width), np.float32)
        if not self.empty:
            out[self.rect.slices] = self.mask
        return out


class ClickSelector:
    """Interactive selection on one image. Thread-safe; run prepare/add_click off the UI thread."""

    def __init__(self, segmenter: PromptSegmenter, image: LoadedImage, opts: SelectOptions = SelectOptions()):
        self.segmenter = segmenter
        self.image = image
        self.opts = opts
        self._lock = threading.RLock()
        self._embedding: ImageEmbedding | None = None
        self._clicks: list[Click] = []
        self._box: tuple[float, float, float, float] | None = None
        self._history: list[Selection] = []  # selection after each click, for undo_click

    # ------------------------------------------------------------------ state
    @property
    def ready(self) -> bool:
        return self._embedding is not None

    @property
    def clicks(self) -> tuple[Click, ...]:
        return tuple(self._clicks)

    @property
    def selection(self) -> Selection | None:
        return self._history[-1] if self._history else None

    def prepare(self) -> None:
        """Run the image encoder (slow: do it in a worker as soon as the image is loaded)."""
        with self._lock:
            if self._embedding is None:
                self._embedding = self.segmenter.encode(self.image.as_uint8())

    def clear(self) -> None:
        with self._lock:
            self._clicks.clear()
            self._history.clear()
            self._box = None

    # ------------------------------------------------------------------ prompts
    def add_click(self, x: float, y: float, positive: bool = True) -> Selection:
        """Add a click (image pixel coords, scene convention) and return the new selection."""
        with self._lock:
            self._clicks.append(Click(float(x), float(y), bool(positive)))
            sel = self._predict(self.selection)
            self._history.append(sel)
            return sel

    def set_box(self, x0: float, y0: float, x1: float, y1: float) -> Selection:
        """Start a new selection from a dragged box; clicks added afterwards refine it."""
        with self._lock:
            self._clicks.clear()
            self._box = (float(x0), float(y0), float(x1), float(y1))
            sel = self._predict(None)
            self._history = [sel]
            return sel

    def undo_click(self) -> Selection | None:
        with self._lock:
            if self._clicks:
                self._clicks.pop()
                self._history.pop()
            return self.selection

    def _predict(self, previous: Selection | None) -> Selection:
        self.prepare()
        assert self._embedding is not None
        h, w = self.image.height, self.image.width
        if not self._clicks and self._box is None:
            return _empty_selection()
        pts = np.array([[c.x, c.y] for c in self._clicks], np.float32).reshape(-1, 2)
        labels = np.array([1 if c.positive else 0 for c in self._clicks], np.int64)
        prev = previous.logits if (self.opts.iterative and previous is not None and previous.logits is not None) else None
        pred = self.segmenter.decode(self._embedding, pts, labels, box=self._box, mask_input=prev)
        n_prompts = len(self._clicks) + (2 if self._box is not None else 0)
        idx = pred.best(n_prompts)
        logits = np.asarray(pred.logits[idx], np.float32)
        mask, rect = self.full_res_mask(logits, h, w)
        preview = 1.0 / (1.0 + np.exp(-logits))
        return Selection(mask=mask, rect=rect, score=float(pred.scores[idx]), preview=preview, logits=logits)

    # ------------------------------------------------------------------ full-res mask
    def full_res_mask(self, logits: np.ndarray, height: int, width: int) -> tuple[np.ndarray, Rect]:
        """Low-res logits covering the whole (stretched) image -> refined mask over its bounding rect."""
        rh, rw = logits.shape
        on = logits > self.opts.threshold
        if not on.any():
            return np.zeros((0, 0), np.float32), Rect()
        ys, xs = np.nonzero(on)
        sx, sy = rw / width, rh / height  # full-res px -> low-res cells
        long_side = max(height, width)
        gf_radius = max(2, round(long_side * self.opts.refine.radius_frac))
        feather = max(1.0, long_side * self.opts.feather_frac)
        margin = int(np.ceil(2 / min(sx, sy))) + 3 * gf_radius + int(np.ceil(3 * feather))
        rect = Rect.from_bounds(xs.min() / sx, ys.min() / sy, (xs.max() + 1) / sx, (ys.max() + 1) / sy)
        rect = rect.grow(margin).clip(height, width)

        # Sample the logits at the centre of every full-res pixel in rect (bilinear), so the
        # crop matches what upsampling the whole matte to (width, height) would give.
        m = np.array(
            [[sx, 0.0, sx * (rect.x + 0.5) - 0.5], [0.0, sy, sy * (rect.y + 0.5) - 0.5]], np.float64
        )
        up = cv2.warpAffine(
            logits, m, (rect.w, rect.h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )
        hard = (up > self.opts.threshold).astype(np.float32)
        if not self.opts.refine_edges:
            return hard, rect
        sigma = feather / 2.0
        soft = cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        crop = self.image.rgb[rect.slices]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / np.float32(self.image.max_value)
        mask = guided_refine(gray, soft, self.opts.refine, reference_size=long_side)
        mask[mask < 1.0 / 1024] = 0.0
        mask[mask > 1.0 - 1.0 / 1024] = 1.0
        return mask, rect


def _empty_selection() -> Selection:
    return Selection(mask=np.zeros((0, 0), np.float32), rect=Rect(), score=0.0, preview=np.zeros((1, 1), np.float32))
