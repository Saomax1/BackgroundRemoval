"""The matte editor: every touch-up changes the full-resolution alpha through here.

    editor = MatteEditor(result.alpha, result.image)    # edits result.alpha in place

    # brush (cheap enough for the UI thread: each call touches only the dab area)
    dirty = editor.begin_stroke(x, y, Brush(BrushMode.ERASE, radius=30))
    dirty = editor.continue_stroke(x2, y2)              # -> repaint `dirty`
    patch = editor.end_stroke()                         # -> one undo step

    # click-to-select result (see select.py)
    patch = editor.apply_mask(selection, MaskOp.ADD)

    editor.undo(); editor.redo()                        # built-in history, or
    patch.before / patch.after                          # hand the patch to a QUndoStack

Changes are recorded as 256x256 tiles, never whole copies, so a long brush
stroke across a 24 MP photo costs only the tiles it crossed. The matte is never
resized, and pixels outside an edit are never written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ..core.imageio import LoadedImage
from ..core.refine import RefineOptions, guided_filter
from .brush import Brush, BrushMode, dab_centres, dab_rect, stamp
from .rect import Rect

TILE = 256

TileKey = tuple[int, int]  # (row, col) in tiles


class MaskOp(str, Enum):
    ADD = "add"  # selected area becomes subject: alpha = max(alpha, mask)
    SUBTRACT = "subtract"  # selected area becomes background: alpha = min(alpha, 1 - mask)
    REPLACE = "replace"  # alpha = mask inside the rect (used by "restore" style tools)


@dataclass
class Patch:
    """One undoable change: the tiles it touched, holding the values *not* currently in the matte.

    Right after an edit the tiles hold the "before" values; ``undo`` swaps them into the
    matte (so they then hold "after"), ``redo`` swaps back. ``before``/``after`` give
    rect-shaped copies for undo systems that want plain arrays (Qt's QUndoStack).
    """

    label: str
    rect: Rect
    tiles: dict[TileKey, np.ndarray] = field(default_factory=dict)
    _alpha: np.ndarray | None = field(default=None, repr=False)
    _applied: bool = True  # True: matte shows "after", tiles hold "before"

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tiles.values())

    @property
    def empty(self) -> bool:
        return not self.tiles

    def _materialise(self, want_before: bool) -> np.ndarray:
        assert self._alpha is not None
        out = self._alpha[self.rect.slices].copy()
        tiles_hold_before = self._applied
        if want_before == tiles_hold_before:
            for key, tile in self.tiles.items():
                r = _tile_rect(key, self._alpha.shape).intersect(self.rect)
                tr = _tile_rect(key, self._alpha.shape)
                out[r.y - self.rect.y : r.y1 - self.rect.y, r.x - self.rect.x : r.x1 - self.rect.x] = tile[
                    r.y - tr.y : r.y1 - tr.y, r.x - tr.x : r.x1 - tr.x
                ]
        return out

    @property
    def before(self) -> np.ndarray:
        return self._materialise(True)

    @property
    def after(self) -> np.ndarray:
        return self._materialise(False)

    def _swap(self) -> Rect:
        assert self._alpha is not None
        for key, tile in self.tiles.items():
            sl = _tile_rect(key, self._alpha.shape).slices
            current = self._alpha[sl].copy()
            self._alpha[sl] = tile
            self.tiles[key] = current
        self._applied = not self._applied
        return self.rect

    def undo(self) -> Rect:
        if not self._applied:
            raise RuntimeError("patch already undone")
        return self._swap()

    def redo(self) -> Rect:
        if self._applied:
            raise RuntimeError("patch already applied")
        return self._swap()


class EditHistory:
    """Undo/redo of Patches, bounded by memory and step count (oldest steps drop first)."""

    def __init__(self, max_bytes: int = 1 << 30, max_steps: int = 200):
        self.max_bytes = max_bytes
        self.max_steps = max_steps
        self._undo: list[Patch] = []
        self._redo: list[Patch] = []

    def push(self, patch: Patch) -> None:
        if patch.empty:
            return
        self._undo.append(patch)
        self._redo.clear()
        while self._undo and (len(self._undo) > self.max_steps or self.nbytes > self.max_bytes):
            self._undo.pop(0)

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self._undo) + sum(p.nbytes for p in self._redo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str | None:
        return self._undo[-1].label if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1].label if self._redo else None

    def undo(self) -> Rect | None:
        if not self._undo:
            return None
        patch = self._undo.pop()
        self._redo.append(patch)
        return patch.undo()

    def redo(self) -> Rect | None:
        if not self._redo:
            return None
        patch = self._redo.pop()
        self._undo.append(patch)
        return patch.redo()

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


@dataclass
class _Stroke:
    brush: Brush
    label: str
    target: float
    last: tuple[float, float, float]  # x, y, pressure
    carry: float = 0.0
    rect: Rect = field(default_factory=Rect)
    coverage: dict[TileKey, np.ndarray] = field(default_factory=dict)
    refine_target: dict[TileKey, np.ndarray] = field(default_factory=dict)


class MatteEditor:
    """Edits a float32 full-resolution alpha matte in place, with tile-based undo.

    Not thread-safe: call it from one thread (the UI thread is fine; each brush call
    only touches the dab's pixels). ``image`` is needed only for the REFINE brush.
    Set ``history=None`` when the caller keeps its own undo stack and just wants the
    returned Patches.
    """

    def __init__(
        self,
        alpha: np.ndarray,
        image: LoadedImage | None = None,
        history: EditHistory | None | bool = True,
        refine: RefineOptions = RefineOptions(),
    ):
        if alpha.dtype != np.float32 or alpha.ndim != 2:
            raise ValueError("alpha must be a 2-D float32 array")
        if not alpha.flags.c_contiguous or not alpha.flags.writeable:
            raise ValueError("alpha must be a writeable C-contiguous array (it is edited in place)")
        if image is not None and (image.height, image.width) != alpha.shape:
            raise ValueError(f"alpha {alpha.shape} does not match image {(image.height, image.width)}")
        self.alpha = alpha
        self.image = image
        self.refine = refine
        self.history: EditHistory | None = EditHistory() if history is True else (history or None)
        self._original = alpha.copy()  # the automatic result, for "restore"
        self._snap: dict[TileKey, np.ndarray] = {}  # pre-edit tiles of the edit in progress
        self._stroke: _Stroke | None = None
        self._gray: np.ndarray | None = None

    # ------------------------------------------------------------------ geometry
    @property
    def height(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def width(self) -> int:
        return int(self.alpha.shape[1])

    @property
    def bounds(self) -> Rect:
        return Rect(0, 0, self.width, self.height)

    @property
    def is_modified(self) -> bool:
        return not np.array_equal(self.alpha, self._original)

    @property
    def stroke_active(self) -> bool:
        return self._stroke is not None

    # ------------------------------------------------------------------ brush
    def begin_stroke(self, x: float, y: float, brush: Brush, pressure: float = 1.0, label: str | None = None) -> Rect:
        """Start a stroke and lay the first dab. Returns the area to repaint."""
        if self._stroke is not None:
            self.end_stroke()
        if brush.mode is BrushMode.REFINE and self.image is None:
            raise ValueError("the refine brush needs the image (MatteEditor(alpha, image))")
        target = {BrushMode.KEEP: 1.0, BrushMode.ERASE: 0.0, BrushMode.REFINE: float("nan")}[brush.mode]
        self._stroke = _Stroke(
            brush=brush,
            label=label or {BrushMode.KEEP: "Keep brush", BrushMode.ERASE: "Erase brush", BrushMode.REFINE: "Refine brush"}[brush.mode],
            target=target,
            last=(float(x), float(y), float(pressure)),
        )
        return self._dab(float(x), float(y), float(pressure))

    def continue_stroke(self, x: float, y: float, pressure: float = 1.0) -> Rect:
        """Extend the stroke to ``(x, y)``, laying evenly spaced dabs. Returns the area to repaint."""
        s = self._stroke
        if s is None:
            raise RuntimeError("continue_stroke without begin_stroke")
        x0, y0, p0 = s.last
        step = max(0.5, s.brush.spacing * min(s.brush.dab_radius(p0), s.brush.dab_radius(pressure)))
        centres, s.carry = dab_centres(x0, y0, float(x), float(y), step, s.carry)
        dirty = Rect()
        for cx, cy, t in centres:
            dirty = dirty.union(self._dab(cx, cy, p0 + (pressure - p0) * t))
        s.last = (float(x), float(y), float(pressure))
        return dirty

    def end_stroke(self) -> Patch | None:
        """Finish the stroke and record it as one undo step (returned; None if nothing changed)."""
        s = self._stroke
        if s is None:
            return None
        self._stroke = None
        return self._commit(s.label, s.rect)

    def cancel_stroke(self) -> Rect:
        """Throw the stroke away (Esc mid-drag). Returns the area to repaint."""
        s = self._stroke
        if s is None:
            return Rect()
        self._stroke = None
        self._restore_snapshot()
        return s.rect

    def _dab(self, cx: float, cy: float, pressure: float) -> Rect:
        s = self._stroke
        assert s is not None
        b = s.brush
        radius = b.dab_radius(pressure)
        flow = np.float32(b.dab_flow(pressure))
        rect = dab_rect(cx, cy, radius).clip(self.height, self.width)
        if rect.empty or flow <= 0:
            return Rect()
        weights = stamp(cx, cy, radius, b.hardness, rect) * flow
        self._snapshot(rect)
        if b.mode is BrushMode.REFINE:
            refined = self._refined_region(rect)

        for key in _tiles_in(rect):
            tr = _tile_rect(key, self.alpha.shape)
            r = tr.intersect(rect)
            in_tile = (slice(r.y - tr.y, r.y1 - tr.y), slice(r.x - tr.x, r.x1 - tr.x))
            in_dab = (slice(r.y - rect.y, r.y1 - rect.y), slice(r.x - rect.x, r.x1 - rect.x))
            cov = s.coverage.get(key)
            if cov is None:
                cov = s.coverage[key] = np.zeros((tr.h, tr.w), np.float32)
            np.maximum(cov[in_tile], weights[in_dab], out=cov[in_tile])
            before = self._snap[key][in_tile]
            if b.mode is BrushMode.REFINE:
                tgt = s.refine_target.get(key)
                if tgt is None:
                    tgt = s.refine_target[key] = self._snap[key].copy()
                tgt[in_tile] = refined[in_dab]
                target = tgt[in_tile]
            else:
                target = np.float32(s.target)
            self.alpha[r.slices] = before + (target - before) * cov[in_tile]
        s.rect = s.rect.union(rect)
        return rect

    def _refined_region(self, rect: Rect) -> np.ndarray:
        """Guided-filter the pre-stroke matte over ``rect`` (with margin), steered by the photo."""
        long_side = max(self.height, self.width)
        brush = self._stroke.brush if self._stroke else None
        # At least the auto-refine radius; bigger brushes reach further to find the real edge.
        radius = max(2, round(long_side * self.refine.radius_frac), round((brush.radius if brush else 0) / 4))
        outer = rect.grow(2 * radius).clip(self.height, self.width)
        before = self._pre_edit_region(outer)
        gray = self._gray_region(outer)
        filtered = np.clip(guided_filter(gray, before, radius, self.refine.eps), 0.0, 1.0)
        oy, ox = rect.y - outer.y, rect.x - outer.x
        return filtered[oy : oy + rect.h, ox : ox + rect.w]

    def _gray_region(self, rect: Rect) -> np.ndarray:
        assert self.image is not None
        if self._gray is None:
            img = self.image
            self._gray = cv2.cvtColor(img.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / np.float32(img.max_value)
        return self._gray[rect.slices]

    # ------------------------------------------------------------------ masks
    def apply_mask(self, mask, op: MaskOp = MaskOp.ADD, rect: Rect | None = None, label: str | None = None) -> Patch | None:
        """Merge a float32 mask in [0, 1] into the matte and record one undo step.

        ``mask`` is either a full-size array, an array covering ``rect``, or anything with
        ``.mask`` and ``.rect`` (a click-to-select ``Selection``).
        """
        if self._stroke is not None:
            self.end_stroke()
        if hasattr(mask, "mask") and hasattr(mask, "rect"):
            mask, rect = mask.mask, mask.rect
        mask = np.asarray(mask, np.float32)
        if rect is None:
            rect = Rect(0, 0, mask.shape[1], mask.shape[0])
        if mask.shape != (rect.h, rect.w):
            raise ValueError(f"mask {mask.shape} does not match rect {(rect.h, rect.w)}")
        clipped = rect.clip(self.height, self.width)
        if clipped.empty:
            return None
        mask = mask[clipped.y - rect.y : clipped.y1 - rect.y, clipped.x - rect.x : clipped.x1 - rect.x]
        mask = np.clip(mask, 0.0, 1.0)

        self._snapshot(clipped)
        cur = self.alpha[clipped.slices]
        if op is MaskOp.ADD:
            np.maximum(cur, mask, out=cur)
        elif op is MaskOp.SUBTRACT:
            np.minimum(cur, 1.0 - mask, out=cur)
        else:
            cur[...] = mask
        default = {MaskOp.ADD: "Add selection", MaskOp.SUBTRACT: "Remove selection", MaskOp.REPLACE: "Replace"}[op]
        return self._commit(label or default, clipped)

    def restore_original(self, rect: Rect | None = None) -> Patch | None:
        """Put the automatic result back (whole image, or just ``rect``) as one undo step."""
        rect = (rect or self.bounds).clip(self.height, self.width)
        return self.apply_mask(self._original[rect.slices], MaskOp.REPLACE, rect, label="Restore automatic cutout")

    # ------------------------------------------------------------------ history
    def undo(self) -> Rect | None:
        if self._stroke is not None:
            self.end_stroke()
        return self.history.undo() if self.history else None

    def redo(self) -> Rect | None:
        return self.history.redo() if self.history else None

    @property
    def can_undo(self) -> bool:
        return bool(self.history and self.history.can_undo)

    @property
    def can_redo(self) -> bool:
        return bool(self.history and self.history.can_redo)

    # ------------------------------------------------------------------ tiles
    def _snapshot(self, rect: Rect) -> None:
        for key in _tiles_in(rect):
            if key not in self._snap:
                self._snap[key] = self.alpha[_tile_rect(key, self.alpha.shape).slices].copy()

    def _pre_edit_region(self, rect: Rect) -> np.ndarray:
        out = self.alpha[rect.slices].copy()
        for key in _tiles_in(rect):
            tile = self._snap.get(key)
            if tile is None:
                continue
            tr = _tile_rect(key, self.alpha.shape)
            r = tr.intersect(rect)
            out[r.y - rect.y : r.y1 - rect.y, r.x - rect.x : r.x1 - rect.x] = tile[r.y - tr.y : r.y1 - tr.y, r.x - tr.x : r.x1 - tr.x]
        return out

    def _restore_snapshot(self) -> None:
        for key, tile in self._snap.items():
            self.alpha[_tile_rect(key, self.alpha.shape).slices] = tile
        self._snap = {}

    def _commit(self, label: str, rect: Rect) -> Patch | None:
        changed = {}
        for key, before in self._snap.items():
            if not np.array_equal(before, self.alpha[_tile_rect(key, self.alpha.shape).slices]):
                changed[key] = before
        self._snap = {}
        if not changed:
            return None
        tight = Rect()
        for key in changed:
            tight = tight.union(_tile_rect(key, self.alpha.shape))
        patch = Patch(label=label, rect=tight.intersect(rect) if not rect.empty else tight, tiles=changed, _alpha=self.alpha)
        if self.history is not None:
            self.history.push(patch)
        return patch


def _tile_rect(key: TileKey, shape: tuple[int, ...]) -> Rect:
    ty, tx = key
    y, x = ty * TILE, tx * TILE
    return Rect(x, y, min(TILE, shape[1] - x), min(TILE, shape[0] - y))


def _tiles_in(rect: Rect):
    if rect.empty:
        return
    for ty in range(rect.y // TILE, (rect.y1 - 1) // TILE + 1):
        for tx in range(rect.x // TILE, (rect.x1 - 1) // TILE + 1):
            yield (ty, tx)
