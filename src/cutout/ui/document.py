"""The open image, its alpha matte, and the undo history.

Every edit to the matte goes through this object so the canvas, the undo stack
and the "unsaved changes" state stay in sync:

* A tool that edits a rectangle calls :meth:`CutoutDocument.push_patch` (or
  edits ``doc.alpha`` in place during a drag, calls
  :meth:`notify_alpha_changed` for live feedback, and finishes the stroke with
  :meth:`push_patch` passing the ``before`` copy it took).
* Undo stores only the changed rectangle, before and after, never a full copy
  of the matte (a 24 MP matte is 96 MB). Re-running the automatic removal is
  the one full-size undo step.

The matte is never resized: it is always ``image.height x image.width`` float32 in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRect, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..core.engine import CutoutResult
from ..core.imageio import LoadedImage
from ..core.refine import compose_rgba

UNDO_LIMIT = 40


class CutoutDocument(QObject):
    image_changed = Signal()  # a new image was opened (or the document was cleared)
    alpha_changed = Signal(QRect)  # this image-space rectangle of the matte changed
    result_changed = Signal()  # a new automatic result arrived (model / device / timings)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.image: LoadedImage | None = None
        self.alpha: np.ndarray | None = None
        self.result: CutoutResult | None = None
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(UNDO_LIMIT)

    # ----------------------------------------------------------------- state
    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_alpha(self) -> bool:
        return self.alpha is not None

    @property
    def path(self) -> Path | None:
        return self.image.source_path if self.image is not None else None

    @property
    def is_modified(self) -> bool:
        """True when there is a cutout the user has not saved since it last changed."""
        return self.has_alpha and not self.undo_stack.isClean()

    def full_rect(self) -> QRect:
        if self.image is None:
            return QRect()
        return QRect(0, 0, self.image.width, self.image.height)

    # ------------------------------------------------------------- loading
    def set_image(self, image: LoadedImage) -> None:
        self.image = image
        self.alpha = None
        self.result = None
        self.undo_stack.clear()
        self.image_changed.emit()

    def clear(self) -> None:
        self.image = None
        self.alpha = None
        self.result = None
        self.undo_stack.clear()
        self.image_changed.emit()

    def set_auto_result(self, result: CutoutResult) -> None:
        """Install the matte from an automatic removal run. Undoable if it replaces one."""
        if self.image is None or result.alpha.shape != (self.image.height, self.image.width):
            raise ValueError("result does not belong to the open image")
        alpha = np.ascontiguousarray(result.alpha, dtype=np.float32)
        self.result = result
        if self.alpha is None:
            self.alpha = alpha
            self.undo_stack.clear()
            # Clean index -1 marks "never saved", so a fresh cutout counts as unsaved work.
            self.undo_stack.resetClean()
            self.alpha_changed.emit(self.full_rect())
        else:
            self.undo_stack.push(_ReplaceAlphaCommand(self, alpha, "Automatic removal"))
        self.result_changed.emit()

    # ------------------------------------------------------------- editing
    def notify_alpha_changed(self, rect: QRect) -> None:
        """Tell views that ``rect`` of ``alpha`` was edited in place (no undo entry)."""
        rect = rect.intersected(self.full_rect())
        if not rect.isEmpty():
            self.alpha_changed.emit(rect)

    def push_patch(
        self,
        rect: QRect,
        after: np.ndarray,
        before: np.ndarray | None = None,
        text: str = "Edit",
    ) -> None:
        """Record an edit of ``rect`` as one undo step and apply it.

        ``after`` is the new ``rect.height() x rect.width()`` float32 matte region.
        If the tool already wrote ``after`` into ``alpha`` during a drag, pass the
        region's original values as ``before``; otherwise ``before`` is read here.
        """
        if self.alpha is None:
            raise RuntimeError("no matte to edit yet")
        rect = QRect(rect)
        if rect != rect.intersected(self.full_rect()):
            raise ValueError(f"patch {rect} is outside the image {self.full_rect()}")
        shape = (rect.height(), rect.width())
        if after.shape != shape:
            raise ValueError(f"patch data {after.shape} does not match rect {shape}")
        if before is None:
            before = _region(self.alpha, rect).copy()
        elif before.shape != shape:
            raise ValueError(f"before data {before.shape} does not match rect {shape}")
        self.undo_stack.push(
            _PatchCommand(self, rect, before.astype(np.float32), after.astype(np.float32), text)
        )

    def mark_saved(self) -> None:
        self.undo_stack.setClean()

    def export_rgba(self, decontaminate: bool | None = None) -> np.ndarray:
        """Full-resolution RGBA for saving. Pure function of a snapshot, safe off the UI thread."""
        if self.image is None or self.alpha is None:
            raise RuntimeError("nothing to export")
        if decontaminate is None:
            decontaminate = self.result.decontaminate if self.result is not None else True
        return compose_rgba(self.image, self.alpha.copy(), decontaminate)

    def snapshot_alpha(self) -> np.ndarray:
        """A copy of the matte, for handing to a worker thread while the user keeps editing."""
        if self.alpha is None:
            raise RuntimeError("no matte yet")
        return self.alpha.copy()


def _region(a: np.ndarray, rect: QRect) -> np.ndarray:
    return a[rect.top() : rect.top() + rect.height(), rect.left() : rect.left() + rect.width()]


class _PatchCommand(QUndoCommand):
    def __init__(self, doc: CutoutDocument, rect: QRect, before, after, text: str):
        super().__init__(text)
        self.doc, self.rect, self.before, self.after = doc, rect, before, after

    def _write(self, data: np.ndarray) -> None:
        np.clip(data, 0.0, 1.0, out=_region(self.doc.alpha, self.rect))
        self.doc.alpha_changed.emit(QRect(self.rect))

    def redo(self) -> None:
        self._write(self.after)

    def undo(self) -> None:
        self._write(self.before)


class _ReplaceAlphaCommand(QUndoCommand):
    def __init__(self, doc: CutoutDocument, new_alpha: np.ndarray, text: str):
        super().__init__(text)
        self.doc = doc
        self.other = new_alpha  # whichever matte is not currently installed

    def _swap(self) -> None:
        self.doc.alpha, self.other = self.other, self.doc.alpha
        self.doc.alpha_changed.emit(self.doc.full_rect())

    def redo(self) -> None:
        self._swap()

    def undo(self) -> None:
        self._swap()
