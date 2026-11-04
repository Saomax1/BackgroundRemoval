"""Mark Face: drag a box over a face to say where the head is, for Head only.

Used when the face finder found no face (a face turned away, a very stylised
drawing) or picked the wrong one. The window does the cutting (``on_box``); this
tool only draws the box.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QPen
from PySide6.QtWidgets import QGraphicsRectItem

from .tools import CanvasTool

# Smaller drags are taken for stray clicks.
MIN_BOX_PX = 4


class HeadBoxTool(CanvasTool):
    id = "mark_face"
    label = "Mark Face"
    tooltip = ("Drag a box over a face, from the forehead to the chin, to keep only that head (F). "
               "Use it when Head only finds no face or the wrong one.")

    def __init__(self, on_box: Callable[[float, float, float, float], None]):
        super().__init__()
        self.on_box = on_box
        self._start: QPointF | None = None
        self._item: QGraphicsRectItem | None = None

    def deactivate(self) -> None:
        self._clear()
        super().deactivate()

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)

    def mouse_press(self, pos: QPointF, event) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or self.ctx is None:
            return False
        self._clear()
        self._start = QPointF(pos)
        item = QGraphicsRectItem(QRectF(pos, pos))
        pen = QPen(QColor(91, 141, 239), 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)  # same width at any zoom
        item.setPen(pen)
        item.setBrush(QBrush(QColor(91, 141, 239, 40)))
        item.setZValue(10)
        self.ctx.canvas.scene().addItem(item)
        self._item = item
        return True

    def mouse_move(self, pos: QPointF, event) -> bool:
        if self._start is None or self._item is None:
            return False
        self._item.setRect(QRectF(self._start, pos).normalized())
        return True

    def mouse_release(self, pos: QPointF, event) -> bool:
        if self._start is None or event.button() != Qt.MouseButton.LeftButton:
            return False
        box = QRectF(self._start, pos).normalized()
        self._clear()
        if box.width() < MIN_BOX_PX or box.height() < MIN_BOX_PX:
            if self.ctx is not None:
                self.ctx.show_message("Drag a box over the face, from the forehead to the chin.")
            return True
        self.on_box(box.left(), box.top(), box.right(), box.bottom())
        return True

    def _clear(self) -> None:
        self._start = None
        if self._item is not None:
            try:
                scene = self._item.scene()
                if scene is not None:
                    scene.removeItem(self._item)
            except RuntimeError:  # the canvas cleared its scene (a new image) and deleted it
                pass
            self._item = None
