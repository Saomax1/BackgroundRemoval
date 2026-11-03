"""Touch-up tools in the editor window: Keep / Erase / Refine brushes and click-to-select.

Thin Qt adapters over the pure-NumPy core in ``cutout.editing``:

* :class:`MatteBridge` keeps one ``MatteEditor`` on the document's matte (rebuilt
  when "Redo Auto" swaps the matte) and turns each finished edit into one undo
  step on the window's ``QUndoStack`` (only the changed rectangle is stored).
* :class:`BrushTool` paints with a soft round brush whose size is set in
  *screen* pixels, so zooming in automatically gives finer control. Alt swaps
  Keep and Erase while held. ``[`` / ``]`` or Ctrl+wheel change the size.
* :class:`SelectTool` runs SAM 2.1 clicks on a worker thread: left-click adds,
  right-click excludes, the selection is tinted blue, Enter keeps it and Delete
  removes it from the cutout.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QKeyEvent, QMouseEvent, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from ..core.models import ModelNotInstalled
from ..editing import Brush, BrushMode, MaskOp, MatteEditor, Patch, Rect, Selection
from .tools import CanvasTool, ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from ..core.engine import CutoutEngine
    from .document import CutoutDocument
    from .main_window import MainWindow

OVERLAY_Z = 10.0
SELECTION_COLOR = (91, 141, 239)  # accent blue
MAX_OVERLAY_SIDE = 2048  # selection tint is drawn at most this big, then scaled


def to_qrect(r: Rect) -> QRect:
    return QRect(r.x, r.y, r.w, r.h)


# ===================================================================== bridge
class MatteBridge:
    """One ``MatteEditor`` per matte, sending its patches to the document's undo stack."""

    def __init__(self, document: "CutoutDocument"):
        self.doc = document
        self._editor: MatteEditor | None = None

    def editor(self) -> MatteEditor:
        doc = self.doc
        if doc.alpha is None or doc.image is None:
            raise RuntimeError("no cutout to edit yet")
        ed = self._editor
        if ed is None or ed.alpha is not doc.alpha or ed.image is not doc.image:
            ed = MatteEditor(doc.alpha, doc.image, history=None)
            self._editor = ed
        return ed

    def commit(self, patch: Patch | None) -> None:
        """Record a finished edit (already applied to the matte) as one undo step."""
        if patch is None or patch.empty:
            return
        self.doc.push_patch(to_qrect(patch.rect), patch.after, before=patch.before, text=patch.label)


# ===================================================================== brush
class BrushSettings(QObject):
    """Shared by the three brushes, so switching Keep/Erase/Refine keeps the size."""

    changed = Signal()
    MIN_SIZE, MAX_SIZE = 2, 400  # brush diameter in screen pixels

    def __init__(self, size: int = 60, softness: int = 40):
        super().__init__()
        self._size = size
        self._softness = softness

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        value = int(min(self.MAX_SIZE, max(self.MIN_SIZE, round(value))))
        if value != self._size:
            self._size = value
            self.changed.emit()

    @property
    def softness(self) -> int:
        return self._softness

    @softness.setter
    def softness(self, value: int) -> None:
        value = int(min(100, max(0, value)))
        if value != self._softness:
            self._softness = value
            self.changed.emit()

    @property
    def hardness(self) -> float:
        return 1.0 - self._softness / 100.0


class BrushOptions(QWidget):
    def __init__(self, settings: BrushSettings):
        super().__init__()
        self.settings = settings
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(BrushSettings.MIN_SIZE, BrushSettings.MAX_SIZE)
        self.size_slider.setFixedWidth(180)
        self.size_slider.setToolTip("Brush size on screen ( [ and ] or Ctrl+wheel )")
        self.size_label = QLabel()
        self.size_label.setMinimumWidth(48)
        self.soft_slider = QSlider(Qt.Orientation.Horizontal)
        self.soft_slider.setRange(0, 100)
        self.soft_slider.setFixedWidth(120)
        self.soft_slider.setToolTip("Edge softness: 0 = hard edge, 100 = very soft")
        self.hint = QLabel("Hold Alt to switch Keep ↔ Erase  ·  Esc cancels a stroke")
        self.hint.setObjectName("optionsHint")

        row.addWidget(QLabel("Size"))
        row.addWidget(self.size_slider)
        row.addWidget(self.size_label)
        row.addSpacing(10)
        row.addWidget(QLabel("Softness"))
        row.addWidget(self.soft_slider)
        row.addSpacing(16)
        row.addWidget(self.hint)
        row.addStretch(1)

        self.size_slider.valueChanged.connect(lambda v: setattr(settings, "size", v))
        self.soft_slider.valueChanged.connect(lambda v: setattr(settings, "softness", v))
        settings.changed.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        for slider, value in ((self.size_slider, self.settings.size), (self.soft_slider, self.settings.softness)):
            if slider.value() != value:
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
        self.size_label.setText(f"{self.settings.size} px")


class BrushTool(CanvasTool):
    LABELS = {
        BrushMode.KEEP: ("Keep", "Paint back parts of the subject that were removed (K)"),
        BrushMode.ERASE: ("Erase", "Paint away leftover background (E)"),
        BrushMode.REFINE: ("Refine Edge", "Brush over hair or fur to snap the edge to the photo (R)"),
    }

    def __init__(self, mode: BrushMode, bridge: MatteBridge, settings: BrushSettings, options: BrushOptions):
        super().__init__()
        self.mode = BrushMode(mode)
        self.id = f"brush-{self.mode.value}"
        self.label, self.tooltip = self.LABELS[self.mode]
        self.bridge = bridge
        self.settings = settings
        self._options = options
        self._outline: list[QGraphicsEllipseItem] = []
        self._hover: QPointF | None = None
        self._stroking = False

    # -------------------------------------------------------------- lifecycle
    def activate(self, ctx: ToolContext) -> None:
        super().activate(ctx)
        scene = ctx.canvas.scene()
        # Two rings (dark under light) so the outline shows on any colour.
        for color, width in ((QColor(0, 0, 0, 160), 3.0), (QColor(255, 255, 255, 230), 1.2)):
            item = QGraphicsEllipseItem()
            pen = QPen(color, width)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)
            item.setZValue(OVERLAY_Z + 1)
            item.setVisible(False)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            scene.addItem(item)
            self._outline.append(item)
        self.settings.changed.connect(self._update_outline)

    def deactivate(self) -> None:
        if self._stroking:
            self._cancel()
        try:
            self.settings.changed.disconnect(self._update_outline)
        except (RuntimeError, TypeError):
            pass
        for item in self._outline:
            if item.scene() is not None:
                item.scene().removeItem(item)
        self._outline = []
        super().deactivate()

    def options_widget(self) -> QWidget:
        return self._options

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.BlankCursor if self._outline else Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------ size
    def image_radius(self) -> float:
        zoom = self.ctx.canvas.image_to_view_scale() if self.ctx else 1.0
        return max(0.5, self.settings.size / 2.0 / max(zoom, 1e-6))

    def _update_outline(self) -> None:
        if not self._outline:
            return
        if self._hover is None:
            for item in self._outline:
                item.setVisible(False)
            return
        r = self.image_radius()
        rect = QRectF(self._hover.x() - r, self._hover.y() - r, 2 * r, 2 * r)
        for item in self._outline:
            item.setRect(rect)
            item.setVisible(True)

    def view_changed(self) -> None:
        self._update_outline()

    def hover_leave(self) -> None:
        self._hover = None
        self._update_outline()

    def wheel_with_modifier(self, delta_steps: float, modifiers) -> bool:
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.settings.size = self.settings.size * (1.15 ** delta_steps)
            return True
        return False

    def key_press(self, event: QKeyEvent) -> bool:
        key = event.key()
        if key == Qt.Key.Key_BracketLeft:
            self.settings.size = min(self.settings.size - 1, self.settings.size / 1.2)
            return True
        if key == Qt.Key.Key_BracketRight:
            self.settings.size = max(self.settings.size + 1, self.settings.size * 1.2)
            return True
        if key == Qt.Key.Key_Escape and self._stroking:
            self._cancel()
            return True
        return False

    # ------------------------------------------------------------- painting
    def _mode_for(self, event: QMouseEvent) -> BrushMode:
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            if self.mode is BrushMode.KEEP:
                return BrushMode.ERASE
            if self.mode is BrushMode.ERASE:
                return BrushMode.KEEP
        return self.mode

    def mouse_press(self, pos: QPointF, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        self._hover = pos
        brush = Brush(mode=self._mode_for(event), radius=self.image_radius(), hardness=self.settings.hardness)
        editor = self.bridge.editor()
        dirty = editor.begin_stroke(pos.x(), pos.y(), brush)
        self._stroking = True
        self.ctx.document.notify_alpha_changed(to_qrect(dirty))
        self._update_outline()
        return True

    def mouse_move(self, pos: QPointF, event: QMouseEvent) -> bool:
        self._hover = pos
        self._update_outline()
        if self._stroking and event.buttons() & Qt.MouseButton.LeftButton:
            dirty = self.bridge.editor().continue_stroke(pos.x(), pos.y())
            if not dirty.empty:
                self.ctx.document.notify_alpha_changed(to_qrect(dirty))
            return True
        return False

    def mouse_release(self, pos: QPointF, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or not self._stroking:
            return False
        self._stroking = False
        self.bridge.commit(self.bridge.editor().end_stroke())
        return True

    def _cancel(self) -> None:
        self._stroking = False
        dirty = self.bridge.editor().cancel_stroke()
        if not dirty.empty:
            self.ctx.document.notify_alpha_changed(to_qrect(dirty))


# =========================================================== click-to-select
class SelectOptionsBar(QWidget):
    def __init__(self):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)
        self.hint = QLabel("Click the part you want  ·  right-click to exclude")
        self.hint.setObjectName("optionsHint")
        self.keep = QPushButton("Keep Selection")
        self.keep.setToolTip("Add the blue area to the cutout (Enter)")
        self.remove = QPushButton("Remove Selection")
        self.remove.setToolTip("Remove the blue area from the cutout (Delete)")
        self.clear = QPushButton("Clear")
        self.clear.setToolTip("Start over (Esc). Backspace undoes the last click.")
        for b in (self.keep, self.remove, self.clear):
            b.setObjectName("optionButton")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.hint)
        row.addSpacing(10)
        row.addWidget(self.keep)
        row.addWidget(self.remove)
        row.addWidget(self.clear)
        row.addStretch(1)
        self.set_has_selection(False)

    def set_has_selection(self, on: bool) -> None:
        for b in (self.keep, self.remove, self.clear):
            b.setEnabled(on)


class SelectTool(CanvasTool):
    id = "select"
    label = "Select"
    tooltip = "Click to select a whole part of the picture, then keep or remove it (S)"

    def __init__(self, engine: "CutoutEngine", bridge: MatteBridge):
        super().__init__()
        self.engine = engine
        self.bridge = bridge
        self._options = SelectOptionsBar()
        self._options.keep.clicked.connect(lambda: self.apply(MaskOp.ADD))
        self._options.remove.clicked.connect(lambda: self.apply(MaskOp.SUBTRACT))
        self._options.clear.clicked.connect(self.clear)
        self.selector = None
        self._selector_image = None
        self._preparing = False
        self._pending = False
        self._overlay: QGraphicsPixmapItem | None = None
        self._markers: list[QGraphicsEllipseItem] = []
        self._doc = None

    @property
    def selection(self) -> Selection | None:
        return self.selector.selection if self.selector is not None else None

    @property
    def busy(self) -> bool:
        return self._preparing or self._pending

    # -------------------------------------------------------------- lifecycle
    def activate(self, ctx: ToolContext) -> None:
        super().activate(ctx)
        self._doc = ctx.document
        self._doc.image_changed.connect(self._on_image_changed)
        self._ensure_ready()

    def deactivate(self) -> None:
        self.clear()
        if self._doc is not None:
            try:
                self._doc.image_changed.disconnect(self._on_image_changed)
            except (RuntimeError, TypeError):
                pass
        super().deactivate()

    def options_widget(self) -> QWidget:
        return self._options

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.PointingHandCursor)

    def _on_image_changed(self) -> None:
        self.clear()
        self.selector = None
        self._selector_image = None

    def _ensure_ready(self) -> bool:
        """Create the selector for the open image and start its encoder. True when clicks can run."""
        doc = self.ctx.document
        if doc.image is None or not doc.has_alpha:
            return False
        if self.selector is not None and self._selector_image is doc.image:
            return self.selector.ready and not self._preparing
        if self._preparing:
            return False
        try:
            self.selector = self.engine.click_selector(doc.image)
        except ModelNotInstalled as exc:
            self.selector = None
            self.ctx.report_error(
                "Click-to-select model not installed",
                ModelNotInstalled(f"{exc}\n\nThe brushes still work without it."),
            )
            return False
        self._selector_image = doc.image
        self._preparing = True
        ctx = self.ctx
        ctx.set_busy("Getting click-to-select ready…")
        selector = self.selector

        def done(_):
            self._preparing = False
            ctx.set_busy(None)
            if selector is self.selector:
                ctx.show_message("Click-to-select is ready: click the part you want.")

        def failed(exc):
            self._preparing = False
            ctx.set_busy(None)
            if selector is self.selector:
                self.selector = None
                self._selector_image = None
            ctx.report_error("Click-to-select failed to start", exc)

        ctx.run_in_background(selector.prepare, on_done=done, on_error=failed)
        return False

    # ------------------------------------------------------------- clicking
    def mouse_press(self, pos: QPointF, event: QMouseEvent) -> bool:
        button = event.button()
        if button not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return False
        doc = self.ctx.document
        if not (0 <= pos.x() < doc.image.width and 0 <= pos.y() < doc.image.height):
            return True
        if not self._ensure_ready():
            return True
        if self._pending:
            self.ctx.show_message("Still working on the last click…")
            return True
        positive = button == Qt.MouseButton.LeftButton
        self._add_marker(pos, positive)
        self._pending = True
        self.ctx.show_message("Selecting…")
        selector, ctx = self.selector, self.ctx

        def done(sel: Selection):
            self._pending = False
            if selector is self.selector and self.ctx is ctx:
                self._show(sel)
                ctx.show_message("Enter keeps the blue area, Delete removes it, Esc clears.")

        def failed(exc):
            self._pending = False
            if selector is self.selector and self.ctx is ctx:
                self._pop_marker()
            ctx.report_error("Click-to-select failed", exc)

        ctx.run_in_background(selector.add_click, pos.x(), pos.y(), positive, on_done=done, on_error=failed)
        return True

    def key_press(self, event: QKeyEvent) -> bool:
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return self.apply(MaskOp.ADD)
        if key == Qt.Key.Key_Delete:
            return self.apply(MaskOp.SUBTRACT)
        if key == Qt.Key.Key_Escape:
            had = self.selection is not None
            self.clear()
            return had
        if key == Qt.Key.Key_Backspace and self.selector is not None and not self._pending:
            if self.selector.clicks:
                sel = self.selector.undo_click()
                self._pop_marker()
                self._show(sel)
                return True
        return False

    def apply(self, op: MaskOp) -> bool:
        """Merge the current selection into the cutout as one undo step."""
        sel = self.selection
        if sel is None or sel.empty or self._pending or self.ctx is None:
            return False
        self.bridge.commit(self.bridge.editor().apply_mask(sel, op))
        self.clear()
        return True

    def clear(self) -> None:
        if self.selector is not None and not self._pending:
            self.selector.clear()
        self._show(None)
        for m in self._markers:
            if m.scene() is not None:
                m.scene().removeItem(m)
        self._markers = []

    # -------------------------------------------------------------- overlay
    def _scene(self):
        return self.ctx.canvas.scene() if self.ctx is not None else None

    def _show(self, sel: Selection | None) -> None:
        scene = self._scene()
        if self._overlay is not None and self._overlay.scene() is not None:
            self._overlay.scene().removeItem(self._overlay)
        self._overlay = None
        has = sel is not None and not sel.empty
        self._options.set_has_selection(has)
        if not has or scene is None:
            return
        step = max(1, math.ceil(max(sel.rect.w, sel.rect.h) / MAX_OVERLAY_SIDE))
        m = np.ascontiguousarray(sel.mask[::step, ::step])
        h, w = m.shape
        rgba = np.empty((h, w, 4), np.uint8)
        a = np.round(np.clip(m, 0.0, 1.0) * 120.0).astype(np.uint16)
        for i, c in enumerate(SELECTION_COLOR):
            rgba[..., i] = (a * c // 255).astype(np.uint8)  # premultiplied
        rgba[..., 3] = a.astype(np.uint8)
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888_Premultiplied).copy()
        item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setScale(float(step))
        item.setPos(sel.rect.x, sel.rect.y)
        item.setZValue(OVERLAY_Z)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(item)
        self._overlay = item

    def _add_marker(self, pos: QPointF, positive: bool) -> None:
        scene = self._scene()
        if scene is None:
            return
        r = 6.0
        item = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        item.setPos(pos)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        item.setBrush(QBrush(QColor(52, 199, 89) if positive else QColor(255, 69, 58)))
        pen = QPen(QColor("white"), 2)
        item.setPen(pen)
        item.setZValue(OVERLAY_Z + 2)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(item)
        self._markers.append(item)

    def _pop_marker(self) -> None:
        if self._markers:
            m = self._markers.pop()
            if m.scene() is not None:
                m.scene().removeItem(m)


# ================================================================== install
def install_touchup_tools(window: "MainWindow") -> dict[str, CanvasTool]:
    """Register the brushes and click-to-select on the window. Returns them by id."""
    bridge = MatteBridge(window.doc)
    settings = BrushSettings()
    options = BrushOptions(settings)
    tools: dict[str, CanvasTool] = {}
    for mode, key in ((BrushMode.KEEP, "K"), (BrushMode.ERASE, "E"), (BrushMode.REFINE, "R")):
        tool = BrushTool(mode, bridge, settings, options)
        window.register_tool(tool, shortcut=key)
        tools[tool.id] = tool
    select = SelectTool(window.engine, bridge)
    window.register_tool(select, shortcut="S")
    tools[select.id] = select
    window.touchup = tools
    window.brush_settings = settings
    return tools
