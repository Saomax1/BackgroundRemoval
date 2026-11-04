"""Touch-up tools in the editor window: Keep / Erase / Refine brushes and click-to-select.

Thin Qt adapters over the pure-NumPy core in ``cutout.editing``:

* :class:`MatteBridge` keeps one ``MatteEditor`` on the document's matte (rebuilt
  when "Redo Auto" swaps the matte) and turns each finished edit into one undo
  step on the window's ``QUndoStack`` (only the changed rectangle is stored).
* :class:`BrushTool` paints with a round brush whose size is set in pixels of
  the full-resolution image, from 1 px (exactly one pixel, no feathering) up to
  1000 px. Size: slider, number box, "1 px" button, ``[`` / ``]`` or Ctrl+wheel.
  Alt swaps Keep and Erase while held. Zoom in (up to 6400%, with a pixel grid
  from 800%) to place single pixels.
* :class:`SelectTool` runs SAM 2.1 clicks on a worker thread: left-click adds,
  right-click excludes, the selection is tinted blue, Enter keeps it and Delete
  removes it from the cutout.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QKeyEvent, QMouseEvent, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from ..core.models import ModelNotInstalled
from ..editing import Brush, BrushMode, MaskOp, MatteEditor, Patch, Rect, Selection
from ..editing.brush import dab_rect, is_pixel_exact, pixel_stamp, snap_centre
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
    """Shared by the three brushes, so switching Keep/Erase/Refine keeps the size.

    ``size`` is the brush diameter in *image* pixels, so 1 means exactly one pixel of
    the full-resolution photo whatever the zoom. Brushes under 5 px paint whole pixels
    with no feathering (see ``editing.brush.PIXEL_EXACT_BELOW``).
    """

    changed = Signal()
    MIN_SIZE, MAX_SIZE = 1, 1000

    def __init__(self, size: int = 40, softness: int = 40):
        super().__init__()
        self._size = int(min(self.MAX_SIZE, max(self.MIN_SIZE, size)))
        self._softness = softness

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: float) -> None:
        value = int(min(self.MAX_SIZE, max(self.MIN_SIZE, round(value))))
        if value != self._size:
            self._size = value
            self.changed.emit()

    def step(self, steps: float) -> None:
        """Grow (steps > 0) or shrink the brush: 1 px at a time when small, ~15% when large."""
        size = self._size
        for _ in range(max(1, int(round(abs(steps))))):
            if steps > 0:
                size = size + 1 if size < 10 else max(size + 1, round(size * 1.15))
            else:
                size = size - 1 if size <= 10 else min(size - 1, round(size / 1.15))
        self.size = size

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


SLIDER_STEPS = 1000  # the size slider is logarithmic: fine control at 1-10 px, still reaches 1000


def _slider_to_size(v: int) -> int:
    lo, hi = math.log(BrushSettings.MIN_SIZE), math.log(BrushSettings.MAX_SIZE)
    return int(round(math.exp(lo + (hi - lo) * v / SLIDER_STEPS)))


def _size_to_slider(size: int) -> int:
    lo, hi = math.log(BrushSettings.MIN_SIZE), math.log(BrushSettings.MAX_SIZE)
    return int(round((math.log(max(size, 1)) - lo) / (hi - lo) * SLIDER_STEPS))


class BrushOptions(QWidget):
    def __init__(self, settings: BrushSettings):
        super().__init__()
        self.settings = settings
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)

        tip = "Brush size in pixels of the full-size image, 1 to 1000 ( [ and ] or Ctrl+wheel )"
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(0, SLIDER_STEPS)
        self.size_slider.setFixedWidth(200)
        self.size_slider.setToolTip(tip)
        self.size_box = QSpinBox()
        self.size_box.setRange(BrushSettings.MIN_SIZE, BrushSettings.MAX_SIZE)
        self.size_box.setSuffix(" px")
        self.size_box.setToolTip(tip)
        self.size_box.setKeyboardTracking(False)
        self.one_px = QPushButton("1 px")
        self.one_px.setObjectName("optionButton")
        self.one_px.setToolTip("Single-pixel brush for the finest fixes (zoom in to see each pixel)")
        self.soft_slider = QSlider(Qt.Orientation.Horizontal)
        self.soft_slider.setRange(0, 100)
        self.soft_slider.setFixedWidth(110)
        self.soft_slider.setToolTip("Edge softness: 0 = hard edge, 100 = very soft. Brushes under 5 px are always hard.")
        self.hint = QLabel("Alt swaps Keep ↔ Erase  ·  Esc cancels a stroke")
        self.hint.setObjectName("optionsHint")

        row.addWidget(QLabel("Size"))
        row.addWidget(self.size_slider)
        row.addWidget(self.size_box)
        row.addWidget(self.one_px)
        row.addSpacing(10)
        row.addWidget(QLabel("Softness"))
        row.addWidget(self.soft_slider)
        row.addSpacing(16)
        row.addWidget(self.hint)
        row.addStretch(1)

        self.size_slider.valueChanged.connect(lambda v: setattr(settings, "size", _slider_to_size(v)))
        self.size_box.valueChanged.connect(lambda v: setattr(settings, "size", v))
        self.one_px.clicked.connect(lambda: setattr(settings, "size", 1))
        self.soft_slider.valueChanged.connect(lambda v: setattr(settings, "softness", v))
        settings.changed.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        pairs = ((self.size_slider, _size_to_slider(self.settings.size)),
                 (self.size_box, self.settings.size),
                 (self.soft_slider, self.settings.softness))
        for widget, value in pairs:
            if widget.value() != value:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        self.soft_slider.setEnabled(not is_pixel_exact(self.settings.size / 2.0))


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
        self._outline: list[QGraphicsPathItem] = []
        self._hover: QPointF | None = None
        self._stroking = False

    # -------------------------------------------------------------- lifecycle
    def activate(self, ctx: ToolContext) -> None:
        super().activate(ctx)
        scene = ctx.canvas.scene()
        # Two rings (dark under light) so the outline shows on any colour.
        for color, width in ((QColor(0, 0, 0, 160), 3.0), (QColor(255, 255, 255, 230), 1.2)):
            item = QGraphicsPathItem()
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

    def _outline_too_small(self) -> bool:
        zoom = self.ctx.canvas.image_to_view_scale() if self.ctx else 1.0
        return self.settings.size * zoom < 8  # under ~8 screen px a ring is hard to see

    def cursor(self) -> QCursor:
        # A crosshair marks the spot when the brush is too small on screen to see its ring.
        if self._outline and not self._outline_too_small():
            return QCursor(Qt.CursorShape.BlankCursor)
        return QCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------ size
    def image_radius(self) -> float:
        """Brush radius in image pixels (size is a diameter in full-resolution pixels)."""
        return max(0.5, self.settings.size / 2.0)

    def outline_path(self, pos: QPointF) -> QPainterPath:
        """The area a dab at ``pos`` paints: exact pixel cells for small brushes, else a circle."""
        r = self.image_radius()
        path = QPainterPath()
        if is_pixel_exact(r):
            cx, cy = snap_centre(pos.x(), pos.y(), r)
            rect = dab_rect(cx, cy, r)
            weights = pixel_stamp(cx, cy, r, rect)
            for j, i in zip(*np.nonzero(weights)):
                path.addRect(QRectF(rect.x + int(i), rect.y + int(j), 1, 1))
            return path.simplified()
        path.addEllipse(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r))
        return path

    def _update_outline(self) -> None:
        if not self._outline:
            return
        if self.ctx is not None:
            self.ctx.canvas.viewport().setCursor(self.cursor())
        if self._hover is None:
            for item in self._outline:
                item.setVisible(False)
            return
        path = self.outline_path(self._hover)
        for item in self._outline:
            item.setPath(path)
            item.setVisible(True)

    def view_changed(self) -> None:
        self._update_outline()

    def hover_leave(self) -> None:
        self._hover = None
        self._update_outline()

    def wheel_with_modifier(self, delta_steps: float, modifiers) -> bool:
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self.settings.step(delta_steps)
            return True
        return False

    def key_press(self, event: QKeyEvent) -> bool:
        key = event.key()
        if key == Qt.Key.Key_BracketLeft:
            self.settings.step(-1)
            return True
        if key == Qt.Key.Key_BracketRight:
            self.settings.step(1)
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
    try:
        size = int(window.settings.value("brush_size", 40))
        softness = int(window.settings.value("brush_softness", 40))
    except (TypeError, ValueError):
        size, softness = 40, 40
    settings = BrushSettings(size, softness)
    settings.changed.connect(lambda: (window.settings.setValue("brush_size", settings.size),
                                      window.settings.setValue("brush_softness", settings.softness)))
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
