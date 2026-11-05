"""Zoomable preview of the cutout over a transparency checkerboard.

The preview is the original image drawn with the matte as its alpha channel, in
an 8-bit premultiplied buffer that is updated only in the rectangles that change,
so brush strokes redraw instantly even on a 24 MP photo. Edge colour cleanup
(``compose_rgba``) is skipped here; it runs only on export.

Scene coordinates equal image pixel coordinates (the image item sits at 0,0 with
scale 1), which is what tools receive.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import QLineF, QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QFrame, QGraphicsItem, QGraphicsScene, QGraphicsView, QStyleOptionGraphicsItem

from .document import CutoutDocument
from .tools import CanvasTool

MIN_ZOOM = 0.02
MAX_ZOOM = 64.0
PIXEL_GRID_FROM = 8.0  # zoom at which a grid shows each image pixel
ZOOM_STEP = 1.25
CHECKER_SIZE = 10  # screen pixels per checker square


class Backdrop(str, Enum):
    CHECKER = "checker"
    WHITE = "white"
    BLACK = "black"


def _checker_brush(light: QColor, dark: QColor, size: int = CHECKER_SIZE) -> QBrush:
    pm = QPixmap(size * 2, size * 2)
    pm.fill(light)
    p = QPainter(pm)
    p.fillRect(0, 0, size, size, dark)
    p.fillRect(size, size, size, size, dark)
    p.end()
    return QBrush(pm)


class PreviewItem(QGraphicsItem):
    """Draws an RGBA8888-premultiplied buffer; repaints only exposed / changed areas."""

    def __init__(self, width: int, height: int):
        super().__init__()
        self._w, self._h = width, height
        self.buffer = np.zeros((height, width, 4), np.uint8)
        self.qimage = QImage(self.buffer.data, width, height, width * 4, QImage.Format.Format_RGBA8888_Premultiplied)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)
        # No item cache: Qt's device cache re-rendered brush areas with half-painted edge
        # pixels when zoomed out (lines and boxes left behind by strokes). Painting only the
        # dirty part of the viewport straight from the buffer is fast enough and exact.
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._w, self._h)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        bounds = self.boundingRect()
        exposed = option.exposedRect.intersected(bounds)
        if exposed.isEmpty():
            return
        transform = painter.worldTransform()
        scale = transform.m11()
        # Repaint whole screen pixels. The exposed area is in image pixels; zoomed out, its
        # edges fall inside screen pixels, and drawing only that much leaves those edge
        # pixels half painted: thin lines and boxes around every brush stroke. So clip to the
        # exposed area grown to whole screen pixels, and draw a little more source than that
        # (so smoothing at the edge also samples the real neighbouring pixels).
        inverse, ok = transform.inverted()
        if ok:
            device = transform.mapRect(exposed).toAlignedRect()
            clip = inverse.mapRect(QRectF(device)).intersected(bounds)
        else:
            clip = exposed
        margin = int(np.ceil(2.0 / max(scale, 1e-6))) + 1
        src = clip.toAlignedRect().adjusted(-margin, -margin, margin, margin).intersected(bounds.toRect())
        painter.save()
        painter.setClipRect(clip, Qt.ClipOperation.IntersectClip)
        # Smooth when zoomed out; crisp, true pixels when zoomed in for editing.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 1.0)
        painter.drawImage(QRectF(src), self.qimage, QRectF(src))
        painter.restore()


class CutoutCanvas(QGraphicsView):
    zoom_changed = Signal(float)
    #: Image pixel under the cursor (x, y), or (-1, -1) when outside the image.
    hover_pixel = Signal(int, int)

    def __init__(self, document: CutoutDocument, parent=None):
        super().__init__(parent)
        self.doc = document
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: PreviewItem | None = None
        self._rgb8: np.ndarray | None = None

        self._tool: CanvasTool | None = None
        self._backdrop = Backdrop.CHECKER
        self._show_original = False
        self._fit_mode = True
        self._panning = False
        self._pan_last = QPoint()
        self._space_held = False

        self._canvas_color = QColor(32, 33, 36)
        self._checker = _checker_brush(QColor(204, 204, 204), QColor(153, 153, 153))

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Navigation is drag / Space+drag / wheel; scroll bars would only add clutter.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(False)  # the main window handles drops

        document.image_changed.connect(self._on_image_changed)
        document.alpha_changed.connect(self._on_alpha_changed)

    # ------------------------------------------------------------ public API
    @property
    def zoom(self) -> float:
        return float(self.transform().m11())

    def image_to_view_scale(self) -> float:
        """Screen pixels per image pixel (for sizing overlays such as a brush outline)."""
        return self.zoom

    def set_tool(self, tool: CanvasTool | None) -> None:
        self._tool = tool
        self._update_cursor()

    def set_backdrop(self, backdrop: Backdrop) -> None:
        self._backdrop = Backdrop(backdrop)
        self.viewport().update()

    @property
    def backdrop(self) -> Backdrop:
        return self._backdrop

    def set_show_original(self, on: bool) -> None:
        if on != self._show_original:
            self._show_original = on
            self._refresh(self.doc.full_rect())

    @property
    def show_original(self) -> bool:
        return self._show_original

    def fit_to_window(self) -> None:
        self._fit_mode = True
        if self._item is None:
            return
        self.resetTransform()
        margin = 24
        vw = max(1, self.viewport().width() - 2 * margin)
        vh = max(1, self.viewport().height() - 2 * margin)
        scale = min(vw / self.doc.image.width, vh / self.doc.image.height, 1.0)
        self._apply_scale(max(scale, MIN_ZOOM), keep_fit=True)
        self.centerOn(self._item)

    def actual_pixels(self) -> None:
        self._set_zoom(1.0, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def zoom_in(self) -> None:
        self._set_zoom(self.zoom * ZOOM_STEP, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def zoom_out(self) -> None:
        self._set_zoom(self.zoom / ZOOM_STEP, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def preview_buffer(self) -> np.ndarray | None:
        """The premultiplied RGBA preview (for tests and screenshots)."""
        return None if self._item is None else self._item.buffer

    # ------------------------------------------------------- document hooks
    def _on_image_changed(self) -> None:
        # Remove only the preview: tools keep their own overlay items in the scene.
        if self._item is not None:
            self._scene.removeItem(self._item)
        self._item = None
        self._rgb8 = None
        img = self.doc.image
        if img is None:
            self._scene.setSceneRect(QRectF())
            self.viewport().update()
            return
        self._rgb8 = img.as_uint8()
        self._item = PreviewItem(img.width, img.height)
        self._scene.addItem(self._item)
        # Leave room around the image so it can be scrolled to any edge.
        pad = max(img.width, img.height)
        self._scene.setSceneRect(QRectF(-pad, -pad, img.width + 2 * pad, img.height + 2 * pad))
        self._refresh(self.doc.full_rect())
        self.fit_to_window()

    def _on_alpha_changed(self, rect: QRect) -> None:
        self._refresh(rect)

    def _refresh(self, rect: QRect) -> None:
        if self._item is None or self._rgb8 is None:
            return
        rect = rect.intersected(self.doc.full_rect())
        if rect.isEmpty():
            return
        y0, y1 = rect.top(), rect.top() + rect.height()
        x0, x1 = rect.left(), rect.left() + rect.width()
        buf = self._item.buffer[y0:y1, x0:x1]
        rgb = self._rgb8[y0:y1, x0:x1]
        if self._show_original or self.doc.alpha is None:
            buf[..., :3] = rgb
            buf[..., 3] = 255
        else:
            a = np.round(np.clip(self.doc.alpha[y0:y1, x0:x1], 0.0, 1.0) * 255.0).astype(np.uint16)
            buf[..., :3] = ((rgb.astype(np.uint16) * a[..., None] + 127) // 255).astype(np.uint8)
            buf[..., 3] = a.astype(np.uint8)
        # Zoomed out, a changed pixel also shows in the neighbouring screen pixels
        # (smoothing), so repaint a margin of a couple of screen pixels around it.
        grow = int(np.ceil(2.0 / max(self.zoom, 1e-6))) + 1
        self._item.update(QRectF(rect.adjusted(-grow, -grow, grow, grow)).intersected(self._item.boundingRect()))

    # ------------------------------------------------------------- drawing
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, self._canvas_color)
        if self._item is None:
            return
        area = self._item.boundingRect().intersected(rect)
        if area.isEmpty():
            return
        if self._backdrop is Backdrop.CHECKER:
            # Paint in screen space so the squares stay the same size at every zoom.
            device = self.mapFromScene(area).boundingRect()
            painter.save()
            painter.resetTransform()
            painter.fillRect(device, self._checker)
            painter.restore()
        else:
            painter.fillRect(area, QColor("white") if self._backdrop is Backdrop.WHITE else QColor("black"))

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """When zoomed in far enough to edit single pixels, outline each one."""
        if self._item is None or self.zoom < PIXEL_GRID_FROM:
            return
        area = self._item.boundingRect().intersected(rect)
        if area.isEmpty():
            return
        x0, x1 = int(np.floor(area.left())), int(np.ceil(area.right()))
        y0, y1 = int(np.floor(area.top())), int(np.ceil(area.bottom()))
        pen = QPen(QColor(128, 128, 128, 90), 0)  # width 0 = 1 screen px at any zoom
        painter.save()
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        lines = [QLineF(x, y0, x, y1) for x in range(x0, x1 + 1)]
        lines += [QLineF(x0, y, x1, y) for y in range(y0, y1 + 1)]
        painter.drawLines(lines)
        painter.restore()

    # ------------------------------------------------------------- zooming
    def _set_zoom(self, z: float, anchor: QGraphicsView.ViewportAnchor | None = None) -> None:
        if self._item is None:
            return
        old = self.transformationAnchor()
        if anchor is not None:
            self.setTransformationAnchor(anchor)
        self._apply_scale(z)
        self.setTransformationAnchor(old)

    def _apply_scale(self, z: float, keep_fit: bool = False) -> None:
        z = float(min(MAX_ZOOM, max(MIN_ZOOM, z)))
        factor = z / self.zoom
        if abs(factor - 1.0) > 1e-9:
            self.scale(factor, factor)
        self._fit_mode = keep_fit
        self.zoom_changed.emit(self.zoom)
        if self._tool is not None:
            self._tool.view_changed()

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            if self._tool is not None and self._tool.wheel_with_modifier(steps, mods):
                event.accept()
                return
        self._set_zoom(self.zoom * (ZOOM_STEP**steps), anchor=QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_window()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if self._tool is not None:
            self._tool.view_changed()

    # --------------------------------------------------------------- input
    def _image_pos(self, event: QMouseEvent) -> QPointF:
        # Map the centre of the screen pixel under the pointer, in floating point, so a
        # 1 px brush lands on exactly the image pixel the cursor is over at any zoom.
        inv, ok = self.viewportTransform().inverted()
        return inv.map(event.position() + QPointF(0.5, 0.5))

    def _tool_enabled(self) -> bool:
        if self._tool is None or self._item is None:
            return False
        return self.doc.has_alpha or not self._tool.needs_alpha

    def _wants_pan(self, event: QMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.MiddleButton:
            return True
        if event.button() == Qt.MouseButton.LeftButton:
            return self._space_held or self._tool is None or self._tool.id == "pan" or not self._tool_enabled()
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._item is not None and self._wants_pan(event):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self._tool_enabled() and self._tool.mouse_press(self._image_pos(event), event):
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = self._image_pos(event)
        self._emit_hover(pos)
        if self._panning:
            p = event.position().toPoint()
            delta = p - self._pan_last
            self._pan_last = p
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._tool_enabled() and self._tool.mouse_move(pos, event):
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._panning = False
            self._update_cursor()
            event.accept()
            return
        if self._tool_enabled() and self._tool.mouse_release(self._image_pos(event), event):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self.hover_pixel.emit(-1, -1)
        if self._tool is not None:
            self._tool.hover_leave()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if self._tool_enabled() and self._tool.key_press(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._update_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _emit_hover(self, pos: QPointF) -> None:
        img = self.doc.image
        if img is None:
            return
        x, y = int(np.floor(pos.x())), int(np.floor(pos.y()))
        if 0 <= x < img.width and 0 <= y < img.height:
            self.hover_pixel.emit(x, y)
        else:
            self.hover_pixel.emit(-1, -1)

    def _update_cursor(self) -> None:
        if self._tool is not None and self._tool_enabled():
            self.viewport().setCursor(self._tool.cursor())
        else:
            self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
