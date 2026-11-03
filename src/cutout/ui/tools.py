"""Hooks for touch-up tools (brush, click-to-select, ...).

A tool is a :class:`CanvasTool` subclass registered on the main window::

    window.register_tool(BrushTool(), shortcut="B")

The window adds it to the toolbar; when it is the active tool the canvas sends
it mouse and key events. Positions are already in **image pixel coordinates**
(float, ``0..width`` / ``0..height``) whatever the zoom, so tools never deal
with view transforms. A tool edits ``ctx.document.alpha`` and records undo
steps with ``ctx.document.push_patch`` (see ``document.py``).

Tools must stay light on the UI thread: painting a brush stamp is fine; running
a model (click-to-select) must go through ``ctx.run_in_background``.

The built-in :class:`PanTool` is the default and shows the minimal shape of a tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QCursor, QIcon, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:  # pragma: no cover
    from .canvas import CutoutCanvas
    from .document import CutoutDocument


@dataclass
class ToolContext:
    document: "CutoutDocument"
    canvas: "CutoutCanvas"
    # Run fn(*args) on the thread pool; on_done(result) / on_error(exc) arrive on the UI thread.
    run_in_background: Callable[..., None]
    # Show a short message in the status bar.
    show_message: Callable[[str], None]
    # Show an error to the user: report_error(title, exception).
    report_error: Callable[[str, BaseException], None] = lambda title, exc: None
    # Cover the canvas with the busy overlay: set_busy("Working…") ... set_busy(None).
    set_busy: Callable[[str | None], None] = lambda text: None


class CanvasTool:
    """Base class. Override what you need; return True from an event handler to consume it."""

    #: Stable id, also used for settings.
    id: str = "tool"
    #: Toolbar label and tooltip.
    label: str = "Tool"
    tooltip: str = ""
    #: Optional toolbar icon.
    icon: QIcon | None = None
    #: Tools that edit the matte are disabled until an automatic cutout exists.
    needs_alpha: bool = True

    def __init__(self) -> None:
        self.ctx: ToolContext | None = None

    # lifecycle
    def activate(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def deactivate(self) -> None:
        self.ctx = None

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)

    # input, ``pos`` in image pixel coordinates
    def mouse_press(self, pos: QPointF, event: QMouseEvent) -> bool:
        return False

    def mouse_move(self, pos: QPointF, event: QMouseEvent) -> bool:
        """Called on drag and on hover (``event.buttons()`` tells them apart)."""
        return False

    def mouse_release(self, pos: QPointF, event: QMouseEvent) -> bool:
        return False

    def key_press(self, event: QKeyEvent) -> bool:
        return False

    def wheel_with_modifier(self, delta_steps: float, modifiers: Any) -> bool:
        """Ctrl/Alt + wheel (plain wheel always zooms). Use it for brush size, for example."""
        return False

    def view_changed(self) -> None:
        """The zoom or scroll changed: update any on-screen overlay such as a brush outline."""

    def hover_leave(self) -> None:
        """The pointer left the canvas: hide hover-only overlays."""

    def options_widget(self) -> QWidget | None:
        """Settings shown in the options bar under the toolbar while this tool is active.

        Return the same widget on every call. Tools may share one widget (all brushes do).
        """
        return None


class PanTool(CanvasTool):
    """Default tool: drag to move around. Also works when no cutout exists yet."""

    id = "pan"
    label = "Move"
    tooltip = "Move around the image (H). Hold Space with any tool to pan temporarily."
    needs_alpha = False

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.OpenHandCursor)
