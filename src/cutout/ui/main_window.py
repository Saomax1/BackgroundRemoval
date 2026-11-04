"""The editor window: open or drop one image, remove the background, touch up, save.

Flow: open -> automatic removal starts by itself -> preview over a checkerboard ->
(optional) touch-up tools -> Save PNG at the exact input resolution.

Every engine call runs on ``QThreadPool`` (see ``workers.py``); while one runs a
busy overlay covers the canvas and the actions that would conflict are disabled.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QRectF, QSettings, QSize, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.engine import CutoutEngine, CutoutResult
from ..core import debug, parallel
from ..core.imageio import SUPPORTED_EXTENSIONS, LoadedImage, encode_rgba_png, load_image, write_png_bytes
from ..core.models import REGISTRY, ModelNotInstalled, Subject
from ..core.runtime import GpuOutOfMemory
from ..core.refine import compose_rgba
from . import debug_menu, update_ui
from .canvas import Backdrop, CutoutCanvas
from .document import CutoutDocument
from .tools import CanvasTool, PanTool, ToolContext
from .workers import Task

log = logging.getLogger(__name__)

APP_NAME = "Cutout"
OPEN_FILTER = "Images ({});;All files (*)".format(" ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS)))


def is_supported_image(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_cutout.png")


# ============================================================== small widgets
class EmptyState(QWidget):
    """What the window shows before an image is open: a big, friendly drop zone."""

    open_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(14)

        title = QLabel("Drop an image here")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("People, anime and cartoon characters. The background is removed automatically.")
        sub.setObjectName("emptySub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        button = QPushButton("Open image…")
        button.setObjectName("primaryButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(self.open_requested)
        formats = QLabel("JPG · PNG · WebP · TIFF · BMP   ·   saved as transparent PNG at full size")
        formats.setObjectName("emptyHint")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(6)
        layout.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(6)
        layout.addWidget(formats)

    def set_drag_hover(self, on: bool) -> None:
        self._hover = on
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(32, 33, 36))
        zone = QRectF(self.rect()).adjusted(40, 40, -40, -40)
        color = QColor(91, 141, 239) if self._hover else QColor(90, 92, 98)
        if self._hover:
            p.setBrush(QColor(91, 141, 239, 28))
        pen = QPen(color, 2, Qt.PenStyle.DashLine)
        pen.setDashPattern([6, 5])
        p.setPen(pen)
        p.drawRoundedRect(zone, 18, 18)


class BusyOverlay(QWidget):
    """Translucent cover with a spinner and a message; blocks input to the canvas."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._text = ""
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.hide()

    def start(self, text: str) -> None:
        self._text = text
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(20, 21, 23, 150))
        font = QFont(self.font())
        font.setPointSizeF(font.pointSizeF() * 1.15)
        p.setFont(font)
        tw = p.fontMetrics().horizontalAdvance(self._text)
        pill = QRectF(0, 0, tw + 96, 56)
        pill.moveCenter(QRectF(self.rect()).center())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(44, 46, 51, 240))
        p.drawRoundedRect(pill, 28, 28)
        spin = QRectF(pill.left() + 22, pill.center().y() - 11, 22, 22)
        p.setPen(QPen(QColor(70, 72, 78), 3))
        p.drawEllipse(spin)
        pen = QPen(QColor(91, 141, 239), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(spin, -self._angle * 16, 100 * 16)
        p.setPen(QColor(232, 234, 237))
        p.drawText(QRectF(spin.right() + 14, pill.top(), tw + 10, pill.height()), Qt.AlignmentFlag.AlignVCenter, self._text)


# ================================================================ the window
class MainWindow(QMainWindow):
    #: Emitted after an automatic cutout is installed (tests and tools can listen).
    cutout_ready = Signal()
    #: Emitted after a PNG was written, with its path.
    saved = Signal(str)
    #: Emitted when a background task failed, with a user-facing message.
    task_failed = Signal(str)

    def __init__(
        self,
        engine: CutoutEngine | None = None,
        settings: QSettings | None = None,
        ask_user: bool = True,
    ):
        super().__init__()
        self.engine = engine or CutoutEngine()
        self.settings = settings or QSettings(APP_NAME, APP_NAME)
        self.ask_user = ask_user  # False in tests: never block on dialogs
        self.pool = QThreadPool.globalInstance()
        self.doc = CutoutDocument(self)
        self._generation = 0  # ignore results from removals that were superseded
        self._busy = 0
        self._last_save_path: Path | None = None
        self._tools: dict[str, CanvasTool] = {}
        self._tool_actions: dict[str, QAction] = {}
        self._active_tool: CanvasTool | None = None
        self._live_tasks: set[Task] = set()
        # A save prepared in the background right after a cutout: (image, revision, decon, future).
        self._save_prep: tuple[LoadedImage, int, bool, Any] | None = None

        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self.resize(1280, 840)
        self.setMinimumSize(QSize(760, 520))

        self._build_canvas()
        self._build_actions()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        debug_menu.install(self)
        update_ui.install(self)

        self.doc.image_changed.connect(self._on_image_changed)
        self.doc.result_changed.connect(self._on_result_changed)
        self.doc.alpha_changed.connect(self._on_doc_edited)
        self.doc.undo_stack.cleanChanged.connect(self._on_doc_edited)
        self.doc.undo_stack.indexChanged.connect(self._on_doc_edited)

        self.register_tool(PanTool(), shortcut="H")
        self.activate_tool("pan")
        self._restore_settings()
        self._sync_actions()
        self._update_title()

    # ------------------------------------------------------------ building
    def _build_canvas(self) -> None:
        self.canvas = CutoutCanvas(self.doc)
        self.empty = EmptyState()
        self.empty.open_requested.connect(self.open_dialog)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.canvas)
        self.setCentralWidget(self.stack)
        self.overlay = BusyOverlay(self.stack)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.hover_pixel.connect(self._on_hover_pixel)

    def _action(self, text: str, slot: Callable | None = None, shortcut=None, tip: str = "", checkable=False) -> QAction:
        act = QAction(text, self)
        if shortcut is not None:
            act.setShortcut(QKeySequence(shortcut))
        if tip:
            act.setToolTip(f"{tip} ({act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)})" if shortcut else tip)
            act.setStatusTip(tip)
        act.setCheckable(checkable)
        if slot is not None:
            act.triggered.connect(slot)
        self.addAction(act)  # shortcuts work even when the toolbar is hidden
        return act

    def _build_actions(self) -> None:
        a = self._action
        self.act_open = a("Open…", self.open_dialog, QKeySequence.StandardKey.Open, "Open an image")
        self.act_save = a("Save PNG", self.save, QKeySequence.StandardKey.Save, "Save the cutout as a transparent PNG")
        self.act_save_as = a("Save PNG As…", self.save_as, QKeySequence("Ctrl+Shift+S"), "Save to a new file")
        self.act_close = a("Close Image", self.close_image, QKeySequence.StandardKey.Close, "Close the image")
        self.act_quit = a("Exit", self.close, QKeySequence("Ctrl+Q"))
        self.act_undo = self.doc.undo_stack.createUndoAction(self, "Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = self.doc.undo_stack.createRedoAction(self, "Redo")
        self.act_redo.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        self.addAction(self.act_undo)
        self.addAction(self.act_redo)
        self.act_rerun = a("Remove Background Again", self.start_removal, QKeySequence("Ctrl+R"),
                           "Run the automatic removal again (undoable)")
        self.act_fit = a("Fit to Window", self.canvas.fit_to_window, QKeySequence("Ctrl+0"), "Fit the image in the window")
        self.act_100 = a("Actual Pixels", self.canvas.actual_pixels, QKeySequence("Ctrl+1"), "Zoom to 100%")
        self.act_zoom_in = a("Zoom In", self.canvas.zoom_in, QKeySequence.StandardKey.ZoomIn, "Zoom in")
        self.act_zoom_out = a("Zoom Out", self.canvas.zoom_out, QKeySequence.StandardKey.ZoomOut, "Zoom out")
        self.act_original = a("Show Original", self._toggle_original, QKeySequence("O"),
                              "Compare with the original image", checkable=True)

        self.backdrop_group = QActionGroup(self)
        self.act_backdrops: dict[Backdrop, QAction] = {}
        for bd, label, key in ((Backdrop.CHECKER, "Checkerboard", "1"), (Backdrop.WHITE, "White", "2"),
                               (Backdrop.BLACK, "Black", "3")):
            act = a(label, None, QKeySequence(f"Alt+{key}"), f"Show the cutout on {label.lower()}", checkable=True)
            act.triggered.connect(lambda _c=False, b=bd: self.set_backdrop(b))
            self.backdrop_group.addAction(act)
            self.act_backdrops[bd] = act
        self.act_backdrops[Backdrop.CHECKER].setChecked(True)

        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.toolbar = tb

        tb.addAction(self.act_open)
        tb.addSeparator()

        # Subject: a two-way segmented switch.
        tb.addWidget(self._caption("Subject"))
        self.subject_group = QButtonGroup(self)
        self.subject_buttons: dict[Subject, QToolButton] = {}
        seg = QWidget()
        seg.setObjectName("segmented")
        row = QHBoxLayout(seg)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        for subject, label, tip in ((Subject.PERSON, "Person", "Photos of people"),
                                    (Subject.ANIME, "Anime", "Anime, cartoon and drawn characters")):
            b = QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setObjectName("segLeft" if subject is Subject.PERSON else "segRight")
            self.subject_group.addButton(b)
            self.subject_buttons[subject] = b
            row.addWidget(b)
        self.subject_buttons[Subject.PERSON].setChecked(True)
        self.subject_group.buttonClicked.connect(self._on_subject_clicked)
        tb.addWidget(seg)
        tb.addAction(self.act_rerun)
        self.act_rerun.setText("Redo Auto")
        tb.addSeparator()

        # Tools are inserted before this marker by register_tool().
        self._tools_marker = tb.addSeparator()

        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        tb.addAction(self.act_original)
        self.backdrop_combo = QComboBox()
        self.backdrop_combo.setToolTip("Background shown behind the cutout (preview only)")
        for bd, label in ((Backdrop.CHECKER, "Checkerboard"), (Backdrop.WHITE, "White"), (Backdrop.BLACK, "Black")):
            self.backdrop_combo.addItem(label, bd)
        self.backdrop_combo.currentIndexChanged.connect(
            lambda i: self.set_backdrop(self.backdrop_combo.itemData(i)))
        tb.addWidget(self.backdrop_combo)
        tb.addSeparator()
        tb.addAction(self.act_zoom_out)
        self.act_zoom_out.setText("−")
        tb.addAction(self.act_fit)
        self.act_fit.setText("Fit")
        tb.addAction(self.act_zoom_in)
        self.act_zoom_in.setText("+")
        tb.addSeparator()

        self.save_button = QPushButton("Save PNG")
        self.save_button.setObjectName("primaryButton")
        self.save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_button.setToolTip(self.act_save.toolTip())
        self.save_button.clicked.connect(self.save)
        tb.addWidget(self.save_button)

    def _caption(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("toolbarCaption")
        return lab

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        for act in (self.act_open, None, self.act_save, self.act_save_as, None, self.act_close, None, self.act_quit):
            m.addSeparator() if act is None else m.addAction(act)
        m = mb.addMenu("&Edit")
        m.addAction(self.act_undo)
        m.addAction(self.act_redo)
        m.addSeparator()
        m.addAction(self.act_rerun)
        self.tools_menu = mb.addMenu("&Tools")
        m = mb.addMenu("&View")
        for act in (self.act_fit, self.act_100, self.act_zoom_in, self.act_zoom_out, None, self.act_original):
            m.addSeparator() if act is None else m.addAction(act)
        bm = m.addMenu("Background")
        for act in self.act_backdrops.values():
            bm.addAction(act)
        # Menu texts read better than the compact toolbar labels.
        self.act_fit.setIconText("Fit")

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.lbl_file = QLabel()
        self.lbl_model = QLabel()
        self.lbl_model.setObjectName("modelBadge")
        self.lbl_pixel = QLabel()
        self.lbl_zoom = QLabel()
        for w in (self.lbl_file, self.lbl_model, self.lbl_pixel, self.lbl_zoom):
            sb.addPermanentWidget(w)
        self.lbl_zoom.setMinimumWidth(52)
        self.lbl_pixel.setMinimumWidth(140)

    # ------------------------------------------------------------- tools API
    def tool_context(self) -> ToolContext:
        return ToolContext(
            document=self.doc,
            canvas=self.canvas,
            run_in_background=self.run_in_background,
            show_message=lambda msg: self.statusBar().showMessage(msg, 4000),
        )

    def register_tool(self, tool: CanvasTool, shortcut: str | None = None) -> QAction:
        """Add a touch-up tool to the toolbar and Tools menu."""
        if tool.id in self._tools:
            raise ValueError(f"tool {tool.id!r} already registered")
        act = QAction(tool.label, self)
        act.setCheckable(True)
        if tool.icon is not None:
            act.setIcon(tool.icon)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.setToolTip(tool.tooltip or tool.label)
        act.setStatusTip(tool.tooltip or tool.label)
        act.triggered.connect(lambda _c=False, tid=tool.id: self.activate_tool(tid))
        self.tool_group.addAction(act)
        self.toolbar.insertAction(self._tools_marker, act)
        self.tools_menu.addAction(act)
        self.addAction(act)
        self._tools[tool.id] = tool
        self._tool_actions[tool.id] = act
        self._sync_actions()
        return act

    def activate_tool(self, tool_id: str) -> None:
        tool = self._tools[tool_id]
        if self._active_tool is tool:
            return
        if self._active_tool is not None:
            self._active_tool.deactivate()
        self._active_tool = tool
        tool.activate(self.tool_context())
        self._tool_actions[tool_id].setChecked(True)
        self.canvas.set_tool(tool)

    @property
    def active_tool(self) -> CanvasTool | None:
        return self._active_tool

    def run_in_background(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        task = Task(fn, *args)
        # Hold a Python reference until a signal arrives so the signals object is not collected.
        self._live_tasks.add(task.signals)

        def done(value, s=task.signals):
            self._live_tasks.discard(s)
            if on_done is not None:
                on_done(value)

        def failed(exc, s=task.signals):
            self._live_tasks.discard(s)
            if on_error is not None:
                on_error(exc)
            else:
                self._report_error("Something went wrong", exc)

        task.signals.finished.connect(done)
        task.signals.failed.connect(failed)
        self.pool.start(task)

    # ----------------------------------------------------------- busy state
    def _push_busy(self, text: str) -> None:
        self._busy += 1
        self.overlay.start(text)
        self.statusBar().showMessage(text)
        self._sync_actions()

    def _pop_busy(self) -> None:
        self._busy = max(0, self._busy - 1)
        if self._busy == 0:
            self.overlay.stop()
            self.statusBar().clearMessage()
        self._sync_actions()

    @property
    def is_busy(self) -> bool:
        return self._busy > 0

    def _sync_actions(self) -> None:
        busy = self.is_busy
        has_img = self.doc.has_image
        has_alpha = self.doc.has_alpha
        self.act_open.setEnabled(not busy)
        for act in (self.act_save, self.act_save_as):
            act.setEnabled(has_alpha and not busy)
        self.save_button.setEnabled(has_alpha and not busy)
        self.act_close.setEnabled(has_img and not busy)
        self.act_rerun.setEnabled(has_img and not busy)
        for act in (self.act_fit, self.act_100, self.act_zoom_in, self.act_zoom_out):
            act.setEnabled(has_img)
        self.act_original.setEnabled(has_alpha)
        self.act_undo.setEnabled(self.doc.undo_stack.canUndo() and not busy)
        self.act_redo.setEnabled(self.doc.undo_stack.canRedo() and not busy)
        for b in self.subject_buttons.values():
            b.setEnabled(not busy)
        for tid, act in self._tool_actions.items():
            tool = self._tools[tid]
            act.setEnabled((has_alpha or not tool.needs_alpha) and not busy)

    # -------------------------------------------------------------- opening
    def open_dialog(self) -> None:
        if self.is_busy or not self._confirm_discard():
            return
        start = self.settings.value("last_open_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open image", str(start), OPEN_FILTER)
        if path:
            self.open_path(path, confirmed=True)

    def open_path(self, path: str | os.PathLike, confirmed: bool = False) -> None:
        """Load an image and start the automatic removal on it."""
        path = Path(path)
        if self.is_busy:
            return
        if not confirmed and not self._confirm_discard():
            return
        if not path.is_file():
            self._report_error("Can't open file", FileNotFoundError(f"{path} does not exist."))
            return
        if not is_supported_image(path):
            self._report_error("Unsupported file", ValueError(
                f"{path.name} is not a supported image. Use JPG, PNG, WebP, TIFF or BMP."))
            return
        self.settings.setValue("last_open_dir", str(path.parent))
        self._push_busy(f"Opening {path.name}…")
        self.run_in_background(load_image, path, on_done=self._on_loaded, on_error=self._on_load_failed)

    def _on_loaded(self, image: LoadedImage) -> None:
        self._pop_busy()
        self._last_save_path = None
        self.doc.set_image(image)
        self.start_removal()

    def _on_load_failed(self, exc: BaseException) -> None:
        self._pop_busy()
        self._report_error("Can't open image", exc)

    def close_image(self) -> None:
        if self.is_busy or not self._confirm_discard():
            return
        self._generation += 1
        self.doc.clear()

    # -------------------------------------------------------------- removal
    @property
    def subject(self) -> Subject:
        for subject, b in self.subject_buttons.items():
            if b.isChecked():
                return subject
        return Subject.PERSON

    def set_subject(self, subject: Subject, rerun: bool = True) -> None:
        self.subject_buttons[Subject(subject)].setChecked(True)
        self.settings.setValue("subject", Subject(subject).value)
        if rerun and self.doc.has_image and not self.is_busy:
            self.start_removal()

    def _on_subject_clicked(self, _button) -> None:
        self.set_subject(self.subject, rerun=True)

    def start_removal(self) -> None:
        image = self.doc.image
        if image is None or self.is_busy:
            return
        self._generation += 1
        gen = self._generation
        subject = self.subject
        self._push_busy("Removing background…")
        self.run_in_background(
            self.engine.remove_background, image, subject,
            on_done=lambda r: self._on_removed(gen, image, r),
            on_error=lambda e: self._on_remove_failed(gen, subject, e),
        )

    def _on_removed(self, gen: int, image: LoadedImage, result: CutoutResult) -> None:
        self._pop_busy()
        if gen != self._generation or image is not self.doc.image:
            return  # superseded by another open / run
        self.doc.set_auto_result(result)
        if result.notice:
            self.statusBar().showMessage(result.notice, 15000)
        self.cutout_ready.emit()

    def _on_remove_failed(self, gen: int, subject: Subject, exc: BaseException) -> None:
        self._pop_busy()
        if gen != self._generation:
            return
        if isinstance(exc, ModelNotInstalled):
            self._report_error(
                "Model not installed",
                exc,
                f"No {subject.value} model was found, so the background can't be removed yet.\n\n"
                f"Models folder:\n{self.engine.models_dir}\n\n"
                "Reinstall Cutout, or copy the model files into that folder.",
            )
        elif isinstance(exc, GpuOutOfMemory):
            self._report_error("Out of memory", exc)
        else:
            self._report_error("Background removal failed", exc)

    def warmup_in_background(self) -> None:
        """Load the model for the current subject while the user picks a file."""
        def warm(subject=self.subject):
            try:
                self.engine.warmup(subject)
            except ModelNotInstalled:
                pass  # reported when the user actually runs a removal
        self.run_in_background(warm, on_error=lambda e: log.warning("warmup failed: %s", e))

    # --------------------------------------------------------------- saving
    def save(self) -> None:
        if not self.doc.has_alpha or self.is_busy:
            return
        if self._last_save_path is None:
            self.save_as()
        else:
            self.save_to(self._last_save_path)

    def save_as(self) -> None:
        if not self.doc.has_alpha or self.is_busy:
            return
        suggestion = self._last_save_path or default_output_path(self.doc.path or Path.home() / "cutout.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save transparent PNG", str(suggestion), "PNG image (*.png)")
        if path:
            self.save_to(path)

    def save_to(self, path: str | os.PathLike) -> None:
        """Compose at full resolution and write the PNG on a worker thread."""
        path = Path(path)
        if path.suffix.lower() != ".png":
            path = path.with_name(path.name + ".png")
        if self.doc.path is not None and path.resolve() == self.doc.path.resolve():
            self._report_error("Choose another name", ValueError(
                "Saving over the original image would destroy it. Pick a different file name."))
            return
        image = self.doc.image
        decon = self.doc.result.decontaminate if self.doc.result is not None else True
        undo_index = self.doc.undo_stack.index()
        ready = self._prepared_save(image, decon)
        alpha = self.doc.snapshot_alpha()  # the fallback if the prepared save failed
        prep = self.doc.result.prep if self.doc.result is not None else None
        self._push_busy(f"Saving {path.name}…")

        def work():
            t = time.perf_counter()
            payload = None
            if ready is not None:
                try:
                    payload = ready.result()  # usually finished already; waits if not
                except Exception as exc:  # noqa: BLE001 - fall back to a fresh encode
                    log.warning("prepared save failed, encoding again: %s", exc)
            fresh = payload is None
            if payload is None:
                rgba = compose_rgba(image, alpha, decon, prep)
                payload = encode_rgba_png(rgba, image.icc_profile)
                del rgba
            t_write = time.perf_counter()
            write_png_bytes(path, payload)
            steps = {"compose+encode" if fresh else "wait for prepared PNG": t_write - t,
                     "write": time.perf_counter() - t_write}
            debug.record_timings(f"save {image.width}x{image.height}", steps,
                                 prepared_in_background=not fresh, size=f"{len(payload) / 1e6:.1f} MB")
            log.info("saved %s in %.2fs (%s)", path.name, time.perf_counter() - t,
                     "composed on save" if fresh else "prepared in the background")
            return path

        def done(p: Path):
            self._pop_busy()
            self._last_save_path = p
            if self.doc.image is image and self.doc.undo_stack.index() == undo_index:
                self.doc.mark_saved()
            self._update_title()
            self.statusBar().showMessage(f"Saved {p}", 6000)
            self.saved.emit(str(p))

        def failed(exc: BaseException):
            self._pop_busy()
            self._report_error("Couldn't save the PNG", exc)

        self.run_in_background(work, on_done=done, on_error=failed)

    def _prepare_save_in_background(self) -> None:
        """Compose and encode the PNG on the CPU as soon as a cutout is shown, so Save only
        has to write the file. Thrown away if the matte changes before the user saves."""
        doc = self.doc
        if doc.image is None or doc.alpha is None:
            self._save_prep = None
            return
        image, alpha = doc.image, doc.snapshot_alpha()
        decon = doc.result.decontaminate if doc.result is not None else True
        prep = doc.result.prep if doc.result is not None else None

        def work():
            t = time.perf_counter()
            payload = encode_rgba_png(compose_rgba(image, alpha, decon, prep), image.icc_profile)
            debug.record_timings(f"save prepared in background {image.width}x{image.height}",
                                 {"compose+encode": time.perf_counter() - t}, size=f"{len(payload) / 1e6:.1f} MB")
            return payload

        self._save_prep = (image, doc.revision, decon, parallel.submit(work))

    def _prepared_save(self, image: LoadedImage, decon: bool):
        """The background-prepared PNG future for the matte as it is now, or ``None``."""
        sp = self._save_prep
        if sp is None:
            return None
        p_image, p_rev, p_decon, future = sp
        if p_image is image and p_rev == self.doc.revision and p_decon == decon and not future.cancelled():
            return future
        return None

    # ---------------------------------------------------------------- view
    def set_backdrop(self, backdrop: Backdrop) -> None:
        backdrop = Backdrop(backdrop)
        self.canvas.set_backdrop(backdrop)
        self.act_backdrops[backdrop].setChecked(True)
        idx = self.backdrop_combo.findData(backdrop)
        if idx >= 0 and idx != self.backdrop_combo.currentIndex():
            self.backdrop_combo.blockSignals(True)
            self.backdrop_combo.setCurrentIndex(idx)
            self.backdrop_combo.blockSignals(False)
        self.settings.setValue("backdrop", backdrop.value)

    def _toggle_original(self, on: bool) -> None:
        self.canvas.set_show_original(on)

    # ------------------------------------------------------- status / title
    def _on_image_changed(self) -> None:
        img = self.doc.image
        self._save_prep = None  # the old image's prepared PNG is no use any more
        self.act_original.setChecked(False)
        self.canvas.set_show_original(False)
        if img is None:
            self.stack.setCurrentWidget(self.empty)
            self.lbl_file.clear()
            self.lbl_model.clear()
            self.lbl_pixel.clear()
            self.lbl_zoom.clear()
        else:
            self.stack.setCurrentWidget(self.canvas)
            depth = "16-bit" if img.max_value > 255 else "8-bit"
            name = img.source_path.name if img.source_path else "image"
            self.lbl_file.setText(f"{name}  ·  {img.width} × {img.height}  ·  {depth}")
            self.lbl_model.clear()
            self.canvas.setFocus()
        self._sync_actions()
        self._update_title()

    def _on_result_changed(self) -> None:
        r = self.doc.result
        if r is None:
            self.lbl_model.clear()
            return
        self._prepare_save_in_background()
        spec = REGISTRY.get(r.model_id)
        name = spec.id if spec else r.model_id
        device = r.device.upper()
        total = sum(r.timings.values())
        self.lbl_model.setText(f"{name}  ·  {device}  ·  {total:.1f} s")
        self.lbl_model.setProperty("cpu", r.device == "cpu")
        self.lbl_model.setToolTip(
            "Running on the CPU: the NVIDIA GPU could not be used, so removal is slower."
            if r.device == "cpu" else "Model and device used for the automatic removal"
        )
        self.lbl_model.style().unpolish(self.lbl_model)
        self.lbl_model.style().polish(self.lbl_model)
        self._sync_actions()

    def _on_zoom_changed(self, z: float) -> None:
        self.lbl_zoom.setText(f"{z * 100:.0f}%")

    def _on_hover_pixel(self, x: int, y: int) -> None:
        if x < 0:
            self.lbl_pixel.clear()
            return
        text = f"{x}, {y}"
        if self.doc.alpha is not None:
            text += f"  ·  {self.doc.alpha[y, x] * 100:.0f}% opaque"
        self.lbl_pixel.setText(text)

    @Slot()
    def _on_doc_edited(self, *_args) -> None:
        self._update_title()

    def _update_title(self) -> None:
        if self.doc.image is None:
            self.setWindowTitle(APP_NAME)
            return
        name = self.doc.path.name if self.doc.path else "image"
        mark = " •" if self.doc.is_modified else ""
        self.setWindowTitle(f"{name}{mark} — {APP_NAME}")
        self._sync_actions()

    # --------------------------------------------------------- drag & drop
    def _dropped_path(self, event) -> Path | None:
        md = event.mimeData()
        if not md.hasUrls():
            return None
        urls = [u for u in md.urls() if u.isLocalFile()]
        if len(urls) != 1:
            return None
        p = Path(urls[0].toLocalFile())
        return p if is_supported_image(p) else None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not self.is_busy and self._dropped_path(event) is not None:
            event.acceptProposedAction()
            self.empty.set_drag_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.empty.set_drag_hover(False)

    def dropEvent(self, event: QDropEvent) -> None:
        self.empty.set_drag_hover(False)
        path = self._dropped_path(event)
        if path is not None:
            event.acceptProposedAction()
            # Defer so the drag source (Explorer) is released before any dialog appears.
            QTimer.singleShot(0, lambda: self.open_path(path))

    # ------------------------------------------------------- dialogs / exit
    def _confirm_discard(self) -> bool:
        if not self.doc.is_modified or not self.ask_user:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(APP_NAME)
        box.setText("Save the cutout before continuing?")
        box.setInformativeText("Your changes will be lost if you don't save them.")
        box.setStandardButtons(
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            # Saving is asynchronous; the user continues after it finishes.
            self.save()
            return False
        return choice == QMessageBox.StandardButton.Discard

    def _report_error(self, title: str, exc: BaseException, text: str | None = None) -> None:
        message = text or (str(exc) or exc.__class__.__name__)
        log.warning("%s: %s", title, exc)
        self.task_failed.emit(f"{title}: {message}")
        self.statusBar().showMessage(title, 6000)
        if self.ask_user:
            debug_menu.show_error(self, title, message, exc)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_busy and self.ask_user:
            event.ignore()
            self.statusBar().showMessage("Please wait for the current step to finish.", 4000)
            return
        if not self._confirm_discard():
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def _restore_settings(self) -> None:
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        try:
            self.set_subject(Subject(self.settings.value("subject", Subject.PERSON.value)), rerun=False)
        except ValueError:
            pass
        try:
            self.set_backdrop(Backdrop(self.settings.value("backdrop", Backdrop.CHECKER.value)))
        except ValueError:
            pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.overlay.isVisible():
            self.overlay.setGeometry(self.stack.rect())
