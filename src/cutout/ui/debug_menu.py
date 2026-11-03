"""Help menu: debug mode switch, copy/save a debug report, open the log folder.

Also provides the error dialog (with a "Copy report" button) and the
"Cutout closed unexpectedly last time" prompt. ``MainWindow`` calls
``install(window)`` once; everything else hangs off the window's signals so the
editor code stays untouched.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox, QWidget

from ..core import debug

if TYPE_CHECKING:  # pragma: no cover
    from .main_window import MainWindow

log = logging.getLogger(__name__)

SETTINGS_KEY = "debug/enabled"
COPIED_HINT = "Debug report copied. Paste it into the chat with Claude."


def saved_preference(settings) -> bool:
    value = settings.value(SETTINGS_KEY, False)
    return value in (True, "true", "1", 1)


def install(window: "MainWindow") -> None:
    mb = window.menuBar()
    m = mb.addMenu("&Help")
    window.help_menu = m  # update_ui adds the update items below these

    act_debug = QAction("Debug Mode", window)
    act_debug.setCheckable(True)
    act_debug.setChecked(debug.enabled())
    act_debug.setStatusTip("Write detailed logs so problems can be diagnosed")
    act_debug.toggled.connect(lambda on: _set_debug(window, on))
    m.addAction(act_debug)
    m.addSeparator()

    act_copy = QAction("Copy Debug Report", window)
    act_copy.setStatusTip("Copy a report of this PC, the GPU and the recent log, to paste into a chat")
    act_copy.triggered.connect(lambda: copy_report(window))
    m.addAction(act_copy)

    act_save = QAction("Save Debug Report…", window)
    act_save.triggered.connect(lambda: save_report_dialog(window))
    m.addAction(act_save)

    act_folder = QAction("Open Log Folder", window)
    act_folder.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(debug.log_dir()))))
    m.addAction(act_folder)

    window.act_debug = act_debug
    window.act_copy_report = act_copy
    window.act_save_report = act_save

    badge = QLabel("DEBUG MODE")
    badge.setObjectName("debugBadge")
    badge.setStyleSheet("color: #f6b26b; font-weight: 600;")
    badge.setToolTip("Detailed logging is on. Help > Copy Debug Report to share a problem.")
    badge.setVisible(debug.enabled())
    window.statusBar().insertPermanentWidget(0, badge)  # status messages would hide a normal widget
    window.lbl_debug = badge

    _log_user_actions(window)


def _set_debug(window: "MainWindow", on: bool) -> None:
    if on == debug.enabled():
        window.lbl_debug.setVisible(on)
        return
    debug.set_enabled(on)
    window.settings.setValue(SETTINGS_KEY, on)
    window.lbl_debug.setVisible(on)
    if on:
        window.statusBar().showMessage(
            "Debug mode is on. Restart Cutout to also capture GPU driver messages.", 8000)
    else:
        window.statusBar().showMessage("Debug mode is off.", 4000)


def _log_user_actions(window: "MainWindow") -> None:
    """Breadcrumbs: what the user did just before a failure."""
    def on_image():
        img = window.doc.image
        if img is None:
            log.info("image closed")
        else:
            log.info("opened image %s: %dx%d, %s", getattr(img.source_path, "name", "?"),
                     img.width, img.height, img.rgb.dtype)

    def on_result():
        res = window.doc.result
        if res is None:
            return
        log.info("cutout done: %dx%d px, subject %s, model %s on %s, %s",
                 res.image.width, res.image.height, window.subject.value, res.model_id, res.device,
                 ", ".join(f"{k} {v:.2f}s" for k, v in res.timings.items()))
        if debug.enabled():  # nvidia-smi takes a moment: keep it off the UI thread
            window.run_in_background(debug.log_memory, "after cutout", on_error=lambda _e: None)

    window.doc.image_changed.connect(on_image)
    window.doc.result_changed.connect(on_result)
    window.act_rerun.triggered.connect(lambda: log.info("remove background again"))
    window.subject_group.buttonClicked.connect(lambda _b: log.info("subject set to %s", window.subject.value))
    window.cutout_ready.connect(lambda: log.info("cutout shown"))
    window.saved.connect(lambda p: log.info("saved %s", Path(p).name))
    window.task_failed.connect(lambda msg: log.error("shown to user: %s", msg))


# ----------------------------------------------------------------- reports
def _models_dir(window) -> Path | None:
    return getattr(getattr(window, "engine", None), "models_dir", None)


def _with_report(window, extra: str, then) -> None:
    """Build the report off the UI thread (it queries the GPU and loads DLLs), then call ``then``."""
    window.statusBar().showMessage("Building debug report…")
    window.run_in_background(
        debug.build_report, extra, _models_dir(window),
        on_done=then,
        on_error=lambda e: window.statusBar().showMessage(f"Could not build the report: {e}", 8000),
    )


def copy_report(window, extra: str = "") -> None:
    def done(text: str):
        QGuiApplication.clipboard().setText(text)
        window.statusBar().showMessage(COPIED_HINT, 10000)
        if getattr(window, "ask_user", False):
            QMessageBox.information(window, "Debug report copied", COPIED_HINT)

    _with_report(window, extra, done)


def save_report_dialog(window, extra: str = "") -> None:
    suggestion = debug.log_dir() / "cutout-report.txt"
    if getattr(window, "ask_user", False):
        path, _ = QFileDialog.getSaveFileName(window, "Save debug report", str(suggestion), "Text file (*.txt)")
        if not path:
            return
    else:
        path = str(suggestion)

    def done(text: str):
        Path(path).write_text(text, encoding="utf-8")
        window.statusBar().showMessage(f"Debug report saved to {path}", 10000)

    _with_report(window, extra, done)


# ----------------------------------------------------------------- dialogs
def show_error(window, title: str, message: str, exc: BaseException | None = None) -> None:
    """A warning box with a "Copy Report" button, and the traceback under Details in debug mode."""
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(message)
    trace = ""
    if exc is not None:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if debug.enabled():
            box.setDetailedText(trace)
    copy_btn = box.addButton("Copy Report", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Ok)
    box.exec()
    if box.clickedButton() is copy_btn:
        copy_report(window, f"{title}: {message}\n\n{trace}".strip())


def offer_crash_report(window: QWidget) -> None:
    """If the last run ended without closing normally, offer to copy a report about it."""
    if not debug.previous_run_crashed():
        return
    box = QMessageBox(window)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Cutout closed unexpectedly")
    box.setText("Cutout didn't close normally last time.\n\n"
                "Copy a debug report to paste into the chat so the problem can be fixed?")
    copy_btn = box.addButton("Copy Report", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is copy_btn:
        copy_report(window, "The previous run of Cutout closed unexpectedly.")
