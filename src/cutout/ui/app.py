"""App entry point: theme, QApplication, main window.

    python -m cutout.ui [--debug] [image]
    cutout-gui [image]
    from cutout.ui import main; main(["photo.jpg"])   # what the packaged Cutout.exe does
"""

from __future__ import annotations

import sys
from pathlib import Path

import logging

from PySide6.QtCore import QSettings, Qt, QTimer, QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from ..core import debug
from . import debug_menu, update_ui
from .main_window import APP_NAME, MainWindow

ACCENT = "#5b8def"

STYLE = f"""
QMainWindow, QStatusBar {{ background: #1f2023; }}
QToolBar#mainToolbar {{
    background: #2a2b2f; border: none; border-bottom: 1px solid #36373c;
    padding: 6px 10px; spacing: 6px;
}}
QToolBar#mainToolbar QToolButton {{
    color: #e8eaed; background: transparent; border: 1px solid transparent;
    border-radius: 6px; padding: 5px 10px;
}}
QToolBar#mainToolbar QToolButton:hover {{ background: #36383d; }}
QToolBar#mainToolbar QToolButton:checked {{ background: #3a4a6b; border-color: {ACCENT}; }}
QToolBar#mainToolbar QToolButton:disabled {{ color: #6b6d72; }}
QToolBar#mainToolbar::separator {{ background: #3a3b40; width: 1px; margin: 4px 6px; }}
QLabel#toolbarCaption {{ color: #9aa0a6; padding-right: 4px; }}

QWidget#segmented QToolButton {{
    background: #1f2023; border: 1px solid #44464c; padding: 5px 14px; border-radius: 0;
}}
QWidget#segmented QToolButton#segLeft {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QWidget#segmented QToolButton#segRight {{
    border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-left: none;
}}
QWidget#segmented QToolButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}

QPushButton#primaryButton {{
    background: {ACCENT}; color: white; border: none; border-radius: 7px;
    padding: 7px 18px; font-weight: 600;
}}
QPushButton#primaryButton:hover {{ background: #6f9cf2; }}
QPushButton#primaryButton:pressed {{ background: #4a7bd9; }}
QPushButton#primaryButton:disabled {{ background: #3a3c42; color: #7d8086; }}

QComboBox {{
    background: #1f2023; color: #e8eaed; border: 1px solid #44464c; border-radius: 6px;
    padding: 4px 10px; min-width: 110px;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: #2a2b2f; color: #e8eaed; border: 1px solid #44464c; selection-background-color: #3a4a6b;
}}

QLabel#emptyTitle {{ color: #e8eaed; font-size: 22pt; font-weight: 600; }}
QLabel#emptySub {{ color: #bdc1c6; font-size: 11pt; }}
QLabel#emptyHint {{ color: #80868b; }}
QLabel#warmupLabel {{ color: #bdc1c6; }}
QProgressBar#warmupBar {{ background: #36373c; border: none; border-radius: 3px; }}
QProgressBar#warmupBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QStatusBar {{ color: #bdc1c6; border-top: 1px solid #36373c; }}
QStatusBar QLabel {{ color: #bdc1c6; padding: 0 10px; }}
QLabel#modelBadge[cpu="true"] {{ color: #f6b26b; }}

QMenuBar {{ background: #2a2b2f; color: #e8eaed; }}
QMenuBar::item:selected {{ background: #36383d; }}
QMenu {{ background: #2a2b2f; color: #e8eaed; border: 1px solid #3a3b40; }}
QMenu::item:selected {{ background: #3a4a6b; }}
QToolTip {{ background: #2a2b2f; color: #e8eaed; border: 1px solid #44464c; padding: 4px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: "#1f2023",
        QPalette.ColorRole.WindowText: "#e8eaed",
        QPalette.ColorRole.Base: "#1a1b1e",
        QPalette.ColorRole.AlternateBase: "#26272b",
        QPalette.ColorRole.Text: "#e8eaed",
        QPalette.ColorRole.Button: "#2a2b2f",
        QPalette.ColorRole.ButtonText: "#e8eaed",
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.ToolTipBase: "#2a2b2f",
        QPalette.ColorRole.ToolTipText: "#e8eaed",
        QPalette.ColorRole.PlaceholderText: "#80868b",
        QPalette.ColorRole.Link: ACCENT,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor("#6b6d72"))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def _set_windows_app_id() -> None:
    """Group the taskbar button under our own id instead of python.exe's."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Cutout.App")
        except Exception:  # noqa: BLE001 - purely cosmetic
            pass


_QT_LEVELS = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def _qt_message(mode, _context, message) -> None:
    logging.getLogger("qt").log(_QT_LEVELS.get(mode, logging.INFO), "%s", message)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    debug_flag = "--debug" in args
    args = [a for a in args if a != "--debug"]
    # The saved Help > Debug Mode choice is read before Qt starts, so the whole run is logged.
    saved = debug_menu.saved_preference(QSettings(APP_NAME, APP_NAME))
    debug.setup(debug=debug_flag or saved or debug.env_requests_debug())
    qInstallMessageHandler(_qt_message)
    _set_windows_app_id()
    app = QApplication.instance() or QApplication([sys.argv[0], *[a for a in args if a.startswith("-")]])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    apply_theme(app)

    window = MainWindow()
    window.show()
    QTimer.singleShot(0, lambda: debug_menu.offer_crash_report(window))
    QTimer.singleShot(0, lambda: update_ui.on_startup(window))
    window.warmup_in_background()

    files = [a for a in args if not a.startswith("-")]
    if files:
        path = Path(files[0])
        QTimer.singleShot(0, lambda: window.open_path(path))
    code = app.exec()
    debug.mark_clean_exit()
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
