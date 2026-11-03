"""Help menu: debug mode toggle and the copy-paste debug report (headless)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtCore")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import FakeSegmenter  # noqa: E402
from cutout.core import debug  # noqa: E402
from cutout.core.engine import CutoutEngine  # noqa: E402
from cutout.ui import debug_menu  # noqa: E402
from cutout.ui.main_window import MainWindow  # noqa: E402
from test_ui import wait_until  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    debug._reset_for_tests()
    debug.setup(console=False, directory=tmp_path / "logs")
    engine = CutoutEngine(models_dir=tmp_path / "models", segmenter_factory=FakeSegmenter)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(engine=engine, settings=settings, ask_user=False)
    w.show()
    yield w
    w.close()
    debug._reset_for_tests()


def test_help_menu_toggles_and_saves_debug_mode(window):
    titles = [a.text() for a in window.menuBar().actions()]
    assert "&Help" in titles
    assert not window.act_debug.isChecked() and not window.lbl_debug.isVisibleTo(window)
    window.act_debug.trigger()
    assert debug.enabled()
    assert window.lbl_debug.isVisibleTo(window)
    assert debug_menu.saved_preference(window.settings)
    window.act_debug.trigger()
    assert not debug.enabled() and not debug_menu.saved_preference(window.settings)


def test_copy_report_puts_report_on_clipboard(window):
    QGuiApplication.clipboard().setText("")
    window.open_path(window.engine.models_dir / "missing.png")  # a failure the user would report
    debug_menu.copy_report(window, "test failure")
    wait_until(lambda: QGuiApplication.clipboard().text().startswith("CUTOUT DEBUG REPORT"))
    text = QGuiApplication.clipboard().text()
    assert "test failure" in text and "does not exist" in text


def test_save_report_writes_file(window):
    debug_menu.save_report_dialog(window)
    target = debug.log_dir() / "cutout-report.txt"
    wait_until(target.exists)
    assert "===== Models" in target.read_text(encoding="utf-8")
