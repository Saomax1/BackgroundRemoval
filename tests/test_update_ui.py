"""Help > Check for Updates and the update / finish-setup dialog (headless)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtCore")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import FakeSegmenter  # noqa: E402
from cutout.core.engine import CutoutEngine  # noqa: E402
from cutout.ui import update_ui  # noqa: E402
from cutout.ui.main_window import MainWindow  # noqa: E402
from cutout.update import client  # noqa: E402
from test_ui import wait_until  # noqa: E402
from test_updates import FILES, make_release, release  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("CUTOUT_UPDATES_DIR", str(tmp_path / "updates"))
    monkeypatch.delenv("CUTOUT_APP_DIR", raising=False)
    engine = CutoutEngine(models_dir=tmp_path / "models", segmenter_factory=FakeSegmenter)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(engine=engine, settings=settings, ask_user=False)
    w.show()
    yield w
    w.close()


def test_help_menu_has_update_items(window):
    texts = [a.text() for a in window.help_menu.actions()]
    assert "Check for Updates…" in texts and "Check for Updates at Startup" in texts
    assert not window.act_auto_updates.isChecked()  # offline by default
    window.act_auto_updates.setChecked(True)
    assert update_ui._flag(window.settings.value(update_ui.STARTUP_CHECK_KEY))


def test_check_from_source_explains_instead_of_going_online(window, monkeypatch):
    monkeypatch.setattr(client, "fetch_manifest", lambda *a: pytest.fail("must not go online"))
    update_ui.check_now(window, manual=True)
    assert "installed app" in window.statusBar().currentMessage()


def test_startup_reports_a_failed_update(window):
    client.result_path().parent.mkdir(parents=True)
    client.result_path().write_text('{"ok": false, "message": "Your previous version is still installed."}')
    update_ui.on_startup(window)
    assert "previous version" in window.statusBar().currentMessage()


def test_first_start_offers_the_download_and_installs(window, tmp_path, monkeypatch):
    app, out, manifest, _ = release(tmp_path, "0.1.1")
    install = tmp_path / "install"
    make_release.stage_setup(app, manifest, install)
    monkeypatch.setenv("CUTOUT_APP_DIR", str(install))
    launched, quit_called = [], []
    monkeypatch.setattr(client, "launch_apply", lambda p: launched.append(p))
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quit_called.append(True)))

    update_ui.on_startup(window)
    wait_until(lambda: getattr(window, "update_dialog", None) is not None)
    dlg = window.update_dialog
    assert dlg.first_run and dlg.isVisible() and ("GB" in dlg.body.text() or "MB" in dlg.body.text())
    dlg.start(mirrors=[out])
    wait_until(lambda: launched)
    assert quit_called and not window.isVisible()
    assert launched[0].name == "apply-plan.json"


def test_update_dialog_shows_notes_and_size(window, tmp_path):
    _a1, _o1, v1, _ = release(tmp_path, "0.1.1")
    _a2, _o2, v2, _ = release(tmp_path, "0.1.2", {**FILES, "Cutout.exe": b"v2"}, previous=v1)
    status = client.local_status(root=_a1)
    plan = client.plan_update(status, v2)
    dlg = update_ui.UpdateDialog(window, status, plan)
    assert "0.1.2 is available" in dlg.heading.text() and "0.1.1" in dlg.heading.text()
    assert "notes for 0.1.2" in dlg.notes.toPlainText()
    assert dlg.btn_go.text() == "Install Update"
