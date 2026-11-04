"""Headless tests for the editor window (Qt offscreen platform, fake segmenter, no models)."""

import os
import time

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
from PySide6.QtCore import QPointF, QRect, QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import FakeSegmenter  # noqa: E402
from cutout.core.engine import CutoutEngine  # noqa: E402
from cutout.core.models import REGISTRY, Subject  # noqa: E402
from cutout.ui.canvas import Backdrop  # noqa: E402
from cutout.ui.main_window import MainWindow, default_output_path  # noqa: E402
from cutout.ui.tools import CanvasTool  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for condition")


@pytest.fixture
def window(qapp, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for mid in ("birefnet-general", "isnet-anime"):
        (models / REGISTRY[mid].filename).write_bytes(b"x")
    engine = CutoutEngine(models_dir=models, segmenter_factory=FakeSegmenter)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(engine=engine, settings=settings, ask_user=False)
    w.resize(1000, 700)
    w.show()
    yield w
    wait_until(lambda: not w.is_busy)
    w.close()


def _photo(tmp_path, h=601, w=403, name="photo.png"):
    rng = np.random.default_rng(1)
    p = tmp_path / name
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(p)
    return p


def _open(window, path):
    window.open_path(path)
    wait_until(lambda: window.doc.has_image and not window.is_busy)
    window.confirm_bar.button.click()
    wait_until(lambda: window.doc.has_alpha and not window.is_busy)


def test_open_waits_for_remove_background_button(window, tmp_path):
    window.open_path(_photo(tmp_path))
    wait_until(lambda: window.doc.has_image and not window.is_busy)
    for _ in range(20):  # give a wrongly started removal time to land
        QApplication.processEvents()
        time.sleep(0.005)
    assert not window.doc.has_alpha and not window.is_busy
    assert window.stack.currentWidget() is window.canvas  # the image is shown to check
    assert window.confirm_bar.isVisible()
    assert window.act_confirm.isEnabled()
    assert window.act_rerun.text() == "Remove Background"
    assert not window.act_save.isEnabled()

    # Switching the subject before confirming only picks the model; nothing runs.
    window.set_subject(Subject.ANIME)
    assert not window.is_busy and not window.doc.has_alpha

    subjects = []
    real = window.engine.remove_background
    window.engine.remove_background = lambda img, subj, *a, **k: (subjects.append(subj), real(img, subj, *a, **k))[1]
    window.act_confirm.trigger()  # Enter
    wait_until(lambda: window.doc.has_alpha and not window.is_busy)
    assert not window.confirm_bar.isVisible()
    assert not window.act_confirm.isEnabled()
    assert window.act_rerun.text() == "Redo Auto"
    assert subjects == [Subject.ANIME]  # the button used the subject picked beforehand


def test_empty_state_then_open_runs_auto_removal(window, tmp_path):
    assert window.stack.currentWidget() is window.empty
    assert not window.act_save.isEnabled()
    assert not window.confirm_bar.isVisible()
    _open(window, _photo(tmp_path))
    assert window.stack.currentWidget() is window.canvas
    assert window.doc.alpha.shape == (601, 403)
    assert window.act_save.isEnabled()
    assert window.doc.is_modified  # a fresh cutout is unsaved work
    buf = window.canvas.preview_buffer()
    assert buf.shape == (601, 403, 4)
    assert buf[300, 201, 3] == 255 and buf[0, 0, 3] == 0
    assert buf[0, 0, :3].max() == 0  # premultiplied: transparent pixels are black
    assert "fake" in window.lbl_model.text()


def test_save_writes_full_resolution_png(window, tmp_path):
    src = _photo(tmp_path, 777, 1001)
    _open(window, src)
    out = default_output_path(src)
    window.save_to(out)
    wait_until(lambda: not window.is_busy and out.exists())
    im = Image.open(out)
    assert im.size == (1001, 777) and im.mode == "RGBA"
    a = np.asarray(im)
    assert a[388, 500, 3] == 255 and a[0, 0, 3] == 0
    # Opaque pixels are bit-exact copies of the input.
    assert np.array_equal(a[388, 500, :3], np.asarray(Image.open(src))[388, 500])
    assert not window.doc.is_modified


def test_save_is_prepared_in_background_and_dropped_after_edits(window, tmp_path):
    from cutout.core.refine import compose_rgba

    src = _photo(tmp_path, 300, 200)
    _open(window, src)
    ready = window._prepared_save(window.doc.image, True)
    assert ready is not None
    wait_until(ready.done)
    out = tmp_path / "prepared.png"
    window.save_to(out)
    wait_until(lambda: not window.is_busy and out.exists())
    expected = compose_rgba(window.doc.image, window.doc.alpha)
    assert np.array_equal(np.asarray(Image.open(out)), expected)

    # Any edit makes the prepared PNG stale: the next save composes the edited matte.
    from PySide6.QtCore import QRect

    window.doc.push_patch(QRect(0, 0, 10, 10), np.ones((10, 10), np.float32))
    assert window._prepared_save(window.doc.image, True) is None
    out2 = tmp_path / "edited.png"
    window.save_to(out2)
    wait_until(lambda: not window.is_busy and out2.exists())
    assert np.asarray(Image.open(out2))[5, 5, 3] == 255


def test_refuses_to_overwrite_original(window, tmp_path):
    src = _photo(tmp_path)
    _open(window, src)
    before = src.read_bytes()
    errors = []
    window.task_failed.connect(errors.append)
    window.save_to(src.with_suffix(".png"))
    assert errors and src.read_bytes() == before


def test_show_original_and_backdrop(window, tmp_path):
    _open(window, _photo(tmp_path))
    window.act_original.trigger()
    assert window.canvas.preview_buffer()[0, 0, 3] == 255
    window.act_original.trigger()
    assert window.canvas.preview_buffer()[0, 0, 3] == 0
    window.set_backdrop(Backdrop.BLACK)
    assert window.canvas.backdrop is Backdrop.BLACK
    assert window.backdrop_combo.currentData() == Backdrop.BLACK
    window.canvas.grab()  # paints without errors


def test_zoom_controls(window, tmp_path):
    _open(window, _photo(tmp_path, 3000, 2000))
    fit = window.canvas.zoom
    assert fit < 1.0
    window.act_100.trigger()
    assert window.canvas.zoom == pytest.approx(1.0)
    window.act_zoom_in.trigger()
    assert window.canvas.zoom > 1.0
    window.act_fit.trigger()
    assert window.canvas.zoom == pytest.approx(fit)


def test_subject_switch_reruns_and_is_undoable(window, tmp_path):
    _open(window, _photo(tmp_path))
    first = window.doc.alpha
    window.doc.alpha[:10, :10] = 0.5  # pretend an edit happened in place
    window.set_subject(Subject.ANIME)
    wait_until(lambda: not window.is_busy and window.doc.alpha is not first)
    assert window.doc.undo_stack.canUndo()
    window.act_undo.trigger()
    assert window.doc.alpha is first


class StampTool(CanvasTool):
    """Minimal tool: a click clears a 5x5 square of the matte, one undo step."""

    id = "stamp"
    label = "Stamp"

    def __init__(self):
        super().__init__()
        self.presses = []

    def mouse_press(self, pos, event):
        self.presses.append((pos.x(), pos.y()))
        x, y = int(pos.x()), int(pos.y())
        rect = QRect(x, y, 5, 5)
        self.ctx.document.push_patch(rect, np.zeros((5, 5), np.float32), text="Stamp")
        return True


def test_tool_hook_receives_image_coordinates_and_undo_works(window, tmp_path):
    tool = StampTool()
    act = window.register_tool(tool, shortcut="S")
    _open(window, _photo(tmp_path))
    assert act.isEnabled()
    window.activate_tool("stamp")
    window.canvas.actual_pixels()
    window.canvas.centerOn(200, 300)
    QApplication.processEvents()

    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent

    view_pos = window.canvas.mapFromScene(QPointF(200.5, 300.5))
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(view_pos), QPointF(view_pos),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.canvas.mousePressEvent(ev)
    assert tool.presses
    x, y = tool.presses[0]
    assert abs(x - 200.5) <= 1 and abs(y - 300.5) <= 1
    xi, yi = int(x), int(y)
    assert window.doc.alpha[yi, xi] == 0.0
    assert window.canvas.preview_buffer()[yi, xi, 3] == 0
    window.act_undo.trigger()
    assert window.doc.alpha[yi, xi] == pytest.approx(1.0)
    assert window.canvas.preview_buffer()[yi, xi, 3] == 255


def test_missing_model_reports_friendly_error(qapp, tmp_path):
    empty = tmp_path / "nomodels"
    empty.mkdir()
    engine = CutoutEngine(models_dir=empty, segmenter_factory=FakeSegmenter)
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    w = MainWindow(engine=engine, settings=settings, ask_user=False)
    errors = []
    w.task_failed.connect(errors.append)
    w.open_path(_photo(tmp_path))
    wait_until(lambda: w.doc.has_image and not w.is_busy)
    w.confirm_bar.button.click()
    wait_until(lambda: errors and not w.is_busy)
    assert "Model not installed" in errors[0]
    assert w.doc.has_image and not w.doc.has_alpha
    w.close()


def test_unsupported_file_rejected(window, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hi")
    errors = []
    window.task_failed.connect(errors.append)
    window.open_path(p)
    assert errors and not window.doc.has_image


def test_task_finishing_after_its_window_closed_does_not_crash(qapp):
    """0.1.16 crash on close: 'RuntimeError: Signal source has been deleted' in workers.run."""
    import shiboken6

    from cutout.ui.workers import Task

    task = Task(lambda: 42)
    task.setAutoDelete(False)
    shiboken6.delete(task.signals)  # what Qt does to it when the app shuts down
    task.run()  # must not raise
    failing = Task(lambda: 1 / 0)
    failing.setAutoDelete(False)
    shiboken6.delete(failing.signals)
    failing.run()


# ------------------------------------------------------------ warm-up loading bar
class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_warmup_tracker_moves_by_time_and_learns_durations(qapp, tmp_path):
    from cutout.ui.warmup import WarmupTracker

    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    clock = _Clock()
    t = WarmupTracker(settings, clock=clock)
    t.on_stage("plan", "birefnet-hr-matting,toonout")
    assert t.active and t.fraction() == 0.0
    # Defaults: person load 5 + warm 40, anime load 5 + warm 3 = 53 s.
    t.on_stage("load", "birefnet-hr-matting")
    assert "Loading the Person model (1 of 2)" in t.text()
    clock.t += 2.5
    assert t.fraction() == pytest.approx(2.5 / 53)
    clock.t += 100  # stage overran: the bar waits just short of the stage end
    assert t.fraction() == pytest.approx(0.95 * 5 / 53)
    t.on_stage("warm", "birefnet-hr-matting")
    assert "Warming up the Person model" in t.text()
    clock.t += 30
    t.on_stage("paused", "birefnet-hr-matting")
    assert "paused" in t.text()
    t.on_stage("warm", "birefnet-hr-matting")  # resumed: the run starts over
    clock.t += 20
    t.on_stage("load", "toonout")
    assert "(2 of 2)" in t.text()
    t.on_stage("done", "")
    assert not t.active and t.fraction() == 1.0 and t.text() == "AI models ready"
    # Measured times replace the defaults next start (the paused attempt is not recorded as is).
    assert float(settings.value("warmup/birefnet-hr-matting/load")) == pytest.approx(102.5)
    assert float(settings.value("warmup/birefnet-hr-matting/warm")) == pytest.approx(20)
    assert t.expected("birefnet-hr-matting", "warm") == pytest.approx(20)


def test_warmup_tracker_counts_skipped_stages_as_done(qapp):
    from cutout.ui.warmup import WarmupTracker

    clock = _Clock()
    t = WarmupTracker(None, clock=clock)
    t.on_stage("plan", "birefnet-hr-matting,toonout")
    t.on_stage("warm", "toonout")  # person was already loaded and warm
    assert t.fraction() == pytest.approx(50 / 53)


def test_loading_bar_shows_during_startup_warmup_then_hides(window, tmp_path):
    from cutout.ui.warmup import WarmupBar

    panel = window.warmup_panel
    status = window.warmup_status
    assert isinstance(panel, WarmupBar)
    seen = []
    window.warmup_tracker.changed.connect(lambda: seen.append(panel.isVisible()))
    window.warmup_in_background()
    wait_until(lambda: window.warmup_tracker.plan and not window.warmup_tracker.active)
    QApplication.processEvents()
    assert True in seen  # was shown on the start screen while warming
    assert not panel.isVisible() and not status.isVisible()


def test_status_bar_loading_bar_only_shows_with_an_image_open(window, tmp_path):
    t = window.warmup_tracker
    t.on_stage("plan", "birefnet-general")
    t.on_stage("load", "birefnet-general")
    qapp_ = QApplication.instance()
    qapp_.processEvents()
    assert window.warmup_panel.isVisible() and not window.warmup_status.isVisible()
    _open(window, _photo(tmp_path))
    t.changed.emit()
    assert window.warmup_status.isVisible() and not window.warmup_panel.isVisible()
    t.on_stage("done", "")
    assert not window.warmup_status.isVisible()
