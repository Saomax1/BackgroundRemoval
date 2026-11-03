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
    wait_until(lambda: window.doc.has_alpha and not window.is_busy)


def test_empty_state_then_open_runs_auto_removal(window, tmp_path):
    assert window.stack.currentWidget() is window.empty
    assert not window.act_save.isEnabled()
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
    act = window.register_tool(tool)
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
