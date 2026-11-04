"""Headless tests for the brushes and click-to-select inside the editor window."""

import os
import time

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import QEvent, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import FakeSegmenter  # noqa: E402
from test_click_select import FakeSam  # noqa: E402
from cutout.core.engine import CutoutEngine  # noqa: E402
from cutout.core.models import PROMPT_REGISTRY, REGISTRY, Subject  # noqa: E402
from cutout.ui.main_window import MainWindow  # noqa: E402

NO = Qt.KeyboardModifier.NoModifier
LEFT, RIGHT = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton


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


def make_window(tmp_path, with_sam=True):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    for mid in ("birefnet-general", "isnet-anime"):
        (models / REGISTRY[mid].filename).write_bytes(b"x")
    if with_sam:
        spec = PROMPT_REGISTRY["sam2.1-hiera-base-plus"]
        (models / spec.encoder_filename).write_bytes(b"x")
        (models / spec.decoder_filename).write_bytes(b"x")
    engine = CutoutEngine(models_dir=models, segmenter_factory=FakeSegmenter, prompt_segmenter_factory=FakeSam)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(engine=engine, settings=settings, ask_user=False)
    w.resize(1000, 700)
    w.show()
    return w


@pytest.fixture
def window(qapp, tmp_path):
    w = make_window(tmp_path)
    yield w
    wait_until(lambda: not w.is_busy)
    w.close()


def open_photo(window, tmp_path, h=600, w=800):
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.full((h, w, 3), 30, np.uint8)
    rgb[np.hypot(xx - w / 2, yy - h / 2) < 150] = 220
    p = tmp_path / "photo.png"
    Image.fromarray(rgb).save(p)
    window.open_path(p)
    wait_until(lambda: window.doc.has_image and not window.is_busy)
    window.confirm_bar.button.click()  # "Remove Background"
    wait_until(lambda: window.doc.has_alpha and not window.is_busy)
    window.canvas.actual_pixels()
    window.canvas.centerOn(w / 2, h / 2)
    QApplication.processEvents()


def mouse(window, kind, x, y, button=LEFT, buttons=None, mods=NO):
    # Exact float view position of the image point, minus the half-pixel the canvas adds back.
    view = window.canvas.viewportTransform().map(QPointF(x, y)) - QPointF(0.5, 0.5)
    types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}
    if buttons is None:
        buttons = Qt.MouseButton.NoButton if kind == "release" else button
    ev = QMouseEvent(types[kind], view, view, button if kind != "move" else Qt.MouseButton.NoButton,
                     buttons, mods)
    getattr(window.canvas, {"press": "mousePressEvent", "move": "mouseMoveEvent",
                            "release": "mouseReleaseEvent"}[kind])(ev)


def key(window, k):
    window.canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, k, NO))


def stroke(window, points, mods=NO):
    (x0, y0), rest = points[0], points[1:]
    mouse(window, "press", x0, y0, mods=mods)
    for x, y in rest:
        mouse(window, "move", x, y, buttons=LEFT, mods=mods)
    mouse(window, "release", *points[-1], mods=mods)


def test_tools_registered_and_disabled_until_cutout(qapp, tmp_path):
    w = make_window(tmp_path)
    try:
        for tid in ("brush-keep", "brush-erase", "brush-refine", "select"):
            assert tid in w._tool_actions and not w._tool_actions[tid].isEnabled()
        open_photo(w, tmp_path)
        assert all(w._tool_actions[t].isEnabled() for t in ("brush-keep", "select"))
        w.activate_tool("brush-keep")
        assert w.options_bar.isVisible()
        w.activate_tool("pan")
        assert not w.options_bar.isVisible()
    finally:
        wait_until(lambda: not w.is_busy)
        w.close()


def test_erase_and_keep_strokes_are_single_undo_steps(window, tmp_path):
    open_photo(window, tmp_path)
    a = window.doc.alpha
    assert a[300, 400] == pytest.approx(1.0) and a[20, 20] == pytest.approx(0.0)
    window.brush_settings.size = 40
    window.activate_tool("brush-erase")
    steps = window.doc.undo_stack.count()
    stroke(window, [(380, 300), (400, 300), (420, 300)])
    assert window.doc.alpha[300, 400] == pytest.approx(0.0)
    assert window.canvas.preview_buffer()[300, 400, 3] == 0
    assert window.doc.undo_stack.count() == steps + 1
    assert window.doc.alpha[300, 400 + 60] == pytest.approx(1.0)  # outside the 20 px radius

    window.activate_tool("brush-keep")
    stroke(window, [(20, 20), (40, 20)])
    assert window.doc.alpha[20, 30] == pytest.approx(1.0)

    window.act_undo.trigger()
    assert window.doc.alpha[20, 30] == pytest.approx(0.0)
    window.act_undo.trigger()
    assert window.doc.alpha[300, 400] == pytest.approx(1.0)
    assert window.canvas.preview_buffer()[300, 400, 3] == 255
    window.act_redo.trigger()
    assert window.doc.alpha[300, 400] == pytest.approx(0.0)


def test_alt_swaps_keep_and_erase(window, tmp_path):
    open_photo(window, tmp_path)
    window.activate_tool("brush-keep")
    stroke(window, [(400, 300), (402, 300)], mods=Qt.KeyboardModifier.AltModifier)
    assert window.doc.alpha[300, 401] == pytest.approx(0.0)


def test_brush_size_is_in_image_pixels_and_adjustable(window, tmp_path):
    open_photo(window, tmp_path)
    window.activate_tool("brush-erase")
    tool = window.active_tool
    window.brush_settings.size = 40
    assert tool.image_radius() == pytest.approx(20.0)
    window.act_zoom_in.trigger()
    assert tool.image_radius() == pytest.approx(20.0)  # zoom does not change what it paints
    key(window, Qt.Key.Key_BracketRight)
    assert window.brush_settings.size > 40
    window.brush_settings.size = 3
    key(window, Qt.Key.Key_BracketLeft)
    key(window, Qt.Key.Key_BracketLeft)
    assert window.brush_settings.size == 1
    key(window, Qt.Key.Key_BracketLeft)
    assert window.brush_settings.size == 1  # never below one pixel
    opts = tool.options_widget()
    opts.size_box.setValue(250)
    assert window.brush_settings.size == 250
    opts.one_px.click()
    assert window.brush_settings.size == 1 and opts.size_box.value() == 1
    opts.size_slider.setValue(opts.size_slider.maximum())
    assert window.brush_settings.size == 1000


@pytest.mark.parametrize("tool_id,start,target", [("brush-erase", 1.0, 0.0), ("brush-keep", 0.0, 1.0)])
def test_one_pixel_brush_changes_exactly_one_pixel(window, tmp_path, tool_id, start, target):
    open_photo(window, tmp_path)
    window.brush_settings.size = 1
    window.activate_tool(tool_id)
    x, y = (400, 300) if start == 1.0 else (50, 50)
    assert window.doc.alpha[y, x] == pytest.approx(start)
    before = window.doc.alpha.copy()
    stroke(window, [(x + 0.3, y + 0.8)])  # anywhere inside the pixel
    changed = np.argwhere(window.doc.alpha != before)
    assert changed.tolist() == [[y, x]]
    assert window.doc.alpha[y, x] == target  # fully, not feathered
    window.act_undo.trigger()
    assert np.array_equal(window.doc.alpha, before)


def test_small_brushes_paint_whole_pixels(window, tmp_path):
    open_photo(window, tmp_path)
    window.activate_tool("brush-erase")
    for size, count in ((2, 4), (3, 9)):
        window.brush_settings.size = size
        before = window.doc.alpha.copy()
        stroke(window, [(400.2, 300.2)])
        diff = window.doc.alpha != before
        assert diff.sum() == count
        assert np.all(window.doc.alpha[diff] == 0.0)
        window.act_undo.trigger()


def test_one_pixel_stroke_draws_a_connected_line(window, tmp_path):
    open_photo(window, tmp_path)
    window.brush_settings.size = 1
    window.activate_tool("brush-erase")
    before = window.doc.alpha.copy()
    stroke(window, [(380.5, 300.5), (390.5, 300.5), (400.5, 300.5)])
    row = window.doc.alpha[300, 380:401]
    assert np.all(row == 0.0)
    assert np.array_equal(window.doc.alpha[299], before[299]) and np.array_equal(window.doc.alpha[301], before[301])


def test_zoom_reaches_pixel_level(window, tmp_path):
    open_photo(window, tmp_path)
    for _ in range(40):
        window.act_zoom_in.trigger()
    assert window.canvas.zoom >= 32
    window.canvas.grab()  # draws the pixel grid without errors


def test_escape_cancels_stroke(window, tmp_path):
    open_photo(window, tmp_path)
    window.activate_tool("brush-erase")
    steps = window.doc.undo_stack.count()
    mouse(window, "press", 400, 300)
    mouse(window, "move", 410, 300, buttons=LEFT)
    assert window.doc.alpha[300, 405] < 1.0
    key(window, Qt.Key.Key_Escape)
    mouse(window, "release", 410, 300)
    assert window.doc.alpha[300, 405] == pytest.approx(1.0)
    assert window.doc.undo_stack.count() == steps


def test_brush_follows_redo_auto(window, tmp_path):
    open_photo(window, tmp_path)
    window.set_subject(Subject.ANIME)  # swaps in a new matte array
    wait_until(lambda: not window.is_busy)
    window.activate_tool("brush-erase")
    stroke(window, [(400, 300), (402, 300)])
    assert window.doc.alpha[300, 401] == pytest.approx(0.0)


def test_click_select_keep_and_remove(window, tmp_path):
    open_photo(window, tmp_path)
    window.activate_tool("select")
    tool = window.active_tool
    wait_until(lambda: not window.is_busy and tool.selector is not None and tool.selector.ready)

    # Remove: click the subject's middle, then Delete.
    mouse(window, "press", 400, 300)
    wait_until(lambda: tool.selection is not None and not tool.busy)
    assert not tool.selection.empty and tool._overlay is not None
    steps = window.doc.undo_stack.count()
    key(window, Qt.Key.Key_Delete)
    assert window.doc.alpha[300, 400] == pytest.approx(0.0, abs=1e-3)
    assert window.doc.undo_stack.count() == steps + 1
    assert tool.selection is None and tool._overlay is None

    # Keep: click the empty corner, then Enter.
    mouse(window, "press", 60, 60)
    wait_until(lambda: tool.selection is not None and not tool.busy)
    key(window, Qt.Key.Key_Return)
    assert window.doc.alpha[60, 60] == pytest.approx(1.0, abs=1e-3)

    window.act_undo.trigger()
    assert window.doc.alpha[60, 60] == pytest.approx(0.0, abs=1e-3)
    window.act_undo.trigger()
    assert window.doc.alpha[300, 400] == pytest.approx(1.0)


def test_click_select_escape_clears(window, tmp_path):
    open_photo(window, tmp_path)
    window.activate_tool("select")
    tool = window.active_tool
    wait_until(lambda: not window.is_busy and tool.selector is not None and tool.selector.ready)
    mouse(window, "press", 400, 300, button=LEFT)
    wait_until(lambda: tool.selection is not None and not tool.busy)
    before = window.doc.alpha.copy()
    key(window, Qt.Key.Key_Escape)
    assert tool.selection is None and not tool._markers
    assert np.array_equal(before, window.doc.alpha)


def test_missing_select_model_reports_error(qapp, tmp_path):
    w = make_window(tmp_path, with_sam=False)
    try:
        errors = []
        w.task_failed.connect(errors.append)
        open_photo(w, tmp_path)
        w.activate_tool("select")
        assert errors and "Click-to-select model not installed" in errors[0]
        # Brushes still work.
        w.activate_tool("brush-erase")
        stroke(w, [(400, 300), (402, 300)])
        assert w.doc.alpha[300, 401] == pytest.approx(0.0)
    finally:
        wait_until(lambda: not w.is_busy)
        w.close()
