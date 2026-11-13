"""Enhance in the editor window (Qt offscreen, fake upscaler, no real models)."""

import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtCore")
from PySide6.QtCore import QSettings  # noqa: E402

from conftest import FakeSegmenter  # noqa: E402
from cutout.core.engine import CutoutEngine  # noqa: E402
from cutout.core.models import ENHANCE_REGISTRY, REGISTRY, Subject, enhance_model_path  # noqa: E402
from cutout.ui.enhance_ui import EnhanceDialog  # noqa: E402
from cutout.ui.main_window import MainWindow  # noqa: E402
from test_enhance import BlurUpscaler, WhiteFace  # noqa: E402
from test_ui import _open, _photo, qapp, wait_until  # noqa: E402,F401


def _window(tmp_path, enhance=True):
    models = tmp_path / "models"
    models.mkdir()
    for mid in ("birefnet-general", "isnet-anime"):
        (models / REGISTRY[mid].filename).write_bytes(b"x")
    if enhance:
        for mid in ("realesrgan-x4plus", "gfpgan-1.4"):
            enhance_model_path(ENHANCE_REGISTRY[mid], models).write_bytes(b"x")
    engine = CutoutEngine(models_dir=models, segmenter_factory=FakeSegmenter,
                          upscaler_factory=BlurUpscaler, face_restorer_factory=WhiteFace)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    w = MainWindow(engine=engine, settings=settings, ask_user=False)
    w.resize(1000, 700)
    w.show()
    return w


@pytest.fixture
def window(qapp, tmp_path):
    w = _window(tmp_path)
    yield w
    wait_until(lambda: not w.is_busy)
    w.close()


def _enhance(window, scale, faces=False):
    done = []
    window.enhanced.connect(lambda: done.append(1))
    window.start_enhance(scale, faces)
    wait_until(lambda: done and not window.is_busy)


def test_enhance_before_the_cutout(window, tmp_path):
    src = _photo(tmp_path, 61, 43)
    window.open_path(src)
    wait_until(lambda: window.doc.has_image and not window.is_busy)
    assert window.act_enhance.isEnabled() and window.confirm_bar.enhance.isVisibleTo(window.confirm_bar)
    _enhance(window, 2)
    img = window.doc.image
    assert (img.height, img.width) == (122, 86)
    assert img.source_path == src  # Save still suggests <name>_cutout.png next to the original
    assert not window.doc.has_alpha and window.confirm_bar.isVisible()  # waits for Remove Background
    assert "enhanced 2x" in window.lbl_file.text()
    assert window.act_revert_enhance.isEnabled() and window.act_save_enhanced.isEnabled()

    window.confirm_bar.button.click()
    wait_until(lambda: window.doc.has_alpha and not window.is_busy)
    assert window.doc.alpha.shape == (122, 86)  # the cutout is made at the enhanced size
    out = tmp_path / "cut.png"
    window.save_to(out)
    wait_until(lambda: not window.is_busy)
    with Image.open(out) as im:
        assert im.size == (86, 122) and im.mode == "RGBA"


def test_enhance_starts_from_the_original_each_time(window, tmp_path):
    window.open_path(_photo(tmp_path, 40, 30))
    wait_until(lambda: window.doc.has_image and not window.is_busy)
    original = window.doc.image
    _enhance(window, 2)
    _enhance(window, 4)
    assert window.doc.image.rgb.shape[:2] == (160, 120)  # 4x the original, not 4x of the 2x
    window.revert_enhance()
    assert window.doc.image is original
    assert not window.act_revert_enhance.isEnabled() and not window.act_save_enhanced.isEnabled()


def test_enhance_after_a_cutout_cuts_out_again(window, tmp_path):
    _open(window, _photo(tmp_path, 50, 40))
    assert window.doc.alpha.shape == (50, 40)
    window.doc.mark_saved()
    done = []
    window.cutout_ready.connect(lambda: done.append(1))
    _enhance(window, 2)
    wait_until(lambda: done and window.doc.has_alpha and not window.is_busy)
    assert window.doc.alpha.shape == (100, 80)
    window.revert_enhance()
    wait_until(lambda: len(done) == 2 and window.doc.has_alpha and not window.is_busy)
    assert window.doc.alpha.shape == (50, 40)


def test_save_enhanced_picture(window, tmp_path):
    window.open_path(_photo(tmp_path, 30, 20))
    wait_until(lambda: window.doc.has_image and not window.is_busy)
    _enhance(window, 4)
    out = tmp_path / "big"
    window.save_enhanced_to(out)
    wait_until(lambda: not window.is_busy)
    with Image.open(tmp_path / "big.png") as im:
        assert im.size == (80, 120) and im.mode == "RGB"


def test_enhance_hidden_without_models(qapp, tmp_path):
    w = _window(tmp_path, enhance=False)
    try:
        w.open_path(_photo(tmp_path))
        wait_until(lambda: w.doc.has_image and not w.is_busy)
        assert not w.act_enhance.isEnabled()
        assert not w.confirm_bar.enhance.isVisibleTo(w.confirm_bar)
    finally:
        w.close()


def test_dialog_remembers_choice_and_disables_too_big(qapp, tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    settings.setValue("enhance/scale", 4)
    dlg = EnhanceDialog(None, 2000, 1500, settings, is_photo=True, faces_available=True)
    assert not dlg.buttons[4].isEnabled()  # 8000 x 6000 is over the limit
    assert dlg.scale == 2 and dlg.wants_faces
    dlg.buttons[1].setChecked(True)
    dlg.fix_faces.setChecked(False)
    dlg.accept()
    assert int(settings.value("enhance/scale")) == 1
    assert settings.value("enhance/fix_faces") in (False, "false")
    anime = EnhanceDialog(None, 100, 100, settings, is_photo=False, faces_available=True)
    assert not anime.wants_faces and not anime.fix_faces.isEnabled()
    assert anime.scale == 1
