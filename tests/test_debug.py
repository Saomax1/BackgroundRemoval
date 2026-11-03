"""Debug mode, log files and the copy-paste debug report."""

import logging

import pytest

from cutout.core import debug


@pytest.fixture
def logs(tmp_path):
    debug._reset_for_tests()
    folder = tmp_path / "logs"
    yield folder
    debug._reset_for_tests()


def test_setup_writes_log_file_and_debug_switch(logs):
    debug.setup(debug=False, console=False, directory=logs)
    assert not debug.enabled() and debug.ort_log_severity() == 3
    logging.getLogger("cutout.test").debug("hidden detail")
    logging.getLogger("cutout.test").info("visible line")
    debug.set_enabled(True)
    assert debug.ort_log_severity() == 1
    logging.getLogger("cutout.test").debug("debug detail")
    text = (logs / debug.LOG_NAME).read_text(encoding="utf-8")
    assert "visible line" in text and "debug detail" in text
    assert "hidden detail" not in text
    assert "debug mode ON" in text


def test_unclean_exit_is_detected_next_start(logs):
    debug.setup(console=False, directory=logs)
    assert not debug.previous_run_crashed()
    debug._reset_for_tests()  # simulate the process dying: no mark_clean_exit()
    debug.setup(console=False, directory=logs)
    assert debug.previous_run_crashed()
    assert "did not close normally" in debug.build_report()
    debug.mark_clean_exit()
    debug._reset_for_tests()
    debug.setup(console=False, directory=logs)
    assert not debug.previous_run_crashed()


def test_report_has_sections_and_hides_home_folder(logs, tmp_path, monkeypatch):
    debug.setup(console=False, directory=logs)
    monkeypatch.setattr(debug.Path, "home", classmethod(lambda cls: tmp_path))
    logging.getLogger("cutout.test").error("boom at %s", tmp_path / "photo.png")
    report = debug.build_report(extra="Background removal failed", models_dir=tmp_path / "models")
    for section in ("What happened", "Memory", "NVIDIA GPU", "ONNX Runtime / CUDA", "Models", "Log"):
        assert f"===== {section}" in report
    assert "boom at %USERPROFILE%" in report
    assert str(tmp_path) not in report
    saved = debug.save_report(extra="x")
    assert saved.parent == logs and saved.read_text(encoding="utf-8").startswith("CUTOUT DEBUG REPORT")


def test_memory_snapshot_never_raises():
    line = debug.memory_snapshot()
    assert "RAM" in line or "memory" in line


def test_engine_logs_image_size_and_model(logs, tmp_path, rng):
    from conftest import FakeSegmenter
    from cutout.core.engine import CutoutEngine
    from cutout.core.imageio import LoadedImage
    from cutout.core.models import REGISTRY, Subject

    debug.setup(debug=True, console=False, directory=logs)
    models = tmp_path / "models"
    models.mkdir()
    (models / REGISTRY["birefnet-general"].filename).write_bytes(b"x")
    engine = CutoutEngine(models_dir=models, segmenter_factory=FakeSegmenter)
    engine.remove_background(LoadedImage(rng.integers(0, 256, (120, 90, 3), dtype="uint8")), Subject.PERSON)
    text = (logs / debug.LOG_NAME).read_text(encoding="utf-8")
    assert "90x120 px, subject person" in text
    assert "model fake" in text
    assert "memory after cutout" in text
