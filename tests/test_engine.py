import numpy as np
import pytest
from PIL import Image

from cutout.core.engine import CutoutEngine
from cutout.core.models import REGISTRY, ModelNotInstalled, Subject, resolve_model
from conftest import FakeSegmenter


def _touch(models_dir, model_id):
    (models_dir / REGISTRY[model_id].filename).write_bytes(b"x")


def test_resolution_preserved_end_to_end(tmp_path, rng):
    _touch(tmp_path, "birefnet-general")
    src = tmp_path / "odd.png"
    Image.fromarray(rng.integers(0, 256, (1001, 777, 3), dtype=np.uint8)).save(src)
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=FakeSegmenter)
    result = engine.remove_background(src, Subject.PERSON)
    assert result.alpha.shape == (1001, 777)
    out = tmp_path / "odd_cutout.png"
    result.save_png(out)
    im = Image.open(out)
    assert im.size == (777, 1001) and im.mode == "RGBA"
    a = np.asarray(im)[..., 3]
    assert a[500, 388] == 255 and a[0, 0] == 0


def test_model_fallback_and_cache(tmp_path):
    _touch(tmp_path, "isnet-anime")
    assert resolve_model(Subject.ANIME, tmp_path).id == "isnet-anime"
    _touch(tmp_path, "toonout")
    assert resolve_model(Subject.ANIME, tmp_path).id == "toonout"
    with pytest.raises(ModelNotInstalled):
        resolve_model(Subject.PERSON, tmp_path)

    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=FakeSegmenter)
    assert engine.segmenter_for(Subject.ANIME) is engine.segmenter_for(Subject.ANIME)


def test_cli_end_to_end(tmp_path, rng, monkeypatch):
    from cutout import cli
    from cutout.core import engine as engine_mod

    _touch(tmp_path, "birefnet-general")
    monkeypatch.setattr(engine_mod, "OnnxSegmenter", FakeSegmenter)
    src = tmp_path / "p.jpg"
    Image.fromarray(rng.integers(0, 256, (120, 90, 3), dtype=np.uint8)).save(src)
    assert cli.main([str(src), "--models-dir", str(tmp_path)]) == 0
    assert Image.open(tmp_path / "p_cutout.png").size == (90, 120)
