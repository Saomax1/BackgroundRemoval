"""Packaging logic that can be checked without Windows, a GPU or the real models."""

import importlib.util
import sys
from pathlib import Path

import pytest

from cutout.core import runtime
from cutout.core.models import DEFAULT_MODELS

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage = _load(ROOT / "packaging" / "stage_models.py")
GB = 10**9


def test_bundled_cuda_dir_only_when_frozen(tmp_path, monkeypatch):
    (tmp_path / runtime.FROZEN_CUDA_DIR).mkdir()
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert runtime.bundled_cuda_dir() is None
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert runtime.bundled_cuda_dir() == tmp_path / runtime.FROZEN_CUDA_DIR


def test_stage_keeps_everything_under_budget():
    ok = {"birefnet-hr-matting": GB, "birefnet-general": GB, "toonout": GB, "isnet-anime": GB // 5}
    keep, dropped = stage.choose(ok, 10 * GB)
    assert set(keep) == set(ok) and dropped == []


def test_stage_drops_redundant_fallback_first():
    ok = {"birefnet-hr-matting": 9 * GB // 10, "birefnet-general": 9 * GB // 10,
          "toonout": 9 * GB // 10, "isnet-anime": GB // 6}
    keep, dropped = stage.choose(ok, int(2.6 * GB))
    # birefnet-general is outranked for both subjects it serves; the small anime fallback stays.
    assert dropped == ["birefnet-general"]
    assert set(keep) == {"birefnet-hr-matting", "toonout", "isnet-anime"}


@pytest.mark.parametrize("missing", ["birefnet-hr-matting", "toonout"])
def test_stage_never_drops_the_best_model_for_a_subject(missing):
    ok = {"birefnet-hr-matting": GB, "birefnet-general": GB, "toonout": GB, "isnet-anime": GB // 5}
    del ok[missing]
    keep, _ = stage.choose(ok, 1)  # impossible budget: only redundant models may go
    for order in DEFAULT_MODELS.values():
        best = next(m for m in order if m in ok)
        assert best in keep
