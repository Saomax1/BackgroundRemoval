import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def disc_alpha(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    r = min(h, w) * 0.35
    d = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    return np.clip((r - d) / 3.0 + 0.5, 0.0, 1.0).astype(np.float32)


class FakeSegmenter:
    """Predicts a centred soft disc at a fixed 256x256, like a real net would at its own size."""

    name = "fake"

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def predict(self, rgb):
        assert rgb.dtype == np.uint8 and rgb.shape[2] == 3
        self.calls += 1
        return disc_alpha(256, 256)


@pytest.fixture(autouse=True)
def _graph_cache_in_tmp(tmp_path_factory, monkeypatch):
    """Keep ONNX Runtime's saved graphs out of the real user folder."""
    monkeypatch.setenv("CUTOUT_GRAPH_CACHE", str(tmp_path_factory.mktemp("graph-cache")))
