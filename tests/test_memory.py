"""Model switching and out-of-memory recovery.

Regression for: run Person, switch to Anime on the same image -> ONNX Runtime
"BFCArena::AllocateRawInternal Failed to allocate memory" because the person
model was still on the GPU. ``FakeGpu`` stands in for a card that only fits one
model: a segmenter holds a slot until it is garbage-collected (as an ORT
session holds its VRAM), and running with two models resident fails with the
same message ORT gives.
"""

import gc
import weakref

import numpy as np
import pytest

from conftest import disc_alpha
from cutout.core.engine import CPU_FALLBACK_NOTICE, CutoutEngine
from cutout.core.models import REGISTRY, Subject
from cutout.core.runtime import GpuOutOfMemory, is_out_of_memory

# The two errors from the user's screenshots, verbatim apart from line breaks.
ORT_OOM_SPLIT = (
    "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Split node. "
    "Name:'/inner/bb/layers.0/blocks.1/Slice_3/GatherSliceToSplitFusion' Status Message: "
    "D:\\a\\_work\\1\\s\\onnxruntime\\core\\framework\\bfc_arena.cc:360 "
    "onnxruntime::BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer of size 52310016"
)
ORT_OOM_RESIZE = (
    "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running Resize node. "
    "Name:'/inner/decoder/Resize_8' Status Message: bfc_arena.cc:360 "
    "onnxruntime::BFCArena::AllocateRawInternal Failed to allocate memory for requested buffer of size 12582912"
)


class FakeGpu:
    def __init__(self, capacity: int = 1, always_oom: bool = False):
        self.capacity = capacity
        self.always_oom = always_oom
        self.live: list[weakref.ref] = []
        self.created: list[tuple[str, str]] = []  # (model id, device)

    def resident(self) -> int:
        self.live = [r for r in self.live if r() is not None]
        return len(self.live)

    def factory(self, spec, path, device):
        seg = _GpuSegmenter(self, spec, device)
        self.created.append((spec.id, device))
        if device != "cpu":
            self.live.append(weakref.ref(seg))
        return seg


class _GpuSegmenter:
    def __init__(self, gpu: FakeGpu, spec, device):
        self.gpu, self.name, self.device = gpu, spec.id, device

    def predict(self, rgb):
        if self.device != "cpu" and (self.gpu.always_oom or self.gpu.resident() > self.gpu.capacity):
            raise RuntimeError(ORT_OOM_SPLIT)
        return disc_alpha(64, 64)


def _install(models_dir, *ids):
    for model_id in ids:
        (models_dir / REGISTRY[model_id].filename).write_bytes(b"x")


@pytest.fixture
def image(rng):
    return rng.integers(0, 256, (300, 200, 3), dtype=np.uint8)


@pytest.fixture
def photo(tmp_path, image):
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.fromarray(image).save(path)
    return path


def test_switching_subject_unloads_previous_model(tmp_path, photo):
    _install(tmp_path, "birefnet-hr-matting", "toonout")
    gpu = FakeGpu(capacity=1)
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=gpu.factory)

    person = engine.remove_background(photo, Subject.PERSON)
    assert person.model_id == "birefnet-hr-matting" and person.notice is None
    anime = engine.remove_background(photo, Subject.ANIME)  # used to fail here
    assert anime.model_id == "toonout" and anime.notice is None
    assert anime.alpha.shape == (300, 200)  # full resolution kept

    assert engine.loaded_models == ["toonout"]
    assert gpu.resident() == 1
    # No OOM retry was needed: the switch itself freed the person model.
    assert gpu.created == [("birefnet-hr-matting", "auto"), ("toonout", "auto")]


def test_same_model_stays_loaded(tmp_path, photo):
    _install(tmp_path, "toonout")
    gpu = FakeGpu()
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=gpu.factory)
    engine.remove_background(photo, Subject.ANIME)
    engine.remove_background(photo, Subject.ANIME)
    assert gpu.created == [("toonout", "auto")]


def test_oom_unloads_everything_and_retries_on_gpu(tmp_path, photo):
    _install(tmp_path, "toonout")
    gpu = FakeGpu(capacity=1)
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=gpu.factory)
    # Something else (e.g. the click-to-select model) is holding the card.
    hog = gpu.factory(REGISTRY["birefnet-general"], None, "auto")
    engine._prompt_segmenters["sam"] = hog
    del hog

    result = engine.remove_background(photo, Subject.ANIME)
    assert result.notice is None and result.device == "n/a"
    assert engine.loaded_models == ["toonout"]
    # The failed session was really freed before the retry (not pinned by the traceback).
    assert gpu.created.count(("toonout", "auto")) == 2


def test_persistent_gpu_oom_falls_back_to_cpu_then_tries_gpu_again(tmp_path, photo):
    _install(tmp_path, "birefnet-hr-matting")
    gpu = FakeGpu(always_oom=True)
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=gpu.factory)

    result = engine.remove_background(photo, Subject.PERSON)
    assert result.notice == CPU_FALLBACK_NOTICE
    assert result.alpha.shape == (300, 200)
    assert gpu.created[-1] == ("birefnet-hr-matting", "cpu")
    assert engine.loaded_models == []  # the CPU copy is not kept around
    gc.collect()
    assert gpu.resident() == 0

    gpu.always_oom = False
    again = engine.remove_background(photo, Subject.PERSON)
    assert again.notice is None
    assert gpu.created[-1] == ("birefnet-hr-matting", "auto")


def test_cuda_only_reports_friendly_error(tmp_path, photo):
    _install(tmp_path, "toonout")
    gpu = FakeGpu(always_oom=True)
    engine = CutoutEngine(models_dir=tmp_path, device="cuda", segmenter_factory=gpu.factory)
    with pytest.raises(GpuOutOfMemory, match="graphics card ran out of memory"):
        engine.remove_background(photo, Subject.ANIME)
    assert all(device == "cuda" for _, device in gpu.created)
    assert engine.loaded_models == []


def test_other_errors_are_not_swallowed(tmp_path, photo):
    _install(tmp_path, "toonout")

    class Broken:
        name = "broken"

        def __init__(self, *a):
            pass

        def predict(self, rgb):
            raise ValueError("bad graph")

    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=Broken)
    with pytest.raises(ValueError, match="bad graph"):
        engine.remove_background(photo, Subject.ANIME)


@pytest.mark.parametrize("msg", [ORT_OOM_SPLIT, ORT_OOM_RESIZE, "CUDA failure 2: out of memory",
                                 "CUDNN_STATUS_ALLOC_FAILED"])
def test_oom_detection(msg):
    assert is_out_of_memory(RuntimeError(msg))


def test_oom_detection_ignores_other_errors():
    assert not is_out_of_memory(RuntimeError("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : wrong rank"))
    assert is_out_of_memory(MemoryError())


def test_release_models(tmp_path, photo):
    _install(tmp_path, "toonout")
    gpu = FakeGpu()
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=gpu.factory)
    engine.remove_background(photo, Subject.ANIME)
    engine.release_models()
    assert engine.loaded_models == [] and gpu.resident() == 0
