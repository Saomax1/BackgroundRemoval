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
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=gpu.factory, max_resident=1)

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


# --- keeping models loaded and warm (first run of a fresh session is slow) -------------------

import threading
import time


class _SlowFirstRun:
    """Records every predict; the first one per instance is 'slow' and must run alone."""

    active = 0
    overlap = False
    created: list = []

    def __init__(self, spec, path, device):
        self.name, self.runs, self.warmed = spec.id, 0, 0
        _SlowFirstRun.created.append(spec.id)

    def predict(self, rgb):
        _SlowFirstRun.active += 1
        _SlowFirstRun.overlap |= _SlowFirstRun.active > 1
        time.sleep(0.2 if self.runs == 0 else 0.01)
        self.runs += 1
        _SlowFirstRun.active -= 1
        return disc_alpha(64, 64)

    def warmup(self, cancel=None):
        self.warmed += 1
        self.predict(np.zeros((8, 8, 3), np.uint8))


class _LongWarmup(_SlowFirstRun):
    """A warm-up that takes ``seconds`` unless cancelled (like ORT's terminate flag)."""

    seconds = 2.0
    started = None
    cancelled = 0

    def warmup(self, cancel=None):
        from cutout.core.segmenter import WarmupCancelled

        self.warmed += 1
        _LongWarmup.started.set()
        end = time.perf_counter() + self.seconds
        while time.perf_counter() < end:
            if cancel is not None and cancel.is_set():
                _LongWarmup.cancelled += 1
                raise WarmupCancelled(self.name)
            time.sleep(0.01)
        self.runs += 1  # its first run is done: later predicts are fast


@pytest.fixture
def slow():
    _SlowFirstRun.active, _SlowFirstRun.overlap, _SlowFirstRun.created = 0, False, []
    return _SlowFirstRun


def test_both_models_stay_loaded_when_they_fit(tmp_path, photo, slow):
    _install(tmp_path, "birefnet-hr-matting", "toonout")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=slow, max_resident=2)
    for subject in (Subject.PERSON, Subject.ANIME, Subject.PERSON, Subject.ANIME):
        engine.remove_background(photo, subject)
    assert slow.created == ["birefnet-hr-matting", "toonout"]  # no reload on switching back
    assert sorted(engine.loaded_models) == ["birefnet-hr-matting", "toonout"]


def test_least_recently_used_model_is_unloaded(tmp_path, photo, slow):
    _install(tmp_path, "birefnet-hr-matting", "toonout", "birefnet-general")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=slow, max_resident=2)
    engine.remove_background(photo, Subject.PERSON)
    engine.remove_background(photo, Subject.ANIME)
    engine.remove_background(photo, Subject.PERSON)  # person is now the most recent
    engine.remove_background(photo, Subject.PERSON, model_id="birefnet-general")
    assert sorted(engine.loaded_models) == ["birefnet-general", "birefnet-hr-matting"]


def test_warmup_all_warms_both_once_and_first_cutout_is_warm(tmp_path, photo, slow):
    _install(tmp_path, "birefnet-hr-matting", "toonout")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=slow, max_resident=2)
    engine.warmup_all(Subject.ANIME)
    engine.warmup_all(Subject.ANIME)  # already warm: nothing runs again
    assert slow.created == ["toonout", "birefnet-hr-matting"]  # current subject first
    segs = {s.name: s for s in engine._segmenters.values()}
    assert all(s.warmed == 1 for s in segs.values())
    result = engine.remove_background(photo, Subject.PERSON)
    assert result.timings["model_load"] < 0.1 and result.timings["inference"] < 0.15


def test_warmup_all_only_warms_the_current_subject_when_one_fits(tmp_path, slow):
    _install(tmp_path, "birefnet-hr-matting", "toonout")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=slow, max_resident=1)
    engine.warmup_all(Subject.PERSON)
    assert slow.created == ["birefnet-hr-matting"]


def test_warmup_skips_subjects_without_a_model(tmp_path, slow):
    _install(tmp_path, "toonout")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=slow, max_resident=2)
    engine.warmup_all(Subject.PERSON)
    assert slow.created == ["toonout"]


def test_a_cutout_during_warmup_waits_instead_of_running_alongside(tmp_path, photo, slow):
    _install(tmp_path, "birefnet-hr-matting")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=slow, max_resident=2)
    warm = threading.Thread(target=engine.warmup, args=(Subject.PERSON,))
    warm.start()
    time.sleep(0.05)  # the warm-up holds the GPU now
    result = engine.remove_background(photo, Subject.PERSON)
    warm.join()
    assert not slow.overlap
    assert result.timings["wait"] > 0.05 and result.timings["inference"] < 0.15  # ran warm


def test_default_budget(monkeypatch):
    from cutout.core import engine as engine_mod

    monkeypatch.delenv("CUTOUT_MAX_LOADED_MODELS", raising=False)
    monkeypatch.setattr(engine_mod.debug, "gpu_usage", lambda: [{"total_mib": 8192}])
    assert engine_mod.default_max_resident("auto") == 2  # RTX 3070
    monkeypatch.setattr(engine_mod.debug, "gpu_usage", lambda: [{"total_mib": 6144}])
    assert engine_mod.default_max_resident("auto") == 1
    monkeypatch.setattr(engine_mod.debug, "gpu_usage", lambda: [])
    assert engine_mod.default_max_resident("auto") == 1  # unknown card: be careful
    assert engine_mod.default_max_resident("cpu") == 2
    monkeypatch.setenv("CUTOUT_MAX_LOADED_MODELS", "1")
    assert engine_mod.default_max_resident("cpu") == 1


@pytest.fixture
def long_warmup(slow):
    _LongWarmup.started, _LongWarmup.cancelled = threading.Event(), 0
    return _LongWarmup


def test_cutout_for_another_model_interrupts_the_warmup_which_then_resumes(tmp_path, photo, long_warmup):
    """v0.1.24 on a 3070: the first Anime cutout waited 23 s behind the Person warm-up."""
    _install(tmp_path, "birefnet-hr-matting", "toonout")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=long_warmup, max_resident=2)
    engine.remove_background(photo, Subject.ANIME)  # anime loaded (as if warmed first)
    warm = threading.Thread(target=engine.warmup, args=(Subject.PERSON,))
    warm.start()
    assert long_warmup.started.wait(5)
    result = engine.remove_background(photo, Subject.ANIME)
    assert result.timings["wait"] < 0.5  # not the whole 2 s warm-up
    assert long_warmup.cancelled == 1
    warm.join(10)
    assert not warm.is_alive()
    person = next(s for s in engine._segmenters.values() if s.name == "birefnet-hr-matting")
    assert person.warmed == 2 and person.runs == 1  # resumed and finished after the cutout
    assert "birefnet-hr-matting" in engine._warm


def test_cutout_for_the_model_being_warmed_waits_for_it(tmp_path, photo, long_warmup):
    _install(tmp_path, "birefnet-hr-matting")
    long_warmup.seconds = 0.5
    try:
        engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=long_warmup, max_resident=2)
        warm = threading.Thread(target=engine.warmup, args=(Subject.PERSON,))
        warm.start()
        assert long_warmup.started.wait(5)
        result = engine.remove_background(photo, Subject.PERSON)
        warm.join()
        assert long_warmup.cancelled == 0
        assert result.timings["inference"] < 0.15  # ran warm
    finally:
        long_warmup.seconds = 2.0


def test_onnx_warmup_can_be_cancelled(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import dataclasses

    from onnx import TensorProto, helper

    from cutout.core.models import REGISTRY
    from cutout.core.segmenter import OnnxSegmenter, WarmupCancelled

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 32, 32])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    node = helper.make_node("ReduceMean", ["input", "axes"], ["out"], keepdims=1)
    model = helper.make_model(helper.make_graph([node], "g", [x], [y], initializer=[axes]),
                              opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, tmp_path / "m.onnx")
    spec = dataclasses.replace(REGISTRY["isnet-anime"], input_size=32)
    seg = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    seg.warmup(cancel=threading.Event())  # not cancelled: runs
    stop = threading.Event()
    stop.set()
    with pytest.raises(WarmupCancelled):
        seg.warmup(cancel=stop)
    seg.cancel_warmup()  # nothing running: harmless
    assert seg.predict(np.zeros((32, 32, 3), np.uint8)).shape == (32, 32)


def test_stop_warmups_cancels_a_running_warmup_and_skips_the_rest(tmp_path, long_warmup):
    """Closing the app mid warm-up used to wait for it (and the update restart with it)."""
    _install(tmp_path, "birefnet-hr-matting", "toonout")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=long_warmup, max_resident=2)
    stages = []
    warm = threading.Thread(target=engine.warmup_all, args=(Subject.PERSON, lambda s, m: stages.append((s, m))))
    warm.start()
    assert long_warmup.started.wait(5)
    t = time.perf_counter()
    engine.stop_warmups()
    warm.join(5)
    assert not warm.is_alive() and time.perf_counter() - t < 1.0
    assert long_warmup.created == ["birefnet-hr-matting"]  # the anime model was never loaded
    assert stages[-1] == ("done", "")
    assert "birefnet-hr-matting" not in engine._warm
