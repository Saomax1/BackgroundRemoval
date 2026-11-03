"""The one entry point the UI, CLI and installer smoke test talk to.

    engine = CutoutEngine()
    result = engine.remove_background("photo.jpg", Subject.PERSON)
    result.save_png("photo_cutout.png")

Everything is synchronous and thread-safe to call from a worker thread
(ONNX Runtime sessions allow concurrent ``run``); the UI must never call it on
its main thread. Sessions are created lazily and cached, because loading a
1 GB model onto the GPU takes seconds; call ``warmup()`` at app start.

Only one automatic-removal model is kept loaded at a time: switching Person ->
Anime unloads the person model first, because the 2048 px person model and the
anime model together do not fit on a typical gaming card. If a run still runs
out of GPU memory, the engine unloads everything, retries once on the GPU, and
then (with ``device="auto"``) runs that one image on the CPU and says so in
``CutoutResult.notice``. The output resolution never changes.

CPU and GPU work in tandem: while the GPU runs the network, a CPU thread prepares
the photo-only half of the full-resolution refine and colour clean-up
(``refine.prepare_image``), and every full-resolution step runs on all cores. The
click-to-select model runs on the CPU by default (``select_device``), so it never
competes with the automatic-removal model for graphics memory.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .imageio import LoadedImage, load_image, save_png
from .models import (
    ModelSpec,
    PromptModelSpec,
    Subject,
    default_models_dir,
    model_path,
    prompt_model_paths,
    resolve_model,
    resolve_prompt_model,
)
from .prompt_segmenter import OnnxSamSegmenter, PromptSegmenter
from . import debug, parallel
from .refine import ImagePrep, RefineOptions, compose_rgba, prepare_image, refine_alpha
from .runtime import GpuOutOfMemory, is_out_of_memory
from .segmenter import OnnxSegmenter, Segmenter

log = logging.getLogger(__name__)

CPU_FALLBACK_NOTICE = (
    "Your graphics card ran out of memory, so this cutout was made on the processor instead. "
    "It is the same quality, just slower. Closing other programs that use the graphics card "
    "(games, browsers with many tabs, video editors) lets the next one run on the GPU again."
)

SegmenterFactory = Callable[[ModelSpec, Path, str], Segmenter]
PromptSegmenterFactory = Callable[[PromptModelSpec, Path, Path, str], PromptSegmenter]


@dataclass
class CutoutResult:
    image: LoadedImage
    alpha: np.ndarray  # float32 HxW in [0, 1], exact input resolution
    model_id: str
    device: str
    timings: dict[str, float] = field(default_factory=dict)
    decontaminate: bool = True
    # Set when something unusual happened that the user should hear about (e.g. CPU fallback).
    notice: str | None = None
    # Photo-only precomputation, reused when composing this image for saving.
    prep: ImagePrep | None = field(default=None, repr=False, compare=False)

    def to_rgba(self) -> np.ndarray:
        return compose_rgba(self.image, self.alpha, self.decontaminate, self.prep)

    def save_png(self, path: str | os.PathLike) -> None:
        save_png(path, self.to_rgba(), self.image.icc_profile)


class CutoutEngine:
    def __init__(
        self,
        models_dir: str | os.PathLike | None = None,
        device: str = "auto",
        refine: RefineOptions = RefineOptions(),
        segmenter_factory: SegmenterFactory | None = None,
        prompt_segmenter_factory: PromptSegmenterFactory | None = None,
        select_device: str | None = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self.device = device
        # Click-to-select runs on the CPU unless asked otherwise: its heavy encoder runs
        # once per image in the background, its per-click decoder is small, and keeping it
        # off the GPU leaves all graphics memory to the automatic-removal model.
        self.select_device = select_device or os.environ.get("CUTOUT_SELECT_DEVICE", "").strip() or "cpu"
        self.refine = refine
        self._factory: SegmenterFactory = segmenter_factory or OnnxSegmenter
        self._segmenters: dict[str, Segmenter] = {}
        self._prompt_factory: PromptSegmenterFactory = prompt_segmenter_factory or OnnxSamSegmenter
        self._prompt_segmenters: dict[str, PromptSegmenter] = {}
        self._lock = threading.Lock()

    def segmenter_for(self, subject: Subject, model_id: str | None = None) -> Segmenter:
        return self._segmenter(resolve_model(subject, self.models_dir, model_id))

    def _segmenter(self, spec: ModelSpec) -> Segmenter:
        with self._lock:
            seg = self._segmenters.get(spec.id)
            if seg is None:
                # Unload the previous model before loading the next one, so the two are
                # never on the GPU together.
                if self._segmenters:
                    self._segmenters.clear()
                    gc.collect()
                seg = self._factory(spec, model_path(spec, self.models_dir), self.device)
                self._segmenters[spec.id] = seg
        return seg

    @property
    def loaded_models(self) -> list[str]:
        """Ids of the models currently held in memory (automatic removal, then click-to-select)."""
        with self._lock:
            return list(self._segmenters) + list(self._prompt_segmenters)

    def release_models(self) -> None:
        """Unload every model and give their GPU and system memory back.

        A run already in progress on another thread keeps its own reference and frees
        its model when it finishes.
        """
        with self._lock:
            self._segmenters.clear()
            self._prompt_segmenters.clear()
        gc.collect()

    def prompt_segmenter(self, model_id: str | None = None) -> PromptSegmenter:
        """The click-to-select model (best installed SAM 2.1). Raises ModelNotInstalled if none."""
        spec = resolve_prompt_model(self.models_dir, model_id)
        with self._lock:
            seg = self._prompt_segmenters.get(spec.id)
            if seg is None:
                enc, dec = prompt_model_paths(spec, self.models_dir)
                device = "cpu" if self.device == "cpu" else self.select_device
                seg = self._prompt_factory(spec, enc, dec, device)
                self._prompt_segmenters[spec.id] = seg
        return seg

    def click_selector(self, image: "CutoutResult | LoadedImage", model_id: str | None = None, opts=None):
        """A ``ClickSelector`` for this image. Call its ``prepare()`` in a worker right away."""
        from ..editing.select import ClickSelector, SelectOptions  # editing depends on core, not vice versa

        img = image.image if isinstance(image, CutoutResult) else image
        return ClickSelector(self.prompt_segmenter(model_id), img, opts or SelectOptions(refine=self.refine))

    def warmup(self, subject: Subject = Subject.PERSON) -> None:
        seg = self.segmenter_for(subject)
        if hasattr(seg, "warmup"):
            seg.warmup()

    def remove_background(
        self,
        image: LoadedImage | str | os.PathLike,
        subject: Subject = Subject.PERSON,
        model_id: str | None = None,
    ) -> CutoutResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        if not isinstance(image, LoadedImage):
            image = load_image(image)
        timings["load"] = time.perf_counter() - t0

        spec = resolve_model(subject, self.models_dir, model_id)
        # CPU side of the tandem: prepare the photo-only half of the refine while the
        # GPU runs the network (ONNX Runtime releases the GIL during the run).
        prep_future = parallel.submit(_timed_prep, image, self.refine)
        rgb = image.as_uint8()
        notice = None
        t1 = time.perf_counter()
        oom: str | None = None
        try:
            seg = self._segmenter(spec)
            coarse = seg.predict(rgb)
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            # Recover outside this block: the live exception's traceback pins the failed
            # session (and its GPU memory) until the block ends.
            oom = str(exc)
        if oom is not None:
            seg = None
            seg, coarse, notice = self._predict_after_oom(spec, rgb, oom)
        timings["inference"] = time.perf_counter() - t1
        del rgb

        t2 = time.perf_counter()
        prep, prep_seconds = prep_future.result()
        timings["refine"] = time.perf_counter() - t2  # includes any wait for the CPU prep
        t3 = time.perf_counter()
        alpha = refine_alpha(image, coarse, self.refine, prep)
        timings["refine"] += time.perf_counter() - t3
        info = getattr(seg, "session_info", None)
        device = "gpu" if (info is not None and info.on_gpu) else ("cpu" if info is not None else "n/a")
        debug.record_timings(
            f"cutout {image.width}x{image.height}", timings, model=seg.name, device=device,
            cpu_prep_alongside_model=f"{prep_seconds:.2f}s", cpu_threads=parallel.cpu_threads(),
        )

        return CutoutResult(
            image=image,
            alpha=alpha,
            model_id=seg.name,
            device=device,
            timings=timings,
            decontaminate=self.refine.decontaminate,
            notice=notice,
            # Keep only what saving needs; the full-size guide image is not worth holding.
            prep=dataclasses.replace(prep, gray=None, guide=None),
        )

    def _predict_after_oom(
        self, spec: ModelSpec, rgb: np.ndarray, first_error: str
    ) -> tuple[Segmenter, np.ndarray, str | None]:
        """Recover from an out-of-memory error: unload everything and retry on the GPU, then
        (``device="auto"`` only) on the CPU. Raises ``GpuOutOfMemory`` with a readable message."""
        log.warning("%s ran out of memory (%s); unloading all models and retrying", spec.id, first_error)
        self.release_models()
        attempts = [self.device] + (["cpu"] if self.device == "auto" else [])
        last = first_error
        for device in attempts:
            failed: str | None = None
            try:
                if device == "cpu":
                    log.warning("running %s on the CPU for this image", spec.id)
                    # Not cached: the next image tries the GPU again.
                    seg = self._factory(spec, model_path(spec, self.models_dir), "cpu")
                else:
                    seg = self._segmenter(spec)
                coarse = seg.predict(rgb)
            except Exception as exc:
                if not is_out_of_memory(exc):
                    raise
                failed = str(exc)
            if failed is None:
                return seg, coarse, (CPU_FALLBACK_NOTICE if device == "cpu" else None)
            log.warning("%s out of memory on %s: %s", spec.id, device, failed)
            last = failed
            seg = None
            self.release_models()

        if self.device == "auto":
            msg = ("Your computer ran out of memory removing this background, on both the graphics "
                   "card and the processor. Close other programs and try again.")
        else:
            where = "graphics card" if self.device == "cuda" else "computer"
            msg = (f"Your {where} ran out of memory removing this background. "
                   "Close other programs that use it and try again.")
        log.error("giving up on %s: %s", spec.id, last)
        raise GpuOutOfMemory(msg)


def _timed_prep(image: LoadedImage, opts: RefineOptions) -> tuple[ImagePrep, float]:
    t = time.perf_counter()
    prep = prepare_image(image, opts)
    return prep, time.perf_counter() - t
