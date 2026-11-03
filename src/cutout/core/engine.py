"""The one entry point the UI, CLI and installer smoke test talk to.

    engine = CutoutEngine()
    result = engine.remove_background("photo.jpg", Subject.PERSON)
    result.save_png("photo_cutout.png")

Everything is synchronous and thread-safe to call from a worker thread
(ONNX Runtime sessions allow concurrent ``run``); the UI must never call it on
its main thread. Sessions are created lazily and cached, because loading a
1 GB model onto the GPU takes seconds; call ``warmup()`` at app start.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .imageio import LoadedImage, load_image, save_png
from .models import ModelSpec, Subject, default_models_dir, model_path, resolve_model
from .refine import RefineOptions, compose_rgba, refine_alpha
from .segmenter import OnnxSegmenter, Segmenter

SegmenterFactory = Callable[[ModelSpec, Path, str], Segmenter]


@dataclass
class CutoutResult:
    image: LoadedImage
    alpha: np.ndarray  # float32 HxW in [0, 1], exact input resolution
    model_id: str
    device: str
    timings: dict[str, float] = field(default_factory=dict)
    decontaminate: bool = True

    def to_rgba(self) -> np.ndarray:
        return compose_rgba(self.image, self.alpha, self.decontaminate)

    def save_png(self, path: str | os.PathLike) -> None:
        save_png(path, self.to_rgba(), self.image.icc_profile)


class CutoutEngine:
    def __init__(
        self,
        models_dir: str | os.PathLike | None = None,
        device: str = "auto",
        refine: RefineOptions = RefineOptions(),
        segmenter_factory: SegmenterFactory | None = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self.device = device
        self.refine = refine
        self._factory: SegmenterFactory = segmenter_factory or OnnxSegmenter
        self._segmenters: dict[str, Segmenter] = {}
        self._lock = threading.Lock()

    def segmenter_for(self, subject: Subject, model_id: str | None = None) -> Segmenter:
        spec = resolve_model(subject, self.models_dir, model_id)
        with self._lock:
            seg = self._segmenters.get(spec.id)
            if seg is None:
                seg = self._factory(spec, model_path(spec, self.models_dir), self.device)
                self._segmenters[spec.id] = seg
        return seg

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

        seg = self.segmenter_for(subject, model_id)
        t1 = time.perf_counter()
        coarse = seg.predict(image.as_uint8())
        timings["inference"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        alpha = refine_alpha(image, coarse, self.refine)
        timings["refine"] = time.perf_counter() - t2

        info = getattr(seg, "session_info", None)
        device = "gpu" if (info is not None and info.on_gpu) else ("cpu" if info is not None else "n/a")
        return CutoutResult(
            image=image,
            alpha=alpha,
            model_id=seg.name,
            device=device,
            timings=timings,
            decontaminate=self.refine.decontaminate,
        )
