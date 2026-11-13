"""The one entry point the UI, CLI and installer smoke test talk to.

    engine = CutoutEngine()
    result = engine.remove_background("photo.jpg", Subject.PERSON)
    result.save_png("photo_cutout.png")

Everything is synchronous and thread-safe to call from a worker thread
(ONNX Runtime sessions allow concurrent ``run``); the UI must never call it on
its main thread. Sessions are created lazily and cached, because loading a
1 GB model onto the GPU takes seconds; call ``warmup()`` at app start.

Up to ``max_resident`` automatic-removal models stay loaded (two on cards with
about 8 GB or more, one otherwise; least recently used goes first), and
``warmup_all()`` runs them once in the background at start, because a fresh
ONNX Runtime session is several times slower on its first run. GPU runs are
serialised, so a cutout started during a warm-up waits for it and then runs
warm. Loading a model makes room first, so two are never loading together. If a run still runs
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

from .enhance import (
    MAX_OUTPUT_PIXELS,
    FaceRestorer,
    OnnxFaceRestorer,
    OnnxUpscaler,
    Upscaler,
    pick_scale,
    restore_faces,
    upscale_image,
)
from .imageio import LoadedImage, load_image, save_png
from .models import (
    EnhanceModelSpec,
    ModelNotInstalled,
    ModelSpec,
    PromptModelSpec,
    Subject,
    default_models_dir,
    enhance_model_path,
    model_path,
    prompt_model_paths,
    resolve_face_model,
    resolve_model,
    resolve_prompt_model,
    resolve_upscaler,
)
from .prompt_segmenter import OnnxSamSegmenter, PromptSegmenter
from .head import Face, FaceFinder, HeadCut, describe, head_only
from . import debug, parallel
from .refine import ImagePrep, RefineOptions, compose_rgba, prepare_image, refine_alpha
from .runtime import GpuOutOfMemory, is_out_of_memory
from .segmenter import OnnxSegmenter, Segmenter, WarmupCancelled

log = logging.getLogger(__name__)

CPU_FALLBACK_NOTICE = (
    "Your graphics card ran out of memory, so this cutout was made on the processor instead. "
    "It is the same quality, just slower. Closing other programs that use the graphics card "
    "(games, browsers with many tabs, video editors) lets the next one run on the GPU again."
)

# Total VRAM from which both automatic-removal models stay loaded. Measured on an 8 GB
# RTX 3070: person (2048) + anime (1024) sessions hold ~2.7 GB together, and a person run
# peaks well under the rest now that the deformable convs are exported in slices.
KEEP_BOTH_MIN_VRAM_MIB = 7500


def default_max_resident(device: str) -> int:
    """2 on a GPU with enough memory (or on the CPU, where RAM is plentiful), else 1."""
    env = os.environ.get("CUTOUT_MAX_LOADED_MODELS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    if device == "cpu":
        return 2
    try:
        gpus = debug.gpu_usage()
    except Exception:  # nvidia-smi missing or odd output: be conservative
        gpus = []
    if not gpus:
        return 1
    return 2 if gpus[0]["total_mib"] >= KEEP_BOTH_MIN_VRAM_MIB else 1


WarmupProgress = Callable[[str, str], None]  # (stage, model id)
SegmenterFactory = Callable[[ModelSpec, Path, str], Segmenter]
PromptSegmenterFactory = Callable[[PromptModelSpec, Path, Path, str], PromptSegmenter]
UpscalerFactory = Callable[[EnhanceModelSpec, Path, str], Upscaler]
FaceRestorerFactory = Callable[[EnhanceModelSpec, Path, str], FaceRestorer]

ENHANCE_CPU_NOTICE = (
    "Your graphics card ran out of memory, so this picture was enhanced on the processor "
    "instead. It is the same quality, just slower."
)


@dataclass
class EnhanceResult:
    image: LoadedImage  # the enhanced picture: ``scale`` times the size, same bit depth and ICC
    scale: int  # what was made (can be lower than asked if the picture would be too big)
    requested_scale: int
    upscaler_id: str
    face_model_id: str | None = None  # set when faces were looked for
    faces_fixed: int = 0
    device: str = "n/a"
    timings: dict[str, float] = field(default_factory=dict)
    notice: str | None = None


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
        max_resident: int | None = None,
        upscaler_factory: UpscalerFactory | None = None,
        face_restorer_factory: FaceRestorerFactory | None = None,
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
        # One network on the GPU at a time: a background warm-up and the user's run must not
        # both hold activation memory, and a run queued behind a warm-up then starts warm.
        self._infer_lock = threading.Lock()
        # Cutouts go before warm-ups: a cutout waiting for the GPU stops a warm-up of another
        # model (which retries once the GPU is free), and no warm-up starts while one waits.
        self._turns = threading.Condition()
        self._cutouts_waiting = 0
        self._warming: tuple[str, Segmenter, threading.Event] | None = None
        self._warm: set[str] = set()
        self._warmups_stopped = False
        self._max_resident = max_resident
        self._face_finder: FaceFinder | None = None
        self._upscaler_factory: UpscalerFactory = upscaler_factory or OnnxUpscaler
        self._face_factory: FaceRestorerFactory = face_restorer_factory or OnnxFaceRestorer

    def segmenter_for(self, subject: Subject, model_id: str | None = None) -> Segmenter:
        return self._segmenter(resolve_model(subject, self.models_dir, model_id))

    @property
    def max_resident(self) -> int:
        """How many automatic-removal models stay loaded at once (decided on first use).

        A freshly loaded ONNX Runtime session is slow on its first run (on an RTX 3070 the
        2048 px person model took ~24 s once, ~7 s after that), so unloading on every
        Person/Anime switch made each switch pay that again. With enough VRAM both models
        stay loaded and warm; on small cards only one does, as before.
        """
        if self._max_resident is None:
            self._max_resident = default_max_resident(self.device)
            log.info("keeping up to %d automatic-removal model(s) loaded", self._max_resident)
        return self._max_resident

    def _segmenter(self, spec: ModelSpec) -> Segmenter:
        with self._lock:
            seg = self._segmenters.pop(spec.id, None)
            if seg is None:
                # Make room first, so the old and new model are never both loading/running.
                while self._segmenters and len(self._segmenters) >= self.max_resident:
                    oldest = next(iter(self._segmenters))
                    del self._segmenters[oldest]
                    self._warm.discard(oldest)
                gc.collect()
                seg = self._factory(spec, model_path(spec, self.models_dir), self.device)
            self._segmenters[spec.id] = seg  # most recently used last
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
            self._warm.clear()
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

    @property
    def face_finder(self) -> FaceFinder:
        """The face detector behind Head only (tiny, CPU only, shipped with the app)."""
        with self._lock:
            if self._face_finder is None:
                self._face_finder = FaceFinder()
            return self._face_finder

    def head_only(
        self,
        image: LoadedImage,
        alpha: np.ndarray,
        subject: Subject = Subject.PERSON,
        faces: list[Face] | None = None,
    ) -> HeadCut:
        """Keep only the head(s) of a cutout: ``alpha`` with the rest made transparent.

        Runs on the CPU (tens of milliseconds), never on the GPU. ``faces`` skips the
        detector (a box the user drew). Raises ``head.NoFaceFound`` if no face is on the
        cutout. ``alpha`` is not changed.
        """
        t = time.perf_counter()
        cut = head_only(image.as_uint8(), alpha, subject, faces, self.face_finder)
        debug.record_timings(f"head only {image.width}x{image.height}",
                             {"head only": time.perf_counter() - t}, faces=len(cut.faces),
                             drawn_by_user=faces is not None, kept=describe(cut.faces))
        return cut

    def warmup(self, subject: Subject = Subject.PERSON, progress: "WarmupProgress | None" = None) -> bool:
        """Load the subject's model and run it once, so the user's first cutout is fast.

        Background work: it waits while any cutout wants the GPU, and a cutout for another
        model stops it mid-run; it then tries again. Returns False only if the model has no
        ``warmup``.
        """
        spec = resolve_model(subject, self.models_dir)
        report = progress or (lambda stage, model_id: None)
        while True:
            with self._turns:
                self._turns.wait_for(lambda: self._cutouts_waiting == 0 or self._warmups_stopped)
            with self._infer_lock:
                if self._warmups_stopped:
                    return False
                if self._cutouts_waiting:
                    continue  # a cutout arrived while we waited for the lock: it goes first
                if spec.id in self._warm and spec.id in self._segmenters:
                    return True
                if spec.id not in self._segmenters:
                    report("load", spec.id)
                seg = self._segmenter(spec)
                if not hasattr(seg, "warmup"):
                    return False
                with self._turns:
                    if self._cutouts_waiting or self._warmups_stopped:
                        continue
                    cancel = threading.Event()
                    self._warming = (spec.id, seg, cancel)
                report("warm", spec.id)
                t = time.perf_counter()
                cancelled = False
                try:
                    with debug.GpuPeak() as peak:  # debug mode only: VRAM high-water mark
                        seg.warmup(cancel=cancel)
                except WarmupCancelled:
                    cancelled = True
                finally:
                    with self._turns:
                        self._warming = None
                if self._warmups_stopped:
                    log.info("stopped warming up %s after %.1fs: the app is closing",
                             spec.id, time.perf_counter() - t)
                    return False
                if not cancelled:
                    log.info("warmed up %s in %.1fs%s", spec.id, time.perf_counter() - t, peak.summary())
                    self._warm.add(spec.id)
                    return True
                log.info("paused warming up %s after %.1fs: a cutout needs the GPU",
                         spec.id, time.perf_counter() - t)
                report("paused", spec.id)

    def stop_warmups(self) -> None:
        """Stop the background warm-up for good (the app is closing): a run in progress is
        cancelled, so closing does not wait for it."""
        with self._turns:
            self._warmups_stopped = True
            warming = self._warming
            self._turns.notify_all()
        if warming is not None:
            warming[2].set()
            if hasattr(warming[1], "cancel_warmup"):
                warming[1].cancel_warmup()

    def warmup_plan(self, first: Subject = Subject.PERSON) -> list[str]:
        """Model ids ``warmup_all(first)`` will warm, in order (installed ones only)."""
        order = [first] + [s for s in Subject if s is not first]
        plan = []
        for subject in order[: self.max_resident]:
            try:
                plan.append(resolve_model(subject, self.models_dir).id)
            except ModelNotInstalled:
                continue
        return plan

    def warmup_all(self, first: Subject = Subject.PERSON, progress: "WarmupProgress | None" = None) -> None:
        """Warm ``first``, then (if they can stay loaded together) the other subjects' models.

        ``progress(stage, model_id)`` is called from this thread with stage ``"load"``,
        ``"warm"``, ``"paused"`` (a cutout took the GPU; it resumes) and finally
        ``("done", "")``.
        """
        order = [first] + [s for s in Subject if s is not first]
        if progress is not None:
            progress("plan", ",".join(self.warmup_plan(first)))
        try:
            for subject in order[: self.max_resident]:
                if self._warmups_stopped:
                    break
                try:
                    self.warmup(subject, progress)
                except ModelNotInstalled:
                    continue
        finally:
            if progress is not None:
                progress("done", "")

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
        t1 = time.perf_counter()
        with self._turns:
            self._cutouts_waiting += 1
            warming = self._warming
        if warming is not None and warming[0] != spec.id:
            # Another model's warm-up: stop it; it resumes once this cutout is done.
            warming[2].set()
            if hasattr(warming[1], "cancel_warmup"):
                warming[1].cancel_warmup()
        # (A warm-up of this same model is left to finish: this cutout then runs warm, which
        # is no slower than running it cold now.)
        try:
            return self._remove_background_turn(image, spec, prep_future, rgb, timings, t1)
        finally:
            with self._turns:
                self._cutouts_waiting -= 1
                self._turns.notify_all()

    def _remove_background_turn(self, image, spec, prep_future, rgb, timings, t1) -> CutoutResult:
        notice = None
        oom: str | None = None
        with self._infer_lock:
            timings["wait"] = time.perf_counter() - t1
            t1 = time.perf_counter()
            try:
                seg = self._segmenter(spec)
                timings["model_load"] = time.perf_counter() - t1
                t1 = time.perf_counter()
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
            first_run = spec.id not in self._warm
            if spec.id in self._segmenters:  # the CPU fallback copy is not kept, so not warm
                self._warm.add(spec.id)
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
            first_run_of_model=first_run,
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

    # ---------------------------------------------------------------- enhance
    def can_enhance(self, subject: Subject = Subject.PERSON) -> bool:
        """True if an upscaler is installed (any one works for either subject)."""
        try:
            resolve_upscaler(subject, self.models_dir)
            return True
        except ModelNotInstalled:
            return False

    def can_fix_faces(self) -> bool:
        try:
            resolve_face_model(self.models_dir)
            return True
        except ModelNotInstalled:
            return False

    def enhance(
        self,
        image: LoadedImage | str | os.PathLike,
        subject: Subject = Subject.PERSON,
        scale: int = 2,
        fix_faces: bool = True,
        face_strength: float = 0.7,
        progress: Callable[[str, int, int], None] | None = None,
        upscaler_id: str | None = None,
    ) -> EnhanceResult:
        """Upscale ``image`` by ``scale`` (1, 2 or 4) and, for photos, repair the faces.

        Takes its turn on the GPU like a cutout (a warm-up is paused for it). The models are
        loaded for this run only and dropped afterwards, so they never stay in graphics
        memory next to the cutout models. Out of memory: everything is unloaded and it is
        retried on the GPU, then (``device="auto"``) on the CPU, with ``notice`` set.
        ``progress(stage, done, total)`` reports ``"upscale"`` tiles and ``"faces"``.
        Raises ``ModelNotInstalled`` without an upscaler and ``enhance.EnhanceTooLarge`` if
        the picture is already too big.
        """
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        if not isinstance(image, LoadedImage):
            image = load_image(image)
        timings["load"] = time.perf_counter() - t0
        subject = Subject(subject)
        up_spec = resolve_upscaler(subject, self.models_dir, upscaler_id)
        made = pick_scale(image.width, image.height, scale)
        face_spec = None
        if fix_faces and subject is Subject.PERSON:
            try:
                face_spec = resolve_face_model(self.models_dir)
            except ModelNotInstalled:
                log.info("enhance: no face model installed; faces are upscaled only")
        report = progress or (lambda stage, done, total: None)

        t1 = time.perf_counter()
        with self._turns:
            self._cutouts_waiting += 1
            warming = self._warming
        if warming is not None:  # any warm-up: enhance uses other models, so stop it
            warming[2].set()
            if hasattr(warming[1], "cancel_warmup"):
                warming[1].cancel_warmup()
        notices: list[str] = []
        try:
            with self._infer_lock:
                timings["wait"] = time.perf_counter() - t1
                t1 = time.perf_counter()
                out, device, cpu = self._run_enhance_model(
                    up_spec, self._upscaler_factory,
                    lambda m: upscale_image(image.rgb, m, made,
                                            progress=lambda d, n: report("upscale", d, n)))
                timings["upscale"] = time.perf_counter() - t1
                if cpu:
                    notices.append(ENHANCE_CPU_NOTICE)
                fixed = 0
                if face_spec is not None:
                    t2 = time.perf_counter()
                    faces = self.face_finder.detect(_as_uint8(out), 0.5)
                    timings["find faces"] = time.perf_counter() - t2
                    if faces:
                        report("faces", 0, len(faces))
                        t2 = time.perf_counter()
                        fixes, _, cpu = self._run_enhance_model(
                            face_spec, self._face_factory,
                            lambda m: restore_faces(out, faces, m, face_strength))
                        fixed = len(fixes)
                        timings["fix faces"] = time.perf_counter() - t2
                        if cpu and not notices:
                            notices.append(ENHANCE_CPU_NOTICE)
                        report("faces", len(faces), len(faces))
        finally:
            with self._turns:
                self._cutouts_waiting -= 1
                self._turns.notify_all()

        if made < scale:
            notices.insert(0, f"Enhanced {made}x instead of {scale}x: at {scale}x the picture would be "
                              f"bigger than the {MAX_OUTPUT_PIXELS // 1_000_000} megapixel limit.")
        debug.record_timings(
            f"enhance {image.width}x{image.height} -> {out.shape[1]}x{out.shape[0]}", timings,
            model=up_spec.id, device=device, scale=f"{made}x", faces_fixed=fixed,
        )
        return EnhanceResult(
            image=LoadedImage(rgb=out, icc_profile=image.icc_profile, source_path=image.source_path),
            scale=made,
            requested_scale=scale,
            upscaler_id=up_spec.id,
            face_model_id=face_spec.id if face_spec is not None else None,
            faces_fixed=fixed,
            device=device,
            timings=timings,
            notice=" ".join(notices) or None,
        )

    def _run_enhance_model(self, spec: EnhanceModelSpec, factory, run):
        """Load ``spec``, call ``run(model)``, drop the model. Returns (value, device, on_cpu_fallback).

        On running out of memory: unload everything, try the GPU once more, then the CPU
        (``device="auto"`` only). Never recovers inside the ``except`` block, whose
        traceback would keep the failed session's memory alive.
        """
        path = enhance_model_path(spec, self.models_dir)
        devices = [self.device]
        if self.device != "cpu":
            devices.append(self.device)
        if self.device == "auto":
            devices.append("cpu")
        last = ""
        for i, device in enumerate(devices):
            failed: str | None = None
            try:
                model = factory(spec, path, device)
                value = run(model)
                info = getattr(model, "session_info", None)
                where = "gpu" if (info is not None and info.on_gpu) else ("cpu" if info is not None else "n/a")
                return value, where, device == "cpu" and i > 0
            except Exception as exc:
                if not is_out_of_memory(exc):
                    raise
                failed = str(exc)
            finally:
                model = None
                gc.collect()
            log.warning("%s ran out of memory on %s (%s); unloading all models", spec.id, device, failed)
            last = failed
            self.release_models()
        log.error("giving up on %s: %s", spec.id, last)
        raise GpuOutOfMemory("Your computer ran out of memory enhancing this picture. "
                             "Close other programs and try again, or pick a smaller scale.")

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


def _as_uint8(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    return np.round(rgb.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)


def _timed_prep(image: LoadedImage, opts: RefineOptions) -> tuple[ImagePrep, float]:
    t = time.perf_counter()
    prep = prepare_image(image, opts)
    return prep, time.perf_counter() - t
