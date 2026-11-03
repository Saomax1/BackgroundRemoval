"""Model registry: every segmentation/matting model the engine can run.

A ``ModelSpec`` is pure data describing how to feed and read one ONNX file, so
adding or swapping a model never touches inference code. For each subject the
engine uses the first model in ``DEFAULT_MODELS[subject]`` whose file is present
in the models directory, so the app keeps working with whatever subset has been
installed (best model first, fallbacks after).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Subject(str, Enum):
    PERSON = "person"
    ANIME = "anime"


class ResizeMode(str, Enum):
    STRETCH = "stretch"  # squash to a square, as BiRefNet was trained
    LETTERBOX = "letterbox"  # keep aspect, pad to a square, as anime-seg ISNet was trained


@dataclass(frozen=True)
class ModelSpec:
    id: str
    subject: Subject
    filename: str
    input_size: int  # square side the network runs at
    resize_mode: ResizeMode
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    # Which graph output holds the final prediction (BiRefNet: last, ISNet: first).
    output_index: int = 0
    # "sigmoid": output is logits. "none": already in [0, 1]. "auto": decide per run.
    activation: str = "auto"
    # Direct download for ready-made ONNX files; None means "built by scripts/export_birefnet_onnx.py".
    url: str | None = None
    sha256: str | None = None
    # Where the PyTorch weights come from when we export the ONNX ourselves.
    export_source: str | None = None
    description: str = ""


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NO_NORM_MEAN = (0.0, 0.0, 0.0)
NO_NORM_STD = (1.0, 1.0, 1.0)


REGISTRY: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        # --- People -------------------------------------------------------------------
        ModelSpec(
            id="birefnet-hr-matting",
            subject=Subject.PERSON,
            filename="birefnet-hr-matting-2048.onnx",
            input_size=2048,
            resize_mode=ResizeMode.STRETCH,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            output_index=-1,
            export_source="hf:ZhengPeng7/BiRefNet_HR-matting",
            description="BiRefNet trained for alpha matting at 2048px. Best hair and soft edges.",
        ),
        ModelSpec(
            id="birefnet-general",
            subject=Subject.PERSON,
            filename="BiRefNet-general-epoch_244.onnx",
            input_size=1024,
            resize_mode=ResizeMode.STRETCH,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            output_index=-1,
            url="https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.onnx",
            description="BiRefNet general segmentation at 1024px. Strong fallback for any subject.",
        ),
        # --- Anime / cartoon ----------------------------------------------------------
        ModelSpec(
            id="toonout",
            subject=Subject.ANIME,
            filename="toonout-birefnet-1024.onnx",
            input_size=1024,
            resize_mode=ResizeMode.STRETCH,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            output_index=-1,
            export_source="hf:joelseytre/toonout",
            description="BiRefNet fine-tuned on anime characters (ToonOut). Best anime hair/edges.",
        ),
        ModelSpec(
            id="isnet-anime",
            subject=Subject.ANIME,
            filename="isnetis.onnx",
            input_size=1024,
            resize_mode=ResizeMode.LETTERBOX,
            mean=NO_NORM_MEAN,
            std=NO_NORM_STD,
            output_index=0,
            activation="none",
            url="https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx",
            description="ISNet from skytnt/anime-seg. Small, fast anime fallback.",
        ),
    )
}

DEFAULT_MODELS: dict[Subject, tuple[str, ...]] = {
    Subject.PERSON: ("birefnet-hr-matting", "birefnet-general"),
    Subject.ANIME: ("toonout", "isnet-anime", "birefnet-general"),
}


class ModelNotInstalled(FileNotFoundError):
    pass


def default_models_dir() -> Path:
    """``%CUTOUT_MODELS_DIR%`` if set, else ``models/`` next to the frozen exe or the repo root."""
    env = os.environ.get("CUTOUT_MODELS_DIR")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):  # PyInstaller build
        return Path(sys.executable).resolve().parent / "models"
    return Path(__file__).resolve().parents[3] / "models"


def model_path(spec: ModelSpec, models_dir: Path) -> Path:
    return Path(models_dir) / spec.filename


def installed_models(models_dir: Path) -> list[ModelSpec]:
    return [s for s in REGISTRY.values() if model_path(s, models_dir).is_file()]


def resolve_model(subject: Subject, models_dir: Path, model_id: str | None = None) -> ModelSpec:
    """Pick the model to run: an explicit id, or the best installed default for the subject."""
    if model_id is not None:
        if model_id not in REGISTRY:
            raise KeyError(f"unknown model {model_id!r}; known: {sorted(REGISTRY)}")
        spec = REGISTRY[model_id]
        if not model_path(spec, models_dir).is_file():
            raise ModelNotInstalled(f"model {model_id!r} not found at {model_path(spec, models_dir)}")
        return spec
    for candidate in DEFAULT_MODELS[subject]:
        spec = REGISTRY[candidate]
        if model_path(spec, models_dir).is_file():
            return spec
    raise ModelNotInstalled(
        f"no {subject.value} model installed in {models_dir}; "
        f"expected one of {[REGISTRY[c].filename for c in DEFAULT_MODELS[subject]]}"
    )


# --- Click-to-select (promptable) models ---------------------------------------------
#
# These take click points instead of segmenting the whole image. They ship as two ONNX
# files: a heavy image encoder that runs once per image (on load, in the background) and a
# tiny prompt decoder that runs on every click (a few ms on GPU). Both are produced by
# ``scripts/export_sam2_onnx.py`` with fixed input/output names (see prompt_segmenter.py).


@dataclass(frozen=True)
class PromptModelSpec:
    id: str
    encoder_filename: str
    decoder_filename: str
    input_size: int = 1024  # SAM 2 stretches the image to a 1024 square
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    low_res_size: int = 256  # side of the mask logits the decoder returns
    export_source: str | None = None
    encoder_sha256: str | None = None
    decoder_sha256: str | None = None
    approx_size_mb: int = 0
    description: str = ""


PROMPT_REGISTRY: dict[str, PromptModelSpec] = {
    spec.id: spec
    for spec in (
        PromptModelSpec(
            id="sam2.1-hiera-large",
            encoder_filename="sam2.1-hiera-large.encoder.onnx",
            decoder_filename="sam2.1-hiera-large.decoder.onnx",
            export_source="hf:facebook/sam2.1-hiera-large",
            approx_size_mb=900,
            description="SAM 2.1 Hiera-L. Most accurate click-to-select.",
        ),
        PromptModelSpec(
            id="sam2.1-hiera-base-plus",
            encoder_filename="sam2.1-hiera-base-plus.encoder.onnx",
            decoder_filename="sam2.1-hiera-base-plus.decoder.onnx",
            export_source="hf:facebook/sam2.1-hiera-base-plus",
            approx_size_mb=330,
            description="SAM 2.1 Hiera-B+. Nearly as good, a third of the size.",
        ),
        PromptModelSpec(
            id="sam2.1-hiera-small",
            encoder_filename="sam2.1-hiera-small.encoder.onnx",
            decoder_filename="sam2.1-hiera-small.decoder.onnx",
            export_source="hf:facebook/sam2.1-hiera-small",
            approx_size_mb=190,
            description="SAM 2.1 Hiera-S. Small fallback.",
        ),
    )
}

DEFAULT_PROMPT_MODELS: tuple[str, ...] = ("sam2.1-hiera-large", "sam2.1-hiera-base-plus", "sam2.1-hiera-small")


def prompt_model_paths(spec: PromptModelSpec, models_dir: Path) -> tuple[Path, Path]:
    return Path(models_dir) / spec.encoder_filename, Path(models_dir) / spec.decoder_filename


def _prompt_installed(spec: PromptModelSpec, models_dir: Path) -> bool:
    return all(p.is_file() for p in prompt_model_paths(spec, models_dir))


def installed_prompt_models(models_dir: Path) -> list[PromptModelSpec]:
    return [s for s in PROMPT_REGISTRY.values() if _prompt_installed(s, models_dir)]


def resolve_prompt_model(models_dir: Path, model_id: str | None = None) -> PromptModelSpec:
    """The click-to-select model to use: an explicit id, or the best one installed."""
    if model_id is not None:
        if model_id not in PROMPT_REGISTRY:
            raise KeyError(f"unknown prompt model {model_id!r}; known: {sorted(PROMPT_REGISTRY)}")
        spec = PROMPT_REGISTRY[model_id]
        if not _prompt_installed(spec, models_dir):
            raise ModelNotInstalled(f"prompt model {model_id!r} not found in {models_dir}")
        return spec
    for candidate in DEFAULT_PROMPT_MODELS:
        spec = PROMPT_REGISTRY[candidate]
        if _prompt_installed(spec, models_dir):
            return spec
    raise ModelNotInstalled(
        f"no click-to-select model installed in {models_dir}; expected the encoder and decoder of one of "
        f"{list(DEFAULT_PROMPT_MODELS)}"
    )
