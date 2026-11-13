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
    # Where the PyTorch weights come from when we export the ONNX ourselves:
    # "hf:<owner>/<repo>" (a transformers BiRefNet repo) or
    # "hf:<owner>/<repo>/<file>.pth" (a fine-tuned state dict for the base BiRefNet).
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
            export_source="hf:joelseytre/toonout/birefnet_finetuned_toonout.pth",
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


# --- Enhance (upscaling and face restoration) ----------------------------------------
#
# Enhance is an optional step before the cutout: it makes a small or blurry picture
# bigger and sharper (Real-ESRGAN), and can repair faces in photos (GFPGAN). Upscalers
# are always 4x networks; 2x and 1x ("clean up only") are made by averaging 2x2 or 4x4
# blocks of the 4x result, which is exact per tile and sharper than a native 2x model.


class EnhanceRole(str, Enum):
    UPSCALE = "upscale"
    FACE = "face"


@dataclass(frozen=True)
class EnhanceModelSpec:
    id: str
    role: EnhanceRole
    filename: str
    # How much bigger the network makes its input (4 for the upscalers, 1 for face models).
    scale: int = 4
    # Square side a face model works at (faces are aligned to it); 0 for upscalers.
    face_size: int = 0
    # Direct download(s) of a ready-made ONNX file, first that works wins.
    urls: tuple[str, ...] = ()
    # PyTorch weights we export ourselves (scripts/export_realesrgan_onnx.py): a URL to a .pth.
    export_source: str | None = None
    # RRDBNet size for the exporter (Real-ESRGAN x4plus: 23 blocks; anime 6B: 6 blocks).
    num_block: int = 23
    sha256: str | None = None
    approx_size_mb: int = 0
    description: str = ""


ENHANCE_REGISTRY: dict[str, EnhanceModelSpec] = {
    spec.id: spec
    for spec in (
        EnhanceModelSpec(
            id="realesrgan-x4plus",
            role=EnhanceRole.UPSCALE,
            filename="realesrgan-x4plus.onnx",
            export_source="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            num_block=23,
            approx_size_mb=67,
            description="Real-ESRGAN x4plus. Upscales and cleans up photos (blur, noise, JPEG blocks).",
        ),
        EnhanceModelSpec(
            id="realesrgan-x4plus-anime-6b",
            role=EnhanceRole.UPSCALE,
            filename="realesrgan-x4plus-anime-6b.onnx",
            export_source=(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            ),
            num_block=6,
            approx_size_mb=18,
            description="Real-ESRGAN x4plus anime 6B. Keeps line art crisp and flat colours clean.",
        ),
        EnhanceModelSpec(
            id="gfpgan-1.4",
            role=EnhanceRole.FACE,
            filename="gfpgan-1.4.onnx",
            scale=1,
            face_size=512,
            urls=(
                "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
                "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
                "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx",
            ),
            approx_size_mb=340,
            description="GFPGAN 1.4. Repairs blurry or blocky faces in photos (not used for anime).",
        ),
    )
}

DEFAULT_UPSCALERS: dict[Subject, tuple[str, ...]] = {
    Subject.PERSON: ("realesrgan-x4plus", "realesrgan-x4plus-anime-6b"),
    Subject.ANIME: ("realesrgan-x4plus-anime-6b", "realesrgan-x4plus"),
}
DEFAULT_FACE_MODELS: tuple[str, ...] = ("gfpgan-1.4",)


def enhance_model_path(spec: EnhanceModelSpec, models_dir: Path) -> Path:
    return Path(models_dir) / spec.filename


def installed_enhance_models(models_dir: Path) -> list[EnhanceModelSpec]:
    return [s for s in ENHANCE_REGISTRY.values() if enhance_model_path(s, models_dir).is_file()]


def _resolve_enhance(order: tuple[str, ...], models_dir: Path, model_id: str | None, what: str) -> EnhanceModelSpec:
    if model_id is not None:
        if model_id not in ENHANCE_REGISTRY:
            raise KeyError(f"unknown enhance model {model_id!r}; known: {sorted(ENHANCE_REGISTRY)}")
        spec = ENHANCE_REGISTRY[model_id]
        if not enhance_model_path(spec, models_dir).is_file():
            raise ModelNotInstalled(f"enhance model {model_id!r} not found at {enhance_model_path(spec, models_dir)}")
        return spec
    for candidate in order:
        spec = ENHANCE_REGISTRY[candidate]
        if enhance_model_path(spec, models_dir).is_file():
            return spec
    raise ModelNotInstalled(
        f"no {what} model installed in {models_dir}; expected one of "
        f"{[ENHANCE_REGISTRY[c].filename for c in order]}"
    )


def resolve_upscaler(subject: Subject, models_dir: Path, model_id: str | None = None) -> EnhanceModelSpec:
    """The upscaler for this subject: an explicit id, or the best one installed."""
    return _resolve_enhance(DEFAULT_UPSCALERS[Subject(subject)], models_dir, model_id, "upscaling")


def resolve_face_model(models_dir: Path, model_id: str | None = None) -> EnhanceModelSpec:
    """The face restoration model: an explicit id, or the best one installed."""
    return _resolve_enhance(DEFAULT_FACE_MODELS, models_dir, model_id, "face restoration")
