"""Cutout: offline background removal for people and anime characters."""

from .core.engine import CutoutEngine, CutoutResult
from .core.imageio import LoadedImage, load_image, save_png
from .core.models import Subject
from .core.refine import RefineOptions, compose_rgba

__all__ = [
    "CutoutEngine",
    "CutoutResult",
    "LoadedImage",
    "RefineOptions",
    "Subject",
    "compose_rgba",
    "load_image",
    "save_png",
]
__version__ = "0.1.0"
