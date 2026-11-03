"""Command-line front end, used for development, benchmarking and the installer smoke test.

    python -m cutout photo.jpg                     -> photo_cutout.png
    python -m cutout art.png --subject anime -o out.png
    python -m cutout --list-models
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from .core.engine import CutoutEngine
from .core.models import REGISTRY, Subject, default_models_dir, installed_models
from .core.refine import RefineOptions
from .core.runtime import diagnose


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cutout", description="Remove the background from one image.")
    p.add_argument("input", nargs="?", type=Path, help="image to process")
    p.add_argument("-o", "--output", type=Path, help="output PNG (default: <input>_cutout.png)")
    p.add_argument("--subject", choices=[s.value for s in Subject], default=Subject.PERSON.value)
    p.add_argument("--model", choices=sorted(REGISTRY), help="force a specific model")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--no-refine", action="store_true", help="skip guided edge refinement")
    p.add_argument("--no-decontaminate", action="store_true", help="skip edge colour cleanup")
    p.add_argument("--list-models", action="store_true", help="show known and installed models")
    p.add_argument("--diagnose", action="store_true", help="print GPU/CUDA setup details and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    models_dir = args.models_dir or default_models_dir()

    if args.diagnose:
        print("\n".join(diagnose()))
        return 0

    if args.list_models:
        have = {s.id for s in installed_models(models_dir)}
        print(f"models dir: {models_dir}")
        for spec in REGISTRY.values():
            mark = "installed" if spec.id in have else "missing  "
            print(f"  [{mark}] {spec.id:22s} {spec.subject.value:7s} {spec.description}")
        return 0

    if args.input is None:
        build_parser().error("input image is required")
    output = args.output or args.input.with_name(f"{args.input.stem}_cutout.png")

    opts = replace(RefineOptions(), guided=not args.no_refine, decontaminate=not args.no_decontaminate)
    engine = CutoutEngine(models_dir=models_dir, device=args.device, refine=opts)
    try:
        result = engine.remove_background(args.input, Subject(args.subject), args.model)
        t = time.perf_counter()
        result.save_png(output)
        result.timings["compose+save"] = time.perf_counter() - t
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    timing = ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items())
    print(
        f"{output}  {result.image.width}x{result.image.height}  "
        f"model={result.model_id} device={result.device}  ({timing})"
    )
    return 0
