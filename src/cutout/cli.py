"""Command-line front end, used for development, benchmarking and the installer smoke test.

    python -m cutout photo.jpg                     -> photo_cutout.png
    python -m cutout art.png --subject anime -o out.png
    python -m cutout small.jpg --enhance 4         -> upscale 4x first, then cut out
    python -m cutout small.jpg --enhance 2 --enhance-only   -> small_enhanced_2x.png (no cutout)
    python -m cutout --list-models
    cutout-cli --check-updates                     (installed app only)
    cutout-cli --update                            download and install the latest release
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from .core.engine import CutoutEngine
from .core.head import NoFaceFound
from .core.enhance import SCALES, EnhanceTooLarge
from .core.models import (
    ENHANCE_REGISTRY,
    PROMPT_REGISTRY,
    REGISTRY,
    Subject,
    default_models_dir,
    installed_enhance_models,
    installed_models,
    installed_prompt_models,
)
from .core.imageio import save_rgb_png
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
    p.add_argument("--head", action="store_true", help="keep only the head (face and hair), cut at the neck")
    e = p.add_argument_group("enhance (upscale and repair before the cutout)")
    e.add_argument("--enhance", type=int, choices=SCALES, metavar="{1,2,4}",
                   help="make the picture this many times bigger and sharper first (1 = clean up only)")
    e.add_argument("--enhance-only", action="store_true",
                   help="save the enhanced picture (no background removal)")
    e.add_argument("--no-fix-faces", action="store_true", help="don't repair faces (photos only)")
    e.add_argument("--face-strength", type=float, default=0.7,
                   help="how much of the repaired face to use, 0-1 (default 0.7)")
    p.add_argument("--no-refine", action="store_true", help="skip guided edge refinement")
    p.add_argument("--no-decontaminate", action="store_true", help="skip edge colour cleanup")
    p.add_argument("--list-models", action="store_true", help="show known and installed models")
    p.add_argument("--diagnose", action="store_true", help="print GPU/CUDA setup details and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    u = p.add_argument_group("updates (installed app only)")
    u.add_argument("--check-updates", action="store_true", help="look for a newer release on GitHub")
    u.add_argument("--update", action="store_true",
                   help="download the latest release (or the parts missing from this one) and install it")
    u.add_argument("--manifest", help="release manifest URL or file to update to (default: latest on GitHub)")
    u.add_argument("--from-dir", type=Path, action="append", default=[],
                   help="take release files from this folder when present instead of downloading them")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    models_dir = args.models_dir or default_models_dir()

    if args.version or args.check_updates or args.update:
        from .update.cli import run

        return run(args)

    if args.diagnose:
        print("\n".join(diagnose()))
        return 0

    if args.list_models:
        have = {s.id for s in installed_models(models_dir)}
        print(f"models dir: {models_dir}")
        for spec in REGISTRY.values():
            mark = "installed" if spec.id in have else "missing  "
            print(f"  [{mark}] {spec.id:22s} {spec.subject.value:7s} {spec.description}")
        have_prompt = {s.id for s in installed_prompt_models(models_dir)}
        for pspec in PROMPT_REGISTRY.values():
            mark = "installed" if pspec.id in have_prompt else "missing  "
            print(f"  [{mark}] {pspec.id:22s} {'click':7s} {pspec.description}")
        have_enh = {s.id for s in installed_enhance_models(models_dir)}
        for espec in ENHANCE_REGISTRY.values():
            mark = "installed" if espec.id in have_enh else "missing  "
            print(f"  [{mark}] {espec.id:22s} {espec.role.value:7s} {espec.description}")
        return 0

    if args.input is None:
        build_parser().error("input image is required")
    if args.enhance_only and args.enhance is None:
        args.enhance = 2
    if args.enhance_only:
        output = args.output or args.input.with_name(f"{args.input.stem}_enhanced_{args.enhance}x.png")
    else:
        output = args.output or args.input.with_name(f"{args.input.stem}_cutout.png")

    opts = replace(RefineOptions(), guided=not args.no_refine, decontaminate=not args.no_decontaminate)
    engine = CutoutEngine(models_dir=models_dir, device=args.device, refine=opts)
    subject = Subject(args.subject)
    try:
        source = args.input
        enhanced = None
        if args.enhance is not None:
            enhanced = engine.enhance(args.input, subject, args.enhance, fix_faces=not args.no_fix_faces,
                                      face_strength=args.face_strength)
            source = enhanced.image
            if enhanced.notice:
                print(f"note: {enhanced.notice}", file=sys.stderr)
            if args.enhance_only:
                t = time.perf_counter()
                save_rgb_png(output, enhanced.image.rgb, enhanced.image.icc_profile)
                enhanced.timings["save"] = time.perf_counter() - t
                timing = ", ".join(f"{k} {v:.2f}s" for k, v in enhanced.timings.items())
                print(f"{output}  {enhanced.image.width}x{enhanced.image.height}  {enhanced.scale}x  "
                      f"model={enhanced.upscaler_id} faces_fixed={enhanced.faces_fixed} "
                      f"device={enhanced.device}  ({timing})")
                return 0
        result = engine.remove_background(source, subject, args.model)
        if enhanced is not None:
            result.timings = {**{f"enhance {k}": v for k, v in enhanced.timings.items()}, **result.timings}
        if args.head:
            t = time.perf_counter()
            result.alpha = engine.head_only(result.image, result.alpha, Subject(args.subject)).alpha
            result.timings["head only"] = time.perf_counter() - t
        t = time.perf_counter()
        result.save_png(output)
        result.timings["compose+save"] = time.perf_counter() - t
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EnhanceTooLarge as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except NoFaceFound:
        print("error: no face found, so the head can't be cut out", file=sys.stderr)
        return 3

    timing = ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items())
    print(
        f"{output}  {result.image.width}x{result.image.height}  "
        f"model={result.model_id} device={result.device}  ({timing})"
    )
    return 0
