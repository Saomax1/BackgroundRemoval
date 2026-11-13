"""Build the Enhance models (upscalers and face repair) and check each one runs (build machine only).

    python scripts/build_enhance_models.py --models-dir build/enhance-models
    python scripts/build_enhance_models.py --models-dir build/enhance-models --only realesrgan-x4plus

Upscalers are exported from their official PyTorch weights
(``scripts/export_realesrgan_onnx.py``, in a child process); face models are
downloaded as ready-made ONNX files (``urls``, first that works). Each model is then
run through the app's own classes on CPU: an upscaler must make a 4x picture that
looks like the input; a face model must take and return a 512x512 face. Anything that
fails is deleted, so a broken model is never shipped; the app hides Enhance (or Fix
faces) when its model is missing. Writes ``manifest.json`` with size, SHA-256 and status.
Exit code is 1 only if no upscaler works.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cutout.core.models import ENHANCE_REGISTRY, EnhanceRole, enhance_model_path  # noqa: E402
from fetch_models import download, sha256_of  # noqa: E402


def _env() -> dict:
    return dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))


def build_one(model_id: str, models_dir: Path) -> str | None:
    spec = ENHANCE_REGISTRY[model_id]
    dest = enhance_model_path(spec, models_dir)
    if dest.is_file():
        print(f"{model_id}: already present ({dest.stat().st_size / 1e6:.0f} MB)")
        return None
    errors = []
    if spec.export_source:
        print(f"{model_id}: exporting from {spec.export_source}")
        cmd = [sys.executable, str(ROOT / "scripts" / "export_realesrgan_onnx.py"), model_id,
               "--models-dir", str(models_dir)]
        code = subprocess.run(cmd, cwd=ROOT, env=_env()).returncode
        if code == 0 and dest.is_file():
            return None
        errors.append(f"export exited with code {code}")
    for url in spec.urls:
        print(f"{model_id}: downloading {url}")
        try:
            download(url, dest)
            return None
        except Exception as exc:  # try the next mirror
            dest.unlink(missing_ok=True)
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    return "; ".join(errors) or "no url and no export_source"


def check_upscaler(model_id: str, models_dir: Path) -> None:
    from cutout.core.enhance import OnnxUpscaler, upscale_image

    spec = ENHANCE_REGISTRY[model_id]
    up = OnnxUpscaler(spec, enhance_model_path(spec, models_dir), device="cpu")
    h, w = 61, 47  # odd sizes, two tiles across, so tiling and cropping are exercised
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[...] = (40, 120, 200)
    rgb[((xx - 23) ** 2 + (yy - 30) ** 2) < 15 ** 2] = (240, 200, 40)
    out = upscale_image(rgb, up, 4, tile=32, pad=8)
    if out.shape != (h * 4, w * 4, 3):
        raise RuntimeError(f"4x output is {out.shape}")
    back = out.reshape(h, 4, w, 4, 3).mean(axis=(1, 3))
    err = float(np.abs(back - rgb).mean())
    print(f"  upscale check: mean difference from the input after shrinking back {err:.1f}/255")
    if err > 12:
        raise RuntimeError("the upscaled picture does not look like the input")


def check_face(model_id: str, models_dir: Path) -> None:
    from cutout.core.enhance import OnnxFaceRestorer

    spec = ENHANCE_REGISTRY[model_id]
    fr = OnnxFaceRestorer(spec, enhance_model_path(spec, models_dir), device="cpu")
    s = spec.face_size
    face = np.full((s, s, 3), 0.6, np.float32)
    out = fr.restore(face)
    if out.shape != (s, s, 3) or not np.isfinite(out).all():
        raise RuntimeError(f"face output is {out.shape}")
    print(f"  face check: {s}x{s} in and out, mean {out.mean():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="model ids (default: all)")
    args = ap.parse_args()
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for mid in args.only or list(ENHANCE_REGISTRY):
        spec = ENHANCE_REGISTRY[mid]
        dest = enhance_model_path(spec, models_dir)
        t = time.perf_counter()
        error = build_one(mid, models_dir)
        if error is None:
            try:
                (check_upscaler if spec.role is EnhanceRole.UPSCALE else check_face)(mid, models_dir)
            except Exception as exc:
                error = f"check failed: {type(exc).__name__}: {exc}"
        entry = {"role": spec.role.value, "file": spec.filename, "ok": error is None,
                 "seconds": round(time.perf_counter() - t, 1)}
        if error is None:
            digest = sha256_of(dest)
            if spec.sha256 and digest != spec.sha256:
                error = f"sha256 {digest} != pinned {spec.sha256}"
                entry["ok"] = False
            else:
                entry.update(sha256=digest, bytes=dest.stat().st_size)
        if error is not None:
            dest.unlink(missing_ok=True)  # never ship a model that does not run
            entry["error"] = error
        manifest[mid] = entry
        print(f"{mid}: {'OK' if entry['ok'] else 'FAILED  ' + entry['error']}")

    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not any(e["ok"] and e["role"] == EnhanceRole.UPSCALE.value for e in manifest.values()):
        print("no working upscaler", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
