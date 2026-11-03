"""Build every model in the registry into one folder and check each one runs (build machine only).

    python scripts/build_models.py --models-dir build/models
    python scripts/build_models.py --models-dir build/models --only isnet-anime birefnet-general

For each model: download it (``url``) or export it from PyTorch (``export_source``,
in a child process so one failed or memory-hungry export cannot take the rest
down), then run it through the real pipeline (``python -m cutout``) on a
generated test image on CPU. Writes ``manifest.json`` next to the models with
size, SHA-256 and status. Exit code is 0 if at least one model per subject
works, because the app runs with any subset (best installed model wins).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cutout.core.models import DEFAULT_MODELS, REGISTRY, Subject, model_path  # noqa: E402
from fetch_models import download, sha256_of  # noqa: E402


def make_test_image(path: Path) -> None:
    """A soft-edged figure on a busy background, odd size so resize/crop bugs show up."""
    import numpy as np
    from PIL import Image

    h, w = 777, 555
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([xx / w * 255, yy / h * 255, (xx + yy) % 64 * 4], -1)
    body = ((xx - w / 2) / 150) ** 2 + ((yy - h * 0.6) / 260) ** 2 < 1
    head = ((xx - w / 2) / 80) ** 2 + ((yy - h * 0.22) / 95) ** 2 < 1
    img = bg.copy()
    img[body | head] = (230, 180, 150)
    Image.fromarray(img.clip(0, 255).astype("uint8")).save(path)


def validate(model_id: str, models_dir: Path, sample: Path, out_dir: Path) -> str | None:
    """Run the model end to end on CPU; return an error string, or None if it worked."""
    from PIL import Image

    out = out_dir / f"check_{model_id}.png"
    cmd = [sys.executable, "-m", "cutout", str(sample), "--model", model_id, "--device", "cpu",
           "--models-dir", str(models_dir), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=_env())
    if proc.returncode != 0:
        return (proc.stderr or proc.stdout).strip()[-800:]
    with Image.open(out) as im, Image.open(sample) as src:
        if im.mode != "RGBA" or im.size != src.size:
            return f"bad output {im.mode} {im.size}, expected RGBA {src.size}"
    print(f"  {proc.stdout.strip()}")
    return None


def _env():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def build_one(model_id: str, models_dir: Path) -> str | None:
    spec = REGISTRY[model_id]
    dest = model_path(spec, models_dir)
    if dest.is_file():
        print(f"{model_id}: already present ({dest.stat().st_size / 1e6:.0f} MB)")
        return None
    try:
        if spec.url:
            print(f"{model_id}: downloading {spec.url}")
            download(spec.url, dest)
        elif spec.export_source:
            print(f"{model_id}: exporting from {spec.export_source}")
            cmd = [sys.executable, str(ROOT / "scripts" / "export_birefnet_onnx.py"), model_id,
                   "--models-dir", str(models_dir)]
            proc = subprocess.run(cmd, cwd=ROOT, env=_env())
            if proc.returncode != 0:
                dest.unlink(missing_ok=True)
                return f"export exited with code {proc.returncode}"
        else:
            return "no url and no export_source"
    except Exception as exc:  # network errors etc.: record and move on
        dest.unlink(missing_ok=True)
        return f"{type(exc).__name__}: {exc}"
    return None if dest.is_file() else "no file produced"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="model ids (default: all)")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    work = models_dir.parent / "model-checks"
    work.mkdir(exist_ok=True)
    sample = work / "sample.png"
    make_test_image(sample)

    manifest = {}
    for mid in args.only or list(REGISTRY):
        spec = REGISTRY[mid]
        t = time.perf_counter()
        error = build_one(mid, models_dir)
        dest = model_path(spec, models_dir)
        if error is None and not args.skip_validate:
            error = validate(mid, models_dir, sample, work)
            if error:
                dest.unlink(missing_ok=True)  # never ship a model that does not run
        entry = {"subject": spec.subject.value, "file": spec.filename, "ok": error is None,
                 "seconds": round(time.perf_counter() - t, 1)}
        if error is None:
            digest = sha256_of(dest)
            if spec.sha256 and digest != spec.sha256:
                entry.update(ok=False, error=f"sha256 {digest} != pinned {spec.sha256}")
                dest.unlink()
            else:
                entry.update(sha256=digest, bytes=dest.stat().st_size)
        else:
            entry["error"] = error
        manifest[mid] = entry
        print(f"{mid}: {'OK' if entry['ok'] else 'FAILED  ' + entry['error']}")

    (models_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    missing = [s.value for s in Subject
               if not any(manifest.get(m, {}).get("ok") for m in DEFAULT_MODELS[s])]
    if missing:
        print(f"no working model for: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
