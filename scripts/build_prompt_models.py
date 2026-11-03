"""Export the click-to-select (SAM 2.1) model for the installer and check it runs (build machine only).

    python scripts/build_prompt_models.py --models-dir build/prompt-models
    python scripts/build_prompt_models.py --models-dir build/prompt-models --model sam2.1-hiera-small

Exports one encoder + decoder pair with scripts/export_sam2_onnx.py (in a child
process), then runs it through the app's own ``OnnxSamSegmenter`` on CPU: one
encode of a generated picture and one click in the middle of the figure, which
must select it. Deletes the pair if anything fails, so a broken export is never
shipped; the app simply hides click-to-select when no prompt model is installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cutout.core.models import PROMPT_REGISTRY, prompt_model_paths  # noqa: E402

DEFAULT = "sam2.1-hiera-base-plus"  # best quality per MB that fits the ~4 GB Setup.exe


def check(model_id: str, models_dir: Path) -> None:
    import numpy as np

    from cutout.core.prompt_segmenter import OnnxSamSegmenter

    spec = PROMPT_REGISTRY[model_id]
    enc, dec = prompt_model_paths(spec, models_dir)
    seg = OnnxSamSegmenter(spec, enc, dec, device="cpu")
    h, w = 600, 800
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[...] = (40, 120, 200)
    disc = (xx - 400) ** 2 + (yy - 300) ** 2 < 150**2
    rgb[disc] = (240, 200, 40)
    emb = seg.encode(rgb)
    pred = seg.decode(emb, np.array([[400.0, 300.0]]), np.array([1]))
    mask = pred.logits[pred.best(1)] > 0
    r = mask.shape[0]
    inside = mask[int(r * 0.4) : int(r * 0.6), int(r * 0.42) : int(r * 0.58)].mean()
    corner = mask[: r // 8, : r // 8].mean()
    print(f"  click check: inside {inside:.2f} selected, corner {corner:.2f} selected")
    if inside < 0.9 or corner > 0.1:
        raise RuntimeError("clicking the disc did not select it")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT, choices=sorted(PROMPT_REGISTRY))
    args = ap.parse_args()
    models_dir = args.models_dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = prompt_model_paths(PROMPT_REGISTRY[args.model], models_dir)

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [sys.executable, str(ROOT / "scripts" / "export_sam2_onnx.py"), args.model, "--models-dir", str(models_dir)]
    try:
        if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
            raise RuntimeError("export failed")
        check(args.model, models_dir)
    except Exception as exc:
        for p in paths:
            p.unlink(missing_ok=True)
        print(f"{args.model}: FAILED {exc}", file=sys.stderr)
        return 1
    print(f"{args.model}: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
