"""Copy the working models into the frozen app (dist/Cutout/models), within a size budget.

    python packaging/stage_models.py build/models dist/Cutout/models

Reads ``manifest.json`` from scripts/build_models.py and copies every model that
passed its check. Inno Setup can put at most ~4 GB in one Setup.exe, and the
CUDA DLLs take ~1 GB of that after compression while model weights barely
compress, so if the models exceed ``--budget-gb`` the script drops *redundant*
ones (a model that, for every subject using it, has a better model installed),
largest first. It never drops the best model for a subject.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutout.core.models import DEFAULT_MODELS, REGISTRY  # noqa: E402


def choose(ok: dict[str, int], budget: int) -> tuple[list[str], list[str]]:
    """Return (keep, dropped) model ids. ``ok`` maps working model id -> size in bytes."""
    keep = set(ok)

    def redundant(mid: str) -> bool:
        users = [order for order in DEFAULT_MODELS.values() if mid in order]
        return bool(users) and all(
            any(better in keep for better in order[: order.index(mid)]) for order in users
        )

    dropped = []
    while sum(ok[m] for m in keep) > budget:
        candidates = sorted((m for m in keep if redundant(m)), key=lambda m: ok[m], reverse=True)
        if not candidates:
            break
        keep.discard(candidates[0])
        dropped.append(candidates[0])
    return [m for m in ok if m in keep], dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--budget-gb", type=float, default=2.6)
    args = ap.parse_args()

    manifest = json.loads((args.src / "manifest.json").read_text())
    ok = {mid: e["bytes"] for mid, e in manifest.items() if e.get("ok") and mid in REGISTRY}
    keep, dropped = choose(ok, int(args.budget_gb * 1e9))
    if not keep:
        print("no working models to stage", file=sys.stderr)
        return 1
    args.dest.mkdir(parents=True, exist_ok=True)
    for mid in keep:
        f = REGISTRY[mid].filename
        shutil.copy2(args.src / f, args.dest / f)
        print(f"staged  {mid:22s} {ok[mid] / 1e6:7.0f} MB")
    for mid in dropped:
        print(f"dropped {mid:22s} {ok[mid] / 1e6:7.0f} MB (redundant, over the {args.budget_gb} GB budget)")
    (args.dest / "manifest.json").write_text(
        json.dumps({m: manifest[m] for m in keep}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
