"""Download the ready-made ONNX models listed in the registry (build-time only; the app is offline).

    python scripts/fetch_models.py               # all downloadable models
    python scripts/fetch_models.py isnet-anime   # just one

Prints each file's SHA-256 so it can be pinned in ``ModelSpec.sha256``.
Models without a ``url`` are produced by ``export_birefnet_onnx.py`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutout.core.models import REGISTRY, default_models_dir, model_path  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="model ids (default: every model with a url)")
    ap.add_argument("--models-dir", type=Path, default=default_models_dir())
    args = ap.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids or [s.id for s in REGISTRY.values() if s.url]
    failed = False
    for mid in ids:
        spec = REGISTRY[mid]
        dest = model_path(spec, args.models_dir)
        if not spec.url:
            print(f"{mid}: no download url, build it with scripts/export_birefnet_onnx.py {mid}")
            continue
        if not dest.exists():
            print(f"{mid}: downloading {spec.url}")
            download(spec.url, dest)
        digest = sha256_of(dest)
        if spec.sha256 and digest != spec.sha256:
            print(f"{mid}: SHA-256 MISMATCH {digest} != {spec.sha256}", file=sys.stderr)
            failed = True
        else:
            print(f"{mid}: ok  sha256={digest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
