"""Turn the frozen app folder into release packs, a manifest and the Setup.exe input.

    python packaging/make_release.py dist/Cutout dist/release --version 0.1.57 \
        --base-url https://github.com/Saomax1/BackgroundRemoval/releases/download/v0.1.57 \
        [--previous build/prev/cutout-release.json] [--notes-file notes.md] [--setup-dir dist/setup]

Writes to the output folder:

* ``cutout-<pack>-<hash>.zip`` for every pack (split into ``.001``, ``.002`` ...
  when over ``--part-size``, because GitHub release files are capped at 2 GB),
* ``cutout-release.json``, the manifest the app reads,
* ``upload.txt``, the files this release must upload. A pack whose content is
  identical to one in ``--previous`` keeps that release's download link, so the
  3 GB of models and GPU files are uploaded once, not on every release.

Also writes ``release.json`` into the app folder, and with ``--setup-dir`` copies
the packs that Setup.exe carries (app, libraries, Qt) plus ``release.json`` there.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutout.update import packs as P  # noqa: E402
from cutout.update.manifest import MANIFEST_NAME, Manifest, Pack, Part  # noqa: E402

GITHUB_LIMIT = 2 * 1024**3
DEFAULT_PART_SIZE = 1_900_000_000
# Model weights barely compress; storing them keeps CI fast.
STORED_SUFFIXES = (".onnx",)


def _sha256(path: Path) -> str:
    return P.file_digest(path)


def write_zip(root: Path, files: list[str], dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", allowZip64=True) as zf:
        for rel in files:
            method = zipfile.ZIP_STORED if rel.lower().endswith(STORED_SUFFIXES) else zipfile.ZIP_DEFLATED
            zf.write(root / rel, rel, compress_type=method, compresslevel=6 if method else None)


def split(path: Path, part_size: int) -> list[Path]:
    size = path.stat().st_size
    if size <= part_size:
        return [path]
    out = []
    with open(path, "rb") as src:
        i = 0
        while True:
            i += 1
            part = path.with_name(f"{path.name}.{i:03d}")
            written = 0
            with open(part, "wb") as dst:
                while written < part_size:
                    chunk = src.read(min(8 * 1024 * 1024, part_size - written))
                    if not chunk:
                        break
                    dst.write(chunk)
                    written += len(chunk)
            if written == 0:
                part.unlink()
                break
            out.append(part)
    path.unlink()
    return out


def build(app: Path, out: Path, version: str, base_url: str, previous: Manifest | None = None,
          notes: str = "", part_size: int = DEFAULT_PART_SIZE,
          skip_reused: bool = False) -> tuple[Manifest, list[Path]]:
    """Return the manifest and the files this release has to upload."""
    if part_size >= GITHUB_LIMIT:
        raise SystemExit("--part-size must stay under GitHub's 2 GB file limit")
    out.mkdir(parents=True, exist_ok=True)
    grouped = P.group(P.list_files(app))
    if P.APP not in grouped:
        raise SystemExit(f"{app} has no app files; run PyInstaller first")
    manifest = Manifest(version=version, packs={}, notes=notes)
    upload: list[Path] = []
    base_url = base_url.rstrip("/")
    for name, files in grouped.items():
        digest = P.pack_hash(app, files)
        size = sum((app / f).stat().st_size for f in files)
        old = previous.packs.get(name) if previous else None
        reused = old is not None and old.hash == digest and bool(old.parts)
        if not (reused and skip_reused):  # local copies also serve the CI install test
            zip_path = out / f"cutout-{name}-{digest[:12]}.zip"
            write_zip(app, files, zip_path)
            local = split(zip_path, part_size)
        if reused:
            parts = old.parts  # same content: keep the earlier release's download
        else:
            parts = [Part(p.name, f"{base_url}/{p.name}", p.stat().st_size, _sha256(p)) for p in local]
            upload += local
        manifest.packs[name] = Pack(name=name, hash=digest, size=size, files=files, parts=parts)
        mb = sum(p.size for p in parts) / 1e6
        print(f"{name:32s} {len(files):5d} files {size / 1e6:9.1f} MB -> {mb:9.1f} MB"
              f" in {len(parts)} part(s){'  (unchanged, reused)' if reused else ''}")
    manifest.save(out / MANIFEST_NAME)
    upload.append(out / MANIFEST_NAME)
    return manifest, upload


def stage_setup(app: Path, manifest: Manifest, setup_dir: Path) -> int:
    """Copy the packs Setup.exe carries into ``setup_dir``; return the bytes left to download."""
    if setup_dir.exists():
        shutil.rmtree(setup_dir)
    for name in P.SETUP_PACKS:
        for rel in manifest.packs[name].files:
            dest = setup_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(app / rel, dest)
            except OSError:
                shutil.copy2(app / rel, dest)
    shutil.copy2(app / P.RELEASE_FILE, setup_dir / P.RELEASE_FILE)
    return sum(p.download_size for n, p in manifest.packs.items() if n not in P.SETUP_PACKS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("app", type=Path, help="frozen app folder (dist/Cutout)")
    ap.add_argument("out", type=Path, help="where to write the packs and manifest")
    ap.add_argument("--version", required=True)
    ap.add_argument("--base-url", required=True, help="download URL prefix of this release's files")
    ap.add_argument("--previous", type=Path, help="the previous release's manifest, to reuse unchanged packs")
    ap.add_argument("--notes-file", type=Path)
    ap.add_argument("--setup-dir", type=Path, help="also stage the Setup.exe input here")
    ap.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    ap.add_argument("--skip-reused", action="store_true", help="don't write zips for packs reused from --previous")
    args = ap.parse_args(argv)

    (args.app / P.RELEASE_FILE).unlink(missing_ok=True)
    (args.app / P.STATE_FILE).unlink(missing_ok=True)
    previous = Manifest.load(args.previous) if args.previous and args.previous.is_file() else None
    if previous:
        print(f"previous release: {previous.version}")
    notes = args.notes_file.read_text(encoding="utf-8") if args.notes_file else ""
    manifest, upload = build(args.app, args.out, args.version, args.base_url, previous, notes, args.part_size,
                             args.skip_reused)
    manifest.save(args.app / P.RELEASE_FILE)
    (args.out / "upload.txt").write_text("\n".join(str(p) for p in upload) + "\n", encoding="utf-8")
    total = sum(p.download_size for p in manifest.packs.values())
    new = sum(p.stat().st_size for p in upload)
    print(f"release {args.version}: {total / 1e9:.2f} GB in all, {new / 1e6:.1f} MB new to upload")
    if args.setup_dir:
        later = stage_setup(args.app, manifest, args.setup_dir)
        print(f"setup input: {args.setup_dir} ({later / 1e9:.2f} GB downloaded on first start)")
        github_env = os.environ.get("GITHUB_ENV")
        if github_env:
            with open(github_env, "a", encoding="utf-8") as f:
                f.write(f"FIRST_RUN_GB={later / 1e9:.1f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
