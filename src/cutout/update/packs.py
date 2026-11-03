"""Which pack each installed file belongs to, and a content hash per pack.

Paths are relative to the install folder, with forward slashes, e.g.
``_internal/nvidia_dlls/cudnn64_9.dll`` or ``models/toonout.onnx``.

The split only decides how big an update is, never whether it is correct: a
pack is downloaded whenever any byte of any file in it changed. It is chosen so
that the files that change on almost every release (the two exes, which carry
all of Cutout's own Python code, and ``_internal/cutout``) sit in a small pack,
and the big files that almost never change sit in their own packs.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

APP = "app"  # Cutout.exe, cutout-cli.exe, Cutout's own data files
LIBS = "libs"  # Python and third-party libraries (NumPy, OpenCV, ONNX Runtime ...)
QT = "qt"  # PySide6 / Qt, needed to show any window
GPU = "gpu-runtime"  # CUDA + cuDNN DLLs and ONNX Runtime's CUDA provider
MODEL_PREFIX = "model-"

# Packs the Setup.exe carries. Everything else is downloaded on first start.
SETUP_PACKS = (APP, LIBS, QT)

# Files the updater writes next to the app; never part of a pack.
RELEASE_FILE = "release.json"
STATE_FILE = "install-state.json"
STATE_FILES = (RELEASE_FILE, STATE_FILE)

# Folders whose contents are fully owned by packs. A file in them that no pack
# lists is left over from an older version and is removed on the next update.
OWNED_DIRS = ("_internal/", "models/")

_CHUNK = 4 * 1024 * 1024


def norm(rel: str) -> str:
    return str(rel).replace("\\", "/").lstrip("/")


def pack_for(rel: str) -> str | None:
    """The pack a file belongs to, or ``None`` for the updater's own state files."""
    r = norm(rel)
    low = r.lower()
    if low in STATE_FILES:
        return None
    if low.startswith("models/") and low.endswith(".onnx"):
        stem = PurePosixPath(low).stem
        return MODEL_PREFIX + re.sub(r"[^a-z0-9.]+", "-", stem).strip("-")
    if low.startswith("_internal/nvidia_dlls/") or low.startswith(
        "_internal/onnxruntime/capi/onnxruntime_providers_cuda"
    ):
        return GPU
    if low.startswith(("_internal/pyside6/", "_internal/shiboken6/")):
        return QT
    if low.startswith("_internal/") and not low.startswith("_internal/cutout/") and low != "_internal/base_library.zip":
        return LIBS
    return APP


def safe_rel(rel: str) -> str:
    """``rel`` normalised, or ``ValueError`` if it could point outside the install folder."""
    raw = str(rel).replace("\\", "/")
    parts = raw.split("/")
    if not raw or raw.startswith("/") or ":" in raw or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe path in update: {rel!r}")
    return raw


def list_files(root: Path) -> list[str]:
    """Every file under ``root`` as a relative path, minus the updater's state files."""
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if pack_for(rel) is not None:
                out.append(rel)
    return sorted(out)


def group(files: list[str]) -> dict[str, list[str]]:
    packs: dict[str, list[str]] = {}
    for rel in files:
        name = pack_for(rel)
        if name is not None:
            packs.setdefault(name, []).append(norm(rel))
    return {k: sorted(v) for k, v in sorted(packs.items())}


def file_digest(path: Path, progress=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            if progress is not None:
                progress(len(chunk))
    return h.hexdigest()


def pack_hash(root: Path, files: list[str], progress=None) -> str | None:
    """SHA-256 over every file's path and content, or ``None`` if a file is missing."""
    root = Path(root)
    h = hashlib.sha256()
    for rel in sorted(norm(f) for f in files):
        path = root / rel
        if not path.is_file():
            return None
        h.update(rel.encode("utf-8") + b"\0" + file_digest(path, progress).encode() + b"\n")
    return h.hexdigest()
