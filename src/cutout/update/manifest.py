"""``cutout-release.json``: what one release contains and where to download it.

    {
      "format": 1,
      "version": "0.1.57",
      "notes": "markdown shown in the update dialog",
      "packs": {
        "app": {"hash": "...", "size": 41000000, "files": ["Cutout.exe", ...],
                "parts": [{"name": "cutout-app-1a2b3c4d5e6f.zip", "url": "https://...",
                           "size": 18000000, "sha256": "..."}]},
        ...
      }
    }

A pack whose zip is over GitHub's 2 GB per-file limit is split into several
parts that are joined after download. A pack that did not change since the
previous release keeps pointing at that release's file, so it is uploaded once.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

FORMAT = 1
REPO = "Saomax1/BackgroundRemoval"
MANIFEST_NAME = "cutout-release.json"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
LATEST_MANIFEST_URL = f"https://github.com/{REPO}/releases/latest/download/{MANIFEST_NAME}"


class ManifestError(ValueError):
    pass


@dataclass
class Part:
    name: str
    url: str
    size: int
    sha256: str


@dataclass
class Pack:
    name: str
    hash: str
    size: int  # bytes once unpacked
    files: list[str]
    parts: list[Part] = field(default_factory=list)

    @property
    def download_size(self) -> int:
        return sum(p.size for p in self.parts)


@dataclass
class Manifest:
    version: str
    packs: dict[str, Pack]
    notes: str = ""
    format: int = FORMAT

    def to_json(self) -> str:
        data = {"format": self.format, "version": self.version, "notes": self.notes,
                "packs": {n: _pack_dict(p) for n, p in sorted(self.packs.items())}}
        return json.dumps(data, indent=1)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        try:
            data = json.loads(text)
            fmt = int(data.get("format", 0))
            if fmt > FORMAT:
                raise ManifestError(
                    "This update needs a newer updater. Download the latest Setup from " + RELEASES_PAGE
                )
            packs = {}
            for name, p in data["packs"].items():
                packs[name] = Pack(
                    name=name, hash=str(p["hash"]), size=int(p.get("size", 0)), files=list(p["files"]),
                    parts=[Part(str(q["name"]), str(q["url"]), int(q["size"]), str(q["sha256"]))
                           for q in p.get("parts", [])],
                )
            return cls(version=str(data["version"]), packs=packs, notes=str(data.get("notes", "")), format=fmt)
        except ManifestError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"the release description is damaged ({exc})") from exc

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def file_set(self) -> set[str]:
        return {f.lower() for p in self.packs.values() for f in p.files}


def _pack_dict(p: Pack) -> dict:
    d = asdict(p)
    d.pop("name")
    return d


def parse_version(text: str) -> tuple[int, ...]:
    """``"v0.1.57"`` -> ``(0, 1, 57)``. Anything that is not a number counts as 0."""
    nums = []
    for piece in str(text).strip().lstrip("vV").split("."):
        m = re.match(r"\d+", piece)
        nums.append(int(m.group()) if m else 0)
    while len(nums) > 1 and nums[-1] == 0:
        nums.pop()
    return tuple(nums)


def is_newer(candidate: str, current: str) -> bool:
    return parse_version(candidate) > parse_version(current)
