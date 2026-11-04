"""Stamp the build's version into the CI checkout: ``<major>.<minor>.<build number>``.

    python packaging/set_version.py 57        ->  0.1.57  (major.minor from src/cutout/__init__.py)

Every build of main becomes a release, so the version just counts builds. Bump
the major or minor number in ``src/cutout/__init__.py`` by hand for a big change.
Prints the version and, in GitHub Actions, sets ``VERSION`` for later steps.
"""

import os
import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "src" / "cutout" / "__init__.py"


def main() -> int:
    build = int(sys.argv[1])
    text = INIT.read_text(encoding="utf-8")
    current = re.search(r'__version__ = "([^"]+)"', text).group(1)
    major_minor = ".".join((current.split(".") + ["0"])[:2])
    version = f"{major_minor}.{build}"
    INIT.write_text(re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', text), encoding="utf-8")
    if os.environ.get("GITHUB_ENV"):
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
            f.write(f"VERSION={version}\n")
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
