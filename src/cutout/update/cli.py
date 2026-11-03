"""``cutout-cli --version / --check-updates / --update``: the updater without a window.

Used by the installer CI to finish a fresh install and apply an update end to end.
"""

from __future__ import annotations

import os
import sys

from . import client
from .manifest import LATEST_MANIFEST_URL, is_newer


def _mb(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


class _Printer:
    def __init__(self) -> None:
        self.last = (-1, "")

    def __call__(self, done: int, total: int, what: str) -> None:
        pct = int(done * 100 / total) if total else 100
        key = (pct // 5, what)
        if key != self.last:
            self.last = key
            print(f"  {pct:3d}%  {what}", flush=True)


def run(args) -> int:
    from .. import __version__

    if args.version:
        print(f"cutout {__version__}")
    status = client.local_status(progress=None)
    if args.version:
        if status is None:
            print("not an installed copy (no release.json), so it cannot update itself")
        else:
            print(f"installed release {status.version} in {status.root}: "
                  f"{len(status.have)} of {len(status.release.packs)} parts present")
            for pack in status.missing:
                print(f"  missing: {client.describe(pack)} ({_mb(pack.download_size)})")
        if not (args.check_updates or args.update):
            return 0
    if status is None:
        print("error: updates only work in the installed app", file=sys.stderr)
        return 2

    try:
        target = client.fetch_manifest(args.manifest or LATEST_MANIFEST_URL)
    except client.UpdateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    if is_newer(status.version, target.version):
        print(f"installed {status.version} is newer than {target.version}; nothing to do")
        return 0
    plan = client.plan_update(status, target)
    if not plan.is_upgrade and not plan.fetch:
        print(f"Cutout {status.version} is up to date and complete")
        return 0
    verb = f"update {status.version} -> {target.version}" if plan.is_upgrade else f"finish installing {target.version}"
    print(f"{verb}: download {_mb(plan.download_size)} in {len(plan.fetch)} part(s), "
          f"remove {len(plan.remove)} old file(s)")
    for pack in plan.fetch:
        print(f"  {client.describe(pack):34s} {_mb(pack.download_size)}")
    if not args.update:
        return 0

    if not client.writable(status.root):
        print(f"error: cannot write to {status.root}; run the Setup from the releases page instead",
              file=sys.stderr)
        return 4
    work = client.work_dir(target)
    try:
        adds = client.Downloader(plan, work, progress=_Printer(), mirrors=args.from_dir).run()
        plan_path = client.write_apply_plan(status.root, plan, adds, work, restart=None, wait_pid=os.getpid())
        helper = client.launch_apply(plan_path)
    except client.UpdateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"downloaded; the update is installed as soon as this command exits (helper pid {helper.pid})")
    print(f"result: {client.result_path()}")
    return 0
