"""Check for, download and apply updates (no Qt here; the window and the CLI share it).

    status = local_status()                      # what is installed (None when run from source)
    target = fetch_manifest()                    # latest release, or status.release to finish setup
    plan = plan_update(status, target)           # which packs to fetch, which old files go
    adds = Downloader(plan, work_dir(target)).run()
    launch_apply(write_apply_plan(status.root, plan, adds, ...))   # then the app exits

Downloads resume after a dropped connection, every part is SHA-256 checked, and
every unpacked pack is checked against its content hash before anything in the
install folder is touched. Swapping the files happens in ``apply_update.ps1``
after the app has closed, because Windows does not let a running program
overwrite its own files. Settings (stored in the registry by Qt) and logs
(``%LOCALAPPDATA%\\Cutout``) live outside the install folder and are never touched.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import packs as P
from .manifest import LATEST_MANIFEST_URL, Manifest, ManifestError, Pack, Part, is_newer

log = logging.getLogger(__name__)

USER_AGENT = "Cutout-Updater"
#: ``progress(done_bytes, total_bytes, what)``
Progress = Callable[[int, int, str], None]
_CHUNK = 1024 * 1024


class UpdateError(RuntimeError):
    """A user-facing reason the update could not be checked, downloaded or installed."""


class Cancelled(UpdateError):
    pass


# ----------------------------------------------------------------- locations


def app_dir() -> Path | None:
    """The install folder of the frozen app (``CUTOUT_APP_DIR`` overrides), else ``None``."""
    env = os.environ.get("CUTOUT_APP_DIR")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def updates_dir() -> Path:
    """``%LOCALAPPDATA%\\Cutout\\updates`` (``CUTOUT_UPDATES_DIR`` overrides)."""
    env = os.environ.get("CUTOUT_UPDATES_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "Cutout" / "updates"
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "cutout" / "updates"


def work_dir(target: Manifest) -> Path:
    return updates_dir() / f"v{target.version}"


def result_path() -> Path:
    return updates_dir() / "last-result.json"


# ------------------------------------------------------------- local status


def load_state(root: Path) -> dict | None:
    try:
        data = json.loads((Path(root) / P.STATE_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data.get("packs"), dict) else None
    except (OSError, ValueError, AttributeError):
        return None


def save_state(root: Path, version: str, have: dict[str, str]) -> None:
    (Path(root) / P.STATE_FILE).write_text(
        json.dumps({"version": version, "packs": dict(sorted(have.items()))}, indent=1), encoding="utf-8"
    )


def _files_exist(root: Path, pack: Pack) -> bool:
    return all((root / f).is_file() for f in pack.files)


@dataclass
class Status:
    root: Path
    release: Manifest  # what this install was built as (its release.json)
    have: dict[str, str]  # pack name -> hash, for packs that are fully installed

    @property
    def version(self) -> str:
        return self.release.version

    @property
    def missing(self) -> list[Pack]:
        return [p for n, p in sorted(self.release.packs.items()) if self.have.get(n) != p.hash]


def installed_packs(root: Path, release: Manifest, progress: Progress | None = None,
                    verify: bool = False) -> dict[str, str]:
    """Which packs of ``release`` are installed in ``root``.

    The first start after Setup (or whenever the saved state is from another
    version) hashes every file once, a few seconds, so files that Setup left in
    place from an older install are reused instead of downloaded again.
    """
    root = Path(root)
    state = load_state(root)
    if not verify and state and state.get("version") == release.version:
        return {n: h for n, h in state["packs"].items()
                if n in release.packs and release.packs[n].hash == h and _files_exist(root, release.packs[n])}
    total = sum(p.size for p in release.packs.values())
    done = 0

    def tick(n: int) -> None:
        nonlocal done
        done += n
        if progress is not None:
            progress(done, total, "Checking installed files")

    have = {}
    for name, pack in sorted(release.packs.items()):
        if _files_exist(root, pack) and P.pack_hash(root, pack.files, tick) == pack.hash:
            have[name] = pack.hash
        else:
            done += pack.size
    try:
        save_state(root, release.version, have)
    except OSError as exc:
        log.warning("cannot save install state: %s", exc)
    return have


def local_status(progress: Progress | None = None, root: Path | None = None) -> Status | None:
    """The installed app's status, or ``None`` when running from source (no updates there)."""
    root = root or app_dir()
    if root is None or not (Path(root) / P.RELEASE_FILE).is_file():
        return None
    release = Manifest.load(Path(root) / P.RELEASE_FILE)
    return Status(Path(root), release, installed_packs(root, release, progress))


def writable(root: Path) -> bool:
    probe = Path(root) / ".cutout-write-test"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


# --------------------------------------------------------------- the plan


@dataclass
class Plan:
    target: Manifest
    fetch: list[Pack]
    remove: list[str] = field(default_factory=list)
    from_version: str = ""

    @property
    def download_size(self) -> int:
        return sum(p.download_size for p in self.fetch)

    @property
    def is_upgrade(self) -> bool:
        return is_newer(self.target.version, self.from_version)


def plan_update(status: Status, target: Manifest) -> Plan:
    """What to fetch and delete to turn the install in ``status`` into ``target``."""
    root = status.root
    fetch = [p for n, p in sorted(target.packs.items()) if status.have.get(n) != p.hash]
    keep = target.file_set()
    remove: set[str] = set()
    for name, h in status.have.items():
        old = status.release.packs.get(name)
        new = target.packs.get(name)
        if old is not None and (new is None or new.hash != h):
            remove.update(f for f in old.files if f.lower() not in keep)
    # Leftovers from versions before this one (e.g. a library that was dropped).
    for rel in P.list_files(root):
        low = rel.lower()
        if low.startswith(P.OWNED_DIRS) and low not in keep:
            remove.add(rel)
    return Plan(target=target, fetch=fetch, remove=sorted(remove), from_version=status.version)


# --------------------------------------------------------------- network


def _open(url: str, start: int = 0, timeout: float = 30):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def fetch_manifest(url: str = LATEST_MANIFEST_URL, timeout: float = 20) -> Manifest:
    """The release manifest at ``url`` (a local path works too, for testing)."""
    if not str(url).startswith(("http://", "https://")):
        return Manifest.load(Path(url))
    try:
        with _open(url, timeout=timeout) as resp:
            text = resp.read(50_000_000).decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("No release has been published on GitHub yet.") from exc
        raise UpdateError(f"GitHub could not be checked right now (error {exc.code}). Try again later.") from exc
    except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
        raise UpdateError("Couldn't reach GitHub. Check the internet connection and try again.") from exc
    try:
        return Manifest.from_json(text)
    except ManifestError as exc:
        raise UpdateError(str(exc)) from exc


def describe(pack: Pack) -> str:
    names = {P.APP: "Cutout", P.LIBS: "libraries", P.QT: "window toolkit", P.GPU: "NVIDIA GPU support"}
    if pack.name.startswith(P.MODEL_PREFIX):
        return "AI model " + pack.name[len(P.MODEL_PREFIX):]
    return names.get(pack.name, pack.name)


class Downloader:
    """Fetches, checks and unpacks the packs of a plan into ``work`` (not the install)."""

    def __init__(self, plan: Plan, work: Path, progress: Progress | None = None,
                 cancel: Callable[[], bool] | None = None, mirrors: Iterable[Path] = (),
                 retries: int = 4):
        self.plan = plan
        self.work = Path(work)
        self.progress = progress or (lambda done, total, what: None)
        self.cancel = cancel or (lambda: False)
        self.mirrors = [Path(m) for m in mirrors]
        self.retries = retries
        self.total = plan.download_size
        self.done = 0

    def _check_cancel(self) -> None:
        if self.cancel():
            raise Cancelled("Update cancelled.")

    def _tick(self, n: int, what: str) -> None:
        self.done += n
        self.progress(self.done, self.total, what)
        self._check_cancel()

    def run(self) -> list[tuple[Path, str]]:
        """Return ``(source file, install-relative destination)`` for everything to move in."""
        self.work.mkdir(parents=True, exist_ok=True)
        _remove_other_work_dirs(self.work)
        adds: list[tuple[Path, str]] = []
        for i, pack in enumerate(self.plan.fetch, 1):
            what = f"Downloading {describe(pack)} ({i} of {len(self.plan.fetch)})"
            folder = self.work / "packs" / f"{pack.name}-{pack.hash[:16]}"
            if (folder / ".verified").is_file():
                self._tick(pack.download_size, what)
            else:
                archive = self._download_pack(pack, what)
                self.progress(self.done, self.total, f"Unpacking {describe(pack)}")
                self._unpack(pack, archive, folder)
                for part in pack.parts:
                    (self.work / "downloads" / part.name).unlink(missing_ok=True)
                archive.unlink(missing_ok=True)
            adds += [(folder / "files" / rel, P.safe_rel(rel)) for rel in pack.files]
        meta = self.work / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        self.plan.target.save(meta / P.RELEASE_FILE)
        state = {"version": self.plan.target.version,
                 "packs": {n: p.hash for n, p in sorted(self.plan.target.packs.items())}}
        (meta / P.STATE_FILE).write_text(json.dumps(state, indent=1), encoding="utf-8")
        adds += [(meta / P.RELEASE_FILE, P.RELEASE_FILE), (meta / P.STATE_FILE, P.STATE_FILE)]
        return adds

    # -- download

    def _download_pack(self, pack: Pack, what: str) -> Path:
        if not pack.parts:
            raise UpdateError(f"The release has no download for {describe(pack)}.")
        downloads = self.work / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        files = [self._download_part(part, downloads / part.name, what) for part in pack.parts]
        if len(files) == 1:
            return files[0]
        joined = downloads / f"{pack.name}-{pack.hash[:16]}.zip"
        with open(joined, "wb") as out:
            for f in files:
                with open(f, "rb") as src:
                    shutil.copyfileobj(src, out, _CHUNK)
        return joined

    def _download_part(self, part: Part, dest: Path, what: str) -> Path:
        if dest.is_file() and dest.stat().st_size == part.size and P.file_digest(dest) == part.sha256:
            self._tick(part.size, what)
            return dest
        dest.unlink(missing_ok=True)
        partial = dest.with_name(dest.name + ".partial")
        for attempt in range(2):  # a second full try if the finished file fails its check
            self._fetch_to(part, partial, what)
            if partial.stat().st_size == part.size and P.file_digest(partial) == part.sha256:
                partial.replace(dest)
                return dest
            log.warning("%s failed its checksum, downloading again", part.name)
            self.done -= partial.stat().st_size
            partial.unlink()
        raise UpdateError(f"The download of {part.name} keeps arriving damaged. Try again later.")

    def _fetch_to(self, part: Part, partial: Path, what: str) -> None:
        for mirror in self.mirrors:
            src = mirror / part.name
            if src.is_file():
                with open(src, "rb") as fin, open(partial, "wb") as fout:
                    while chunk := fin.read(_CHUNK):
                        fout.write(chunk)
                        self._tick(len(chunk), what)
                return
        have = partial.stat().st_size if partial.is_file() else 0
        if have > part.size:
            partial.unlink()
            have = 0
        self.done += have
        for attempt in range(self.retries + 1):
            try:
                self._stream(part, partial, what)
                return
            except Cancelled:
                raise
            except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (403, 404, 410):
                    raise UpdateError(f"{part.name} is no longer available on GitHub (error {exc.code}).") from exc
                if attempt == self.retries:
                    raise UpdateError(
                        "The download stopped and could not be resumed. Check the internet connection "
                        "and try again; it will continue where it left off."
                    ) from exc
                log.info("download of %s interrupted (%s), retrying", part.name, exc)
                for _ in range(2 ** attempt * 2):
                    time.sleep(0.5)
                    self._check_cancel()

    def _stream(self, part: Part, partial: Path, what: str) -> None:
        start = partial.stat().st_size if partial.is_file() else 0
        if start >= part.size:
            return
        with _open(part.url, start=start) as resp:
            resumed = start and getattr(resp, "status", 200) == 206
            if start and not resumed:  # the server ignored the range: start over
                self.done -= start
                start = 0
            with open(partial, "ab" if resumed else "wb") as out:
                while chunk := resp.read(_CHUNK):
                    out.write(chunk)
                    self._tick(len(chunk), what)
        if partial.stat().st_size < part.size:
            raise ConnectionError("connection closed early")

    # -- unpack

    def _unpack(self, pack: Pack, archive: Path, folder: Path) -> None:
        files_dir = folder / "files"
        if folder.exists():
            shutil.rmtree(folder)
        files_dir.mkdir(parents=True)
        expected = {f.lower(): f for f in pack.files}
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    rel = P.safe_rel(info.filename)
                    if rel.lower() not in expected:
                        raise UpdateError(f"The download of {describe(pack)} holds an unexpected file ({rel}).")
                    out = files_dir / expected[rel.lower()]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(out, "wb") as dst:
                        while chunk := src.read(_CHUNK):
                            dst.write(chunk)
                        self._check_cancel()
        except (zipfile.BadZipFile, ValueError) as exc:
            archive.unlink(missing_ok=True)
            raise UpdateError(f"The download of {describe(pack)} was damaged. Try again.") from exc
        if P.pack_hash(files_dir, pack.files) != pack.hash:
            shutil.rmtree(folder, ignore_errors=True)
            raise UpdateError(f"The download of {describe(pack)} did not match the release. Try again.")
        (folder / ".verified").write_text(pack.hash)


def _remove_other_work_dirs(keep: Path) -> None:
    """Free the space of downloads for any other version (an abandoned earlier update)."""
    parent = keep.parent
    for d in parent.glob("v*"):
        if d.is_dir() and d.resolve() != keep.resolve():
            shutil.rmtree(d, ignore_errors=True)


def cleanup_after_update() -> None:
    """Delete finished update downloads (called at startup)."""
    base = updates_dir()
    for d in base.glob("v*"):
        if d.is_dir() and not (d / "backup").exists():
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------- applying


def write_apply_plan(root: Path, plan: Plan, adds: list[tuple[Path, str]], work: Path,
                     restart: Path | None, wait_pid: int | None, retries: int = 30) -> Path:
    root = Path(root)
    work = Path(work)
    backup = work / "backup"
    if backup.exists():  # left by an apply that could not even roll back: keep it, start a new one
        backup.rename(work / f"backup-{int(time.time())}")
    data = {
        "version": plan.target.version,
        "app_dir": str(root),
        "backup_dir": str(backup),
        "wait_pid": int(wait_pid or 0),
        "remove": [P.safe_rel(r) for r in plan.remove],
        "add": [{"src": str(src), "dest": P.safe_rel(dest)} for src, dest in adds],
        "restart": str(restart) if restart else "",
        "log": str(work / "apply.log"),
        "result": str(result_path()),
        "retries": int(retries),
    }
    result_path().parent.mkdir(parents=True, exist_ok=True)
    result_path().unlink(missing_ok=True)
    path = work / "apply-plan.json"
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def apply_script(work: Path) -> Path:
    """Copy the helper script out of the install folder (which it is about to change)."""
    src = Path(__file__).with_name("apply_update.ps1")
    dest = Path(work) / "apply_update.ps1"
    shutil.copyfile(src, dest)
    return dest


def launch_apply(plan_path: Path) -> subprocess.Popen:
    """Start the helper that swaps the files once this process exits (Windows only)."""
    if sys.platform != "win32":
        raise UpdateError("Installing updates is only supported on Windows.")
    script = apply_script(Path(plan_path).parent)
    shell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    cmd = [str(shell) if shell.is_file() else "powershell.exe", "-NoProfile", "-NonInteractive",
           "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File", str(script), "-PlanPath", str(plan_path)]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    log.info("starting update helper: %s", " ".join(cmd))
    return subprocess.Popen(cmd, creationflags=flags, close_fds=True, cwd=str(Path(plan_path).parent))


def take_last_result() -> dict | None:
    """The outcome of the last update the helper applied, once (then forgotten)."""
    path = result_path()
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return None
    path.unlink(missing_ok=True)
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if data.get("ok"):
        cleanup_after_update()
    return data
