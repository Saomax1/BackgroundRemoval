"""Logging, debug mode and crash reports.

Cutout always keeps a small log file, so a failure can be reported even when
debug mode was off. Debug mode adds detail on top of that:

* every log level down to DEBUG, with thread names and full tracebacks;
* ONNX Runtime's own messages (session logs at INFO, native output captured);
* a RAM / GPU memory snapshot whenever a model loads or a cutout finishes.

Debug mode is switched on by any of: ``Help > Debug mode`` in the app (saved),
the "Cutout (debug mode)" Start menu shortcut (``Cutout.exe --debug``), or the
environment variable ``CUTOUT_DEBUG=1``.

``build_report()`` returns one plain-text report (system, GPU, models, recent
log, last crash) that the user can copy or save and paste into a chat.

Nothing here imports Qt, so the CLI and the frozen entry point can use it too.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

APP_DIR_NAME = "Cutout"
LOG_NAME = "cutout.log"
CRASH_NAME = "crash.log"  # faulthandler output of the current run (native crashes, hangs)
LAST_CRASH_NAME = "crash-previous.log"
NATIVE_NAME = "native.log"  # ONNX Runtime / CUDA output written straight to stderr (debug mode)
RUNNING_MARKER = "running.marker"  # left behind when the app did not close cleanly
REPORT_LOG_BYTES = 200_000  # how much of the log a report carries

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_state = {
    "enabled": False,
    "configured": False,
    "log_dir": None,
    "file_handler": None,
    "crash_file": None,
    "native_redirected": False,
    "previous_run_crashed": False,
}


# ------------------------------------------------------------------ switches
def env_requests_debug() -> bool:
    return os.environ.get("CUTOUT_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return bool(_state["enabled"])


def ort_log_severity() -> int:
    """ONNX Runtime session log level: 1 (info) in debug mode, 3 (errors only) otherwise."""
    return 1 if enabled() else 3


def set_enabled(on: bool) -> None:
    """Switch debug mode at run time. ORT's native output is captured from the next start."""
    _state["enabled"] = bool(on)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if on else logging.INFO)
    handler = _state["file_handler"]
    if handler is not None:
        handler.setLevel(logging.DEBUG if on else logging.INFO)
    _quiet_noisy_loggers()
    if on:
        log.info("debug mode ON")
        log.info("%s", system_summary_line())
    else:
        log.info("debug mode off")


def _quiet_noisy_loggers() -> None:
    for name in ("PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.INFO)


# ------------------------------------------------------------------ folders
def log_dir() -> Path:
    """``%LOCALAPPDATA%\\Cutout\\logs`` on Windows; ``CUTOUT_LOG_DIR`` overrides it."""
    if _state["log_dir"] is not None:
        return _state["log_dir"]
    override = os.environ.get("CUTOUT_LOG_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / APP_DIR_NAME / "logs"
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / APP_DIR_NAME.lower() / "logs"


def log_file() -> Path:
    return log_dir() / LOG_NAME


# ------------------------------------------------------------------ setup
def setup(debug: bool = False, console: bool = True, directory: str | os.PathLike | None = None) -> Path:
    """Configure logging for the process. Call once, first thing in ``main``.

    Returns the log folder. Safe to call again (later calls only change debug mode).
    """
    if _state["configured"]:
        set_enabled(debug or enabled())
        return log_dir()
    if directory is not None:
        _state["log_dir"] = Path(directory)
    folder = log_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        folder = Path(os.environ.get("TEMP") or "/tmp") / "cutout-logs"
        folder.mkdir(parents=True, exist_ok=True)
    _state["log_dir"] = folder
    _state["configured"] = True
    _state["enabled"] = bool(debug)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter(_FORMAT, _DATEFMT)
    try:
        fh = logging.handlers.RotatingFileHandler(
            folder / LOG_NAME, maxBytes=2_000_000, backupCount=3, encoding="utf-8", delay=False
        )
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG if debug else logging.INFO)
        root.addHandler(fh)
        _state["file_handler"] = fh
    except OSError as exc:  # read-only profile etc.: keep going without a file
        print(f"cutout: cannot write log file in {folder}: {exc}", file=sys.stderr)
    if console and sys.stderr is not None:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        ch.setLevel(logging.INFO)
        root.addHandler(ch)
    logging.captureWarnings(True)
    _quiet_noisy_loggers()

    _check_previous_run(folder)
    _install_crash_hooks(folder)
    if debug:
        _redirect_native_stderr(folder)

    log.info("=" * 60)
    log.info("Cutout %s starting (debug mode %s)", _app_version(), "ON" if debug else "off")
    log.info("%s", system_summary_line())
    if debug:
        log.debug("argv: %s", sys.argv)
        log.debug("log folder: %s", folder)
    return folder


def mark_clean_exit() -> None:
    """Call when the app closes normally, so the next start knows it did not crash."""
    try:
        (log_dir() / RUNNING_MARKER).unlink(missing_ok=True)
    except OSError:
        pass
    log.info("Cutout closed normally")


def previous_run_crashed() -> bool:
    return bool(_state["previous_run_crashed"])


def _check_previous_run(folder: Path) -> None:
    marker = folder / RUNNING_MARKER
    crash = folder / CRASH_NAME
    crashed = marker.exists()
    try:
        if crash.exists() and crash.stat().st_size > 0:
            crashed = True
            crash.replace(folder / LAST_CRASH_NAME)
        elif crashed:
            # Closed without a native traceback (killed, power loss, frozen): note it anyway.
            (folder / LAST_CRASH_NAME).write_text(
                "The previous run did not close normally (no native traceback was written).\n",
                encoding="utf-8",
            )
        marker.write_text(f"pid {os.getpid()} started {time.ctime()}\n", encoding="utf-8")
    except OSError:
        pass
    _state["previous_run_crashed"] = crashed


def _install_crash_hooks(folder: Path) -> None:
    # Native crashes (CUDA, access violations) kill the process before Python can log;
    # faulthandler still writes every thread's stack to this file.
    try:
        f = open(folder / CRASH_NAME, "w", encoding="utf-8")  # noqa: SIM115 - kept open for the process
        faulthandler.enable(file=f, all_threads=True)
        _state["crash_file"] = f
    except (OSError, RuntimeError, ValueError):
        pass

    def excepthook(exc_type, exc, tb):
        logging.getLogger("cutout.crash").critical(
            "uncaught exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb))
        )
        if sys.__excepthook__ is not None and sys.stderr is not None:
            sys.__excepthook__(exc_type, exc, tb)

    def thread_hook(args):
        logging.getLogger("cutout.crash").critical(
            "uncaught exception in thread %s:\n%s",
            getattr(args.thread, "name", "?"),
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    sys.excepthook = excepthook
    threading.excepthook = thread_hook


def _redirect_native_stderr(folder: Path) -> None:
    """Send what C/C++ code writes to stderr (ONNX Runtime, CUDA) into ``native.log``.

    A windowed exe has no console, so without this those messages are lost.
    """
    if _state["native_redirected"]:
        return
    try:
        f = open(folder / NATIVE_NAME, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
        _state["saved_fd2"] = os.dup(2)
        os.dup2(f.fileno(), 2)
        if sys.stderr is None or not getattr(sys.stderr, "isatty", lambda: False)():
            sys.stderr = f
        _state["native_redirected"] = True
        _state["native_file"] = f
        _state["saved_stderr"] = sys.__stderr__
    except (OSError, ValueError, AttributeError) as exc:
        log.debug("could not capture native stderr: %s", exc)


# ------------------------------------------------------------------ memory
def _fmt_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def ram_usage() -> dict[str, int]:
    """``{"process": bytes, "used": bytes, "total": bytes}`` (keys missing when unknown)."""
    out: dict[str, int] = {}
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            ms = MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(ms)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                out["total"] = ms.ullTotalPhys
                out["used"] = ms.ullTotalPhys - ms.ullAvailPhys

            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(pmc)
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                out["process"] = pmc.WorkingSetSize
        except Exception:  # noqa: BLE001 - diagnostics must never break the app
            pass
        return out
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, val = line.partition(":")
            info[key] = int(val.split()[0]) * 1024
        out["total"] = info["MemTotal"]
        out["used"] = info["MemTotal"] - info.get("MemAvailable", info.get("MemFree", 0))
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                out["process"] = int(line.split()[1]) * 1024
    except (OSError, KeyError, ValueError, IndexError):
        pass
    return out


def _nvidia_smi(*query: str) -> list[list[str]] | None:
    exe = "nvidia-smi"
    if sys.platform == "win32":
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        candidate = Path(sysroot) / "System32" / "nvidia-smi.exe"
        if candidate.exists():
            exe = str(candidate)
    try:
        proc = subprocess.run(
            [exe, *query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in proc.stdout.strip().splitlines() if line.strip()]


def gpu_usage() -> list[dict]:
    """One dict per NVIDIA GPU: name, driver, used/total MiB, and this process's MiB if known."""
    rows = _nvidia_smi("--query-gpu=index,name,driver_version,memory.used,memory.total,uuid")
    if not rows:
        return []
    mine: dict[str, int] = {}
    apps = _nvidia_smi("--query-compute-apps=pid,used_memory,gpu_uuid") or []
    for app in apps:
        try:
            if int(app[0]) == os.getpid():
                mine[app[2]] = mine.get(app[2], 0) + int(app[1])
        except (ValueError, IndexError):
            continue
    gpus = []
    for r in rows:
        try:
            gpus.append({
                "index": r[0], "name": r[1], "driver": r[2],
                "used_mib": int(float(r[3])), "total_mib": int(float(r[4])),
                "process_mib": mine.get(r[5]),
            })
        except (ValueError, IndexError):
            continue
    return gpus


def memory_snapshot() -> str:
    """One line like ``RAM app 1.20 GB, system 9.80/31.90 GB | GPU0 RTX 4070: VRAM 5.1/12.0 GB (app 4.8 GB)``."""
    parts = []
    ram = ram_usage()
    if ram:
        bits = []
        if "process" in ram:
            bits.append(f"app {_fmt_gb(ram['process'])}")
        if "total" in ram:
            bits.append(f"system {_fmt_gb(ram.get('used', 0))}/{_fmt_gb(ram['total'])}")
        parts.append("RAM " + ", ".join(bits))
    gpus = gpu_usage()
    for g in gpus:
        s = f"GPU{g['index']} {g['name']}: VRAM {g['used_mib'] / 1024:.2f}/{g['total_mib'] / 1024:.2f} GB"
        if g.get("process_mib") is not None:
            s += f" (app {g['process_mib'] / 1024:.2f} GB)"
        parts.append(s)
    if not gpus:
        parts.append("GPU: nvidia-smi not available")
    return " | ".join(parts) or "memory: unknown"


def log_memory(when: str) -> None:
    """In debug mode, log a memory snapshot labelled ``when``. A no-op otherwise (nvidia-smi is slow)."""
    if enabled():
        log.debug("memory %s: %s", when, memory_snapshot())


# ------------------------------------------------------------------ report
def _app_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "?"


def system_summary_line() -> str:
    return (
        f"{platform.platform()} | Python {platform.python_version()} | "
        f"frozen={bool(getattr(sys, 'frozen', False))} | CPUs {os.cpu_count()}"
    )


def _anonymise(text: str) -> str:
    """Replace the user's home folder (it contains their name) with a placeholder."""
    home = str(Path.home())
    if len(home) > 3:
        text = text.replace(home, "%USERPROFILE%")
        text = text.replace(home.replace("\\", "/"), "%USERPROFILE%")
    return text


def _tail(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # drop the partial first line
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _section(title: str, body: str) -> str:
    return f"\n===== {title} =====\n{body.rstrip()}\n"


def build_report(extra: str = "", models_dir: str | os.PathLike | None = None) -> str:
    """Everything needed to debug a failure, as one text block to paste into a chat."""
    out = [
        "CUTOUT DEBUG REPORT",
        f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Cutout {_app_version()} | debug mode {'ON' if enabled() else 'off'}",
        system_summary_line(),
    ]
    if extra:
        out.append(_section("What happened", extra))

    try:
        out.append(_section("Memory", memory_snapshot()))
    except Exception as exc:  # noqa: BLE001
        out.append(_section("Memory", f"failed: {exc}"))

    gpus = gpu_usage()
    if gpus:
        out.append(_section("NVIDIA GPU", "\n".join(
            f"GPU{g['index']}: {g['name']}, driver {g['driver']}, VRAM {g['used_mib']}/{g['total_mib']} MiB"
            for g in gpus
        )))
    else:
        out.append(_section("NVIDIA GPU", "nvidia-smi not found or failed (no NVIDIA driver?)"))

    try:
        from .runtime import diagnose

        out.append(_section("ONNX Runtime / CUDA", "\n".join(diagnose())))
    except Exception as exc:  # noqa: BLE001
        out.append(_section("ONNX Runtime / CUDA", f"diagnose failed: {exc!r}"))

    try:
        from .models import PROMPT_REGISTRY, REGISTRY, default_models_dir, installed_models, installed_prompt_models

        mdir = Path(models_dir) if models_dir else default_models_dir()
        have = {s.id for s in installed_models(mdir)} | {s.id for s in installed_prompt_models(mdir)}
        lines = [f"models folder: {mdir}"]
        for spec in [*REGISTRY.values(), *PROMPT_REGISTRY.values()]:
            lines.append(f"  [{'installed' if spec.id in have else 'missing  '}] {spec.id}")
        out.append(_section("Models", "\n".join(lines)))
    except Exception as exc:  # noqa: BLE001
        out.append(_section("Models", f"failed: {exc!r}"))

    folder = log_dir()
    last_crash = _tail(folder / LAST_CRASH_NAME, 30_000) if previous_run_crashed() else ""
    if last_crash.strip():
        out.append(_section("Previous run crashed", last_crash))
    current_crash = _tail(folder / CRASH_NAME, 30_000)
    if current_crash.strip():
        out.append(_section("Native crash (this run)", current_crash))
    native = "\n".join(  # drop ORT's per-optimiser chatter; keep provider, memory and error lines
        line for line in _tail(folder / NATIVE_NAME, 400_000).splitlines()
        if "GraphTransformer" not in line and "TraceSessionOptions" not in line
    )[-40_000:]
    if native.strip():
        out.append(_section("ONNX Runtime native output", native))
    handler = _state["file_handler"]
    if handler is not None:
        try:
            handler.flush()
        except Exception:  # noqa: BLE001
            pass
    out.append(_section(f"Log (last {REPORT_LOG_BYTES // 1000} KB)", _tail(log_file(), REPORT_LOG_BYTES) or "(empty)"))
    out.append("\n===== end of report =====\n")
    return _anonymise("\n".join(out))


def save_report(path: str | os.PathLike | None = None, extra: str = "", models_dir=None) -> Path:
    """Write ``build_report()`` to ``path`` (default: a timestamped file in the log folder)."""
    target = Path(path) if path else log_dir() / time.strftime("cutout-report-%Y%m%d-%H%M%S.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_report(extra, models_dir), encoding="utf-8")
    return target


def _reset_for_tests() -> None:
    root = logging.getLogger()
    for key in ("file_handler",):
        h = _state[key]
        if h is not None:
            root.removeHandler(h)
            h.close()
    if _state["crash_file"] is not None:
        try:
            faulthandler.disable()
            _state["crash_file"].close()
        except Exception:  # noqa: BLE001
            pass
    saved = _state.pop("saved_fd2", None)
    if saved is not None:
        os.dup2(saved, 2)
        os.close(saved)
        sys.stderr = _state.pop("saved_stderr", sys.__stderr__)
        _state.pop("native_file").close()
    _state.update(enabled=False, configured=False, log_dir=None, file_handler=None,
                  crash_file=None, native_redirected=False, previous_run_crashed=False)
