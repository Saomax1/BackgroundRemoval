"""Frozen entry point for ``Cutout.exe`` (windowed): the desktop app.

The UI lives in ``cutout.ui`` and exposes ``main(argv) -> int``. Until that
package lands, the app shows a message instead of crashing silently (a windowed
exe has no console to print a traceback to).
"""

import sys


def _message(text: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "Cutout", 0x10)
    except Exception:
        print(text, file=sys.stderr)


def run() -> int:
    try:
        from cutout.ui import main
    except ImportError as exc:
        _message(
            "The Cutout window is not part of this build yet.\n\n"
            f"({exc})\n\nThe command-line tool cutout-cli.exe in the install folder works."
        )
        return 1
    try:
        return int(main(sys.argv[1:]) or 0)
    except Exception:
        import traceback

        _message("Cutout hit an unexpected error:\n\n" + traceback.format_exc()[-1500:])
        return 1


raise SystemExit(run())
