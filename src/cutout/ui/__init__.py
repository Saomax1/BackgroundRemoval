"""PySide6 desktop app: the cutout editor window.

Layout of this package:

* ``app.py``        QApplication setup, theme, ``main()`` entry point.
* ``main_window.py`` the editor window: open / drop, auto removal, save.
* ``document.py``   ``CutoutDocument``: the open image + its alpha matte + undo stack.
* ``canvas.py``     ``CutoutCanvas``: zoomable checkerboard preview that forwards
                    mouse input to the active tool in *image* coordinates.
* ``tools.py``      ``CanvasTool`` base class and helpers. Touch-up tools (brush,
                    click-to-select) plug in here; see ``MainWindow.register_tool``.
* ``workers.py``    run engine calls on ``QThreadPool`` so the UI never blocks.
"""


def main(argv: list[str] | None = None) -> int:
    """Start the app. ``argv`` may hold one image path to open (Explorer "Open with")."""
    from .app import main as _main

    return _main(argv)


__all__ = ["main"]
