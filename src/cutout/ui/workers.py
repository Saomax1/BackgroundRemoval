"""Run blocking work (model inference, PNG encoding) off the UI thread.

    task = Task(engine.remove_background, path, Subject.PERSON)
    task.signals.finished.connect(on_done)   # delivered on the UI thread
    task.signals.failed.connect(on_error)
    QThreadPool.globalInstance().start(task)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal

log = logging.getLogger(__name__)


class TaskSignals(QObject):
    finished = Signal(object)  # the function's return value
    failed = Signal(object)  # the exception instance


class Task(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()
        # Keep the signals object alive with the runnable; Qt deletes the runnable itself.
        self.setAutoDelete(True)

    def run(self) -> None:  # runs on a pool thread
        try:
            value = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:  # noqa: BLE001 - report every failure to the UI
            log.error("background task failed:\n%s", traceback.format_exc())
            self.signals.failed.emit(exc)
        else:
            self.signals.finished.emit(value)
