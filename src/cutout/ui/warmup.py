"""Loading bar for the startup model warm-up.

The engine warms the models in a worker thread (``CutoutEngine.warmup_all``) and reports
stages: ``plan``, then ``load``/``warm`` per model, ``paused`` when a cutout takes the GPU,
and ``done``. ONNX Runtime has no progress callback, so inside a stage the bar moves by
time against how long that stage took last time (kept in QSettings), and never passes the
end of the stage before the engine says it finished. The first start uses rough defaults.

The bar never blocks: the user can open an image at any time, and a cutout goes ahead
of the warm-up (see the engine).
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from ..core.models import REGISTRY

STAGES = ("load", "warm")
STEPS = 1000  # progress bar resolution


def default_seconds(model_id: str, stage: str) -> float:
    """First-start guesses (RTX 3070): loading ~4-5 s; the first run ~40 s at 2048 px, ~2 s at 1024."""
    if stage == "load":
        return 5.0
    spec = REGISTRY.get(model_id)
    return 40.0 if spec is not None and spec.input_size >= 2048 else 3.0


def subject_name(model_id: str) -> str:
    spec = REGISTRY.get(model_id)
    return spec.subject.value.capitalize() if spec is not None else model_id


class WarmupTracker(QObject):
    """Turns the engine's stage reports into a 0-1 fraction and a line of text (UI thread)."""

    # Emitted from the worker thread; Qt queues it onto the UI thread.
    stage_reported = Signal(str, str)
    changed = Signal()
    finished = Signal()

    def __init__(self, settings: QSettings | None = None, clock=time.monotonic, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.clock = clock
        self.plan: list[str] = []
        self.current: tuple[str, str] | None = None  # (model id, stage)
        self.started_at = 0.0
        self.done_stages: set[tuple[str, str]] = set()
        self.paused = False
        self.active = False
        self.stage_reported.connect(self.on_stage)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.changed)

    # -- called by the engine (any thread)
    def report(self, stage: str, model_id: str) -> None:
        self.stage_reported.emit(stage, model_id)

    # -- UI thread
    def expected(self, model_id: str, stage: str) -> float:
        if self.settings is not None:
            v = self.settings.value(f"warmup/{model_id}/{stage}")
            try:
                if v is not None and float(v) > 0:
                    return float(v)
            except (TypeError, ValueError):
                pass
        return default_seconds(model_id, stage)

    def _all_stages(self) -> list[tuple[str, str]]:
        return [(m, s) for m in self.plan for s in STAGES]

    def _finish_current(self, completed: bool) -> None:
        if self.current is None:
            return
        if completed and not self.paused and self.settings is not None:
            took = self.clock() - self.started_at
            self.settings.setValue(f"warmup/{self.current[0]}/{self.current[1]}", round(took, 2))
        if completed:
            self.done_stages.add(self.current)
        self.current = None

    @Slot(str, str)
    def on_stage(self, stage: str, model_id: str) -> None:
        if stage == "plan":
            self.plan = [m for m in model_id.split(",") if m]
            self.active = bool(self.plan)
            if self.active:
                self._timer.start()
        elif stage in STAGES:
            if self.current is not None and self.current != (model_id, stage):
                self._finish_current(completed=True)
            # Stages the engine skipped (model already loaded) count as done.
            for st in self._all_stages():
                if st == (model_id, stage):
                    break
                self.done_stages.add(st)
            self.current = (model_id, stage)
            self.started_at = self.clock()
            self.paused = False
        elif stage == "paused":
            self.paused = True
            self.current = None  # the run starts over when it resumes
        elif stage == "done":
            self._finish_current(completed=True)
            self.done_stages.update(self._all_stages())
            self.active = False
            self._timer.stop()
            self.changed.emit()
            self.finished.emit()
            return
        self.changed.emit()

    def fraction(self) -> float:
        stages = self._all_stages()
        if not stages:
            return 0.0
        total = sum(self.expected(*st) for st in stages)
        done = sum(self.expected(*st) for st in stages if st in self.done_stages)
        if self.current is not None and self.current not in self.done_stages:
            exp = self.expected(*self.current)
            # Approach the end of the stage but never claim it before the engine does.
            done += min(self.clock() - self.started_at, 0.95 * exp)
        return max(0.0, min(1.0, done / total))

    def text(self) -> str:
        if not self.active:
            return "AI models ready"
        if self.paused:
            return "Warm-up paused while your cutout runs"
        if self.current is None:
            return "Getting the AI models ready…"
        model_id, stage = self.current
        n = len(self.plan)
        i = self.plan.index(model_id) + 1 if model_id in self.plan else 1
        what = f"{subject_name(model_id)} model"
        count = f" ({i} of {n})" if n > 1 else ""
        return f"{'Loading' if stage == 'load' else 'Warming up'} the {what}{count}…"


class WarmupBar(QWidget):
    """A label over a thin progress bar, following a ``WarmupTracker``. Hides when done."""

    def __init__(self, tracker: WarmupTracker, compact: bool = False, show_when=None, parent=None):
        super().__init__(parent)
        self.tracker = tracker
        self.show_when = show_when or (lambda: True)
        self.label = QLabel()
        self.label.setObjectName("warmupLabel")
        self.bar = QProgressBar()
        self.bar.setObjectName("warmupBar")
        self.bar.setRange(0, STEPS)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6 if compact else 8)
        if compact:  # status bar: label and bar side by side
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 8, 0)
            self.bar.setFixedWidth(140)
            layout.addWidget(self.label)
            layout.addWidget(self.bar)
        else:  # start screen: a small panel
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.bar.setFixedWidth(360)
            hint = QLabel("You can open an image now; it goes first.")
            hint.setObjectName("emptyHint")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.label)
            layout.addWidget(self.bar, 0, Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(hint)
        tracker.changed.connect(self.refresh)
        self.refresh()

    @Slot()
    def refresh(self) -> None:
        self.setVisible(self.tracker.active and self.show_when())
        self.label.setText(self.tracker.text())
        self.bar.setValue(round(self.tracker.fraction() * STEPS))
