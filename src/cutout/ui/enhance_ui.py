"""The Enhance dialog: pick how much bigger (1x clean up, 2x, 4x) and whether to fix faces.

The window does the work (``MainWindow.start_enhance``); this only asks. Choices are
remembered in QSettings (``enhance/scale``, ``enhance/fix_faces``).
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..core.enhance import MAX_OUTPUT_PIXELS, SCALES

ENHANCE_TIP = "Make a small or blurry picture bigger and sharper before removing the background"

_LABELS = {
    1: "Clean up only, same size",
    2: "2x bigger",
    4: "4x bigger",
}


def scale_fits(width: int, height: int, scale: int) -> bool:
    return width * scale * height * scale <= MAX_OUTPUT_PIXELS


class EnhanceDialog(QDialog):
    def __init__(self, parent: QWidget | None, width: int, height: int, settings: QSettings,
                 is_photo: bool, faces_available: bool):
        super().__init__(parent)
        self.setWindowTitle("Enhance picture")
        self.setModal(True)
        self.settings = settings
        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        intro = QLabel("Makes the picture sharper and cleaner (blur, noise, JPEG blocks) and, if you like, "
                       "bigger. The background is removed from the enhanced picture afterwards.")
        intro.setWordWrap(True)
        intro.setMinimumWidth(380)
        lay.addWidget(intro)

        self.group = QButtonGroup(self)
        self.buttons: dict[int, QRadioButton] = {}
        wanted = int(settings.value("enhance/scale", 2, type=int))
        fallback = None
        for s in SCALES:
            fits = scale_fits(width, height, s)
            b = QRadioButton(f"{_LABELS[s]}   ({width * s} × {height * s})")
            b.setEnabled(fits)
            if not fits:
                b.setToolTip(f"Too big: Enhance makes pictures up to {MAX_OUTPUT_PIXELS // 1_000_000} megapixels")
            else:
                fallback = s
            self.group.addButton(b, s)
            self.buttons[s] = b
            lay.addWidget(b)
        pick = wanted if wanted in self.buttons and self.buttons[wanted].isEnabled() else fallback
        if pick is not None:
            self.buttons[pick].setChecked(True)

        self.fix_faces = QCheckBox("Fix faces (repairs blurry or blocky faces in photos)")
        self.fix_faces.setChecked(bool(settings.value("enhance/fix_faces", True, type=bool)))
        if not is_photo:
            self.fix_faces.setEnabled(False)
            self.fix_faces.setChecked(False)
            self.fix_faces.setToolTip("Only for photos of people: switch the subject to Person to use it")
        elif not faces_available:
            self.fix_faces.setEnabled(False)
            self.fix_faces.setChecked(False)
            self.fix_faces.setToolTip("The face repair model is not installed")
        lay.addWidget(self.fix_faces)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.ok = buttons.addButton("Enhance", QDialogButtonBox.ButtonRole.AcceptRole)
        self.ok.setObjectName("primaryButton")
        self.ok.setCursor(Qt.CursorShape.PointingHandCursor)
        self.ok.setEnabled(pick is not None)
        self.ok.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    @property
    def scale(self) -> int:
        return max(1, self.group.checkedId())

    @property
    def wants_faces(self) -> bool:
        return self.fix_faces.isEnabled() and self.fix_faces.isChecked()

    def accept(self) -> None:
        self.settings.setValue("enhance/scale", self.scale)
        if self.fix_faces.isEnabled():
            self.settings.setValue("enhance/fix_faces", self.fix_faces.isChecked())
        super().accept()
