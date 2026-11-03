"""Touch-up tools on the full-resolution alpha matte. Pure NumPy/OpenCV, no Qt.

* ``MatteEditor``: keep / erase / refine-edge brush, merging selections, tile-based undo.
* ``ClickSelector``: click-to-select with a promptable model (SAM 2.1), refined at full res.
"""

from .brush import Brush, BrushMode
from .matte import EditHistory, MaskOp, MatteEditor, Patch
from .rect import Rect
from .select import Click, ClickSelector, Selection, SelectOptions

__all__ = [
    "Brush",
    "BrushMode",
    "Click",
    "ClickSelector",
    "EditHistory",
    "MaskOp",
    "MatteEditor",
    "Patch",
    "Rect",
    "SelectOptions",
    "Selection",
]
