"""Brush settings and the soft round stamp.

The brush works on the full-resolution matte in image pixels. A stroke is a
chain of stamps ("dabs") spaced along the pointer path. Each dab writes into a
per-stroke *coverage* buffer with ``max``, and the matte is then blended from
its pre-stroke value towards the target by that coverage. So going over the
same spot twice in one stroke never builds up hard edges, and opacity means
what it says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .rect import Rect


class BrushMode(str, Enum):
    KEEP = "keep"  # paint the subject back in (alpha -> 1)
    ERASE = "erase"  # paint background away (alpha -> 0)
    REFINE = "refine"  # re-fit the edge to the photo (hair, fur) with the guided filter


@dataclass(frozen=True)
class Brush:
    mode: BrushMode = BrushMode.KEEP
    radius: float = 40.0  # image pixels (the UI converts from screen size via its zoom)
    hardness: float = 0.6  # 0 = all feather, 1 = hard edge (still 1 px anti-aliased)
    opacity: float = 1.0  # max effect of one stroke
    spacing: float = 0.15  # dab distance as a fraction of the radius
    pressure_size: bool = False  # pen pressure scales the radius
    pressure_opacity: bool = True  # pen pressure scales the opacity

    def __post_init__(self):
        if not self.radius > 0:
            raise ValueError("brush radius must be > 0")
        if not 0.0 <= self.hardness <= 1.0 or not 0.0 <= self.opacity <= 1.0:
            raise ValueError("hardness and opacity must be in [0, 1]")
        if not self.spacing > 0:
            raise ValueError("spacing must be > 0")

    def dab_radius(self, pressure: float) -> float:
        return max(0.5, self.radius * (pressure if self.pressure_size else 1.0))

    def dab_flow(self, pressure: float) -> float:
        return self.opacity * (pressure if self.pressure_opacity else 1.0)


def dab_rect(cx: float, cy: float, radius: float) -> Rect:
    return Rect.from_bounds(cx - radius, cy - radius, cx + radius, cy + radius)


def stamp(cx: float, cy: float, radius: float, hardness: float, rect: Rect) -> np.ndarray:
    """Weights in [0, 1] of a round dab centred at ``(cx, cy)`` over ``rect`` (float32, rect.h x rect.w).

    Distance is measured to pixel centres, so sub-pixel pointer positions are honoured.
    The falloff ring is at least one pixel wide, which anti-aliases hard brushes.
    """
    xs = np.arange(rect.x, rect.x1, dtype=np.float32) + np.float32(0.5 - cx)
    ys = np.arange(rect.y, rect.y1, dtype=np.float32) + np.float32(0.5 - cy)
    d = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    falloff = max(radius * (1.0 - hardness), 1.0)
    t = np.clip((np.float32(radius) - d) / np.float32(falloff), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)  # smoothstep


def dab_centres(x0: float, y0: float, x1: float, y1: float, step: float, carry: float) -> tuple[list[tuple[float, float, float]], float]:
    """Dab positions along a segment, evenly ``step`` apart, continuing from the previous segment.

    ``carry`` is the distance already travelled since the last dab. Returns the dabs as
    ``(x, y, t)`` with ``t`` in (0, 1] along the segment, and the new carry.
    """
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return [], carry
    out = []
    d = step - carry
    while d <= length:
        t = d / length
        out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, t))
        d += step
    return out, length - (d - step)
