"""Integer pixel rectangles: how the editing tools tell the UI which area changed."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    """``x, y`` top-left pixel, ``w, h`` size. Empty when ``w`` or ``h`` is 0."""

    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    @staticmethod
    def from_bounds(x0: float, y0: float, x1: float, y1: float) -> "Rect":
        """Smallest pixel rect covering the float box ``[x0, x1) x [y0, y1)``."""
        ix0, iy0 = math.floor(x0), math.floor(y0)
        ix1, iy1 = math.ceil(x1), math.ceil(y1)
        return Rect(ix0, iy0, max(0, ix1 - ix0), max(0, iy1 - iy0))

    @property
    def empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def union(self, other: "Rect") -> "Rect":
        if self.empty:
            return other
        if other.empty:
            return self
        x0, y0 = min(self.x, other.x), min(self.y, other.y)
        return Rect(x0, y0, max(self.x1, other.x1) - x0, max(self.y1, other.y1) - y0)

    def intersect(self, other: "Rect") -> "Rect":
        x0, y0 = max(self.x, other.x), max(self.y, other.y)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return Rect()
        return Rect(x0, y0, x1 - x0, y1 - y0)

    def clip(self, height: int, width: int) -> "Rect":
        return self.intersect(Rect(0, 0, width, height))

    def grow(self, n: int) -> "Rect":
        if self.empty:
            return self
        return Rect(self.x - n, self.y - n, self.w + 2 * n, self.h + 2 * n)

    @property
    def slices(self) -> tuple[slice, slice]:
        """``array[rect.slices]`` is the rect's pixels (rows, cols)."""
        return slice(self.y, self.y1), slice(self.x, self.x1)
