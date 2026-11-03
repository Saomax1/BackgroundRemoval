"""Full-resolution matte refinement and RGBA composition.

Networks run at 1024-2048 px, but the output must match the input pixel for
pixel (often 4000-8000 px). Three steps get us there without losing detail:

1. ``upsample_alpha``: stretch the coarse matte to the exact input size.
2. ``guided_refine``: a guided filter, steered by the full-res photo, snaps the
   blurry upsampled edge back onto real image edges (hair strands, fur, lace).
   It is applied only in the uncertain band around the edge, so solid regions
   of the subject and the background are never touched.
3. ``estimate_foreground``: removes background colour bleeding into
   semi-transparent pixels (the "halo" you see when hair is placed on a new
   background), using Blur-Fusion (Forte & Pitie, 2021).

``compose_rgba`` then builds the final image. Fully opaque pixels are copied
bit-for-bit from the input; only partially transparent pixels get new colours.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imageio import LoadedImage


@dataclass(frozen=True)
class RefineOptions:
    guided: bool = True
    # Guided-filter radius as a fraction of the long side (0.0025 -> 10 px on a 4000 px photo).
    radius_frac: float = 0.0025
    eps: float = 1e-3
    # Alpha values strictly between these are "uncertain" and get refined.
    band_low: float = 0.02
    band_high: float = 0.98
    decontaminate: bool = True


def upsample_alpha(alpha: np.ndarray, height: int, width: int) -> np.ndarray:
    alpha = np.asarray(alpha, np.float32)
    if alpha.shape == (height, width):
        return np.clip(alpha, 0.0, 1.0)
    growing = height * width > alpha.shape[0] * alpha.shape[1]
    interp = cv2.INTER_CUBIC if growing else cv2.INTER_AREA
    return np.clip(cv2.resize(alpha, (width, height), interpolation=interp), 0.0, 1.0)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float, subsample: int = 1) -> np.ndarray:
    """Fast guided filter (He & Sun 2015) with a single-channel float32 guide."""
    h, w = guide.shape
    if subsample > 1:
        sw, sh = max(1, w // subsample), max(1, h // subsample)
        g = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
        p = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        r = max(1, radius // subsample)
    else:
        g, p, r = guide, src, radius

    ksize = (2 * r + 1, 2 * r + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    mean_g, mean_p = box(g), box(p)
    var_g = box(g * g) - mean_g * mean_g
    cov_gp = box(g * p) - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a, mean_b = box(a), box(b)
    if subsample > 1:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return mean_a * guide + mean_b


def guided_refine(gray: np.ndarray, alpha: np.ndarray, opts: RefineOptions = RefineOptions()) -> np.ndarray:
    """``gray``: float32 HxW luminance in [0, 1] at full resolution, used as the guide."""
    h, w = alpha.shape
    long_side = max(h, w)
    radius = max(2, round(long_side * opts.radius_frac))
    subsample = max(1, min(radius // 2, long_side // 2048))

    filtered = np.clip(guided_filter(gray, alpha, radius, opts.eps, subsample), 0.0, 1.0)

    uncertain = ((alpha > opts.band_low) & (alpha < opts.band_high)).astype(np.uint8)
    if not uncertain.any():
        return alpha
    k = 2 * radius + 1
    band = cv2.dilate(uncertain, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    band = cv2.blur(band.astype(np.float32), (k, k))  # feather so there is no visible seam
    return np.clip(alpha + band * (filtered - alpha), 0.0, 1.0)


def estimate_foreground(image01: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blur-Fusion foreground colour estimation. Returns float32 HxWx3 in [0, 1].

    Exact values are computed for the partially transparent pixels (the only
    ones the output uses); elsewhere the smooth estimate is returned. The box
    blurs are computed on grids downsampled by ~radius/4, which is visually
    identical and keeps a 24 MP photo at well under a second.
    """
    h, w = alpha.shape
    # The paper's radii (90, 6) are tuned for ~1 MP images; grow them with resolution, never shrink.
    scale = max(1.0, max(h, w) / 1024)
    r1, r2 = round(90 * scale), round(6 * scale)
    s1, s2 = max(1, r1 // 4), max(1, r2 // 4)

    def grid(x: np.ndarray, step: int) -> np.ndarray:
        if step == 1:
            return x
        return cv2.resize(x, (max(1, w // step), max(1, h // step)), interpolation=cv2.INTER_AREA)

    def to_size(x: np.ndarray, like: np.ndarray) -> np.ndarray:
        size = (like.shape[1], like.shape[0])
        if (x.shape[1], x.shape[0]) == size:
            return x
        return cv2.resize(x, size, interpolation=cv2.INTER_LINEAR)

    # Pass 1: large radius, image used as both initial F and B.
    img1, a1 = grid(image01, s1), grid(alpha, s1)
    bf1, bb1 = _blurred_fb(img1, img1, a1, max(1, round(r1 / s1)))

    # Pass 2: small radius, seeded with pass-1 estimates, on a finer grid.
    img2, a2 = grid(image01, s2), grid(alpha, s2)
    bf1, bb1 = to_size(bf1, a2), to_size(bb1, a2)
    fg1 = _fuse(img2, a2[:, :, None], bf1, bb1)
    bf2, bb2 = _blurred_fb(fg1, bb1, a2, max(1, round(r2 / s2)))

    fg = to_size(bf2, alpha)
    if fg is bf2:
        fg = fg.copy()
    bb2 = to_size(bb2, alpha)
    ys, xs = np.nonzero((alpha > 0.0) & (alpha < 1.0))
    if ys.size:
        fg[ys, xs] = _fuse(image01[ys, xs], alpha[ys, xs][:, None], fg[ys, xs], bb2[ys, xs])
    return fg


def _blurred_fb(fg, bg, a, k):
    a3 = a[:, :, None]
    blurred_a = cv2.blur(a, (k, k))[:, :, None]
    blurred_fg = cv2.blur(fg * a3, (k, k)) / (blurred_a + 1e-5)
    blurred_bg = cv2.blur(bg * (1.0 - a3), (k, k)) / ((1.0 - blurred_a) + 1e-5)
    return blurred_fg, blurred_bg


def _fuse(image, a, blurred_fg, blurred_bg):
    fg = blurred_fg + a * (image - a * blurred_fg - (1.0 - a) * blurred_bg)
    return np.clip(fg, 0.0, 1.0, out=fg)


def refine_alpha(image: LoadedImage, coarse_alpha: np.ndarray, opts: RefineOptions = RefineOptions()) -> np.ndarray:
    """Coarse network matte -> float32 matte at the exact input resolution."""
    alpha = upsample_alpha(coarse_alpha, image.height, image.width)
    if opts.guided:
        gray = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / np.float32(image.max_value)
        alpha = guided_refine(gray, alpha, opts)
    return alpha


def compose_rgba(image: LoadedImage, alpha: np.ndarray, decontaminate: bool = True) -> np.ndarray:
    """Build the output RGBA array, same dtype and size as the input.

    This is also what the editor calls after every brush stroke / click, with the
    user-edited alpha, so it must stay a pure function of (image, alpha).
    """
    if alpha.shape != (image.height, image.width):
        raise ValueError(f"alpha {alpha.shape} does not match image {(image.height, image.width)}")
    maxv = image.max_value
    alpha_q = np.round(np.clip(alpha, 0.0, 1.0) * maxv).astype(image.rgb.dtype)

    rgb = image.rgb.copy()
    not_opaque = alpha_q < maxv
    if decontaminate and not_opaque.any():
        fg = estimate_foreground(image.as_float(), alpha_q.astype(np.float32) / maxv)
        rgb[not_opaque] = np.round(fg[not_opaque] * maxv).astype(rgb.dtype)
    return np.dstack([rgb, alpha_q])
