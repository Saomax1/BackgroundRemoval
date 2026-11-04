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

CPU and GPU in tandem: the parts of steps 2 and 3 that depend only on the photo
(``prepare_image``) run on the CPU while the GPU is still running the network, and
every per-pixel full-resolution step is split over all cores (``parallel.map_rows``).
Both are pure re-arrangements: the output is bit-for-bit what the plain
single-threaded code gives.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import parallel
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
    out = cv2.resize(alpha, (width, height), interpolation=interp)

    def clip(y0: int, y1: int) -> None:
        np.clip(out[y0:y1], 0.0, 1.0, out=out[y0:y1])

    parallel.map_rows(clip, height, width)
    return out


# --------------------------------------------------------------- guided filter
@dataclass
class GuideStats:
    """The half of the guided filter that depends only on the guide image.

    It can be computed before the matte exists (while the GPU runs the network).
    """

    guide: np.ndarray  # full-resolution float32 guide
    radius: int
    subsample: int
    g: np.ndarray  # guide on the (possibly subsampled) filter grid
    mean_g: np.ndarray
    var_g: np.ndarray


def _box(x: np.ndarray, r: int) -> np.ndarray:
    k = 2 * r + 1
    return cv2.boxFilter(x, cv2.CV_32F, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def guide_stats(guide: np.ndarray, radius: int, subsample: int = 1) -> GuideStats:
    h, w = guide.shape
    if subsample > 1:
        sw, sh = max(1, w // subsample), max(1, h // subsample)
        g = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
        r = max(1, radius // subsample)
    else:
        g, r = guide, radius
    mean_g = _box(g, r)
    var_g = _box(g * g, r) - mean_g * mean_g
    return GuideStats(guide=guide, radius=radius, subsample=subsample, g=g, mean_g=mean_g, var_g=var_g)


def _guided_coeffs(st: GuideStats, src: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """``mean_a``, ``mean_b`` at full resolution: the filter output is ``mean_a * guide + mean_b``."""
    h, w = st.guide.shape
    if st.subsample > 1:
        r = max(1, st.radius // st.subsample)
        p = cv2.resize(src, (st.g.shape[1], st.g.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        r, p = st.radius, src
    mean_p = _box(p, r)
    cov_gp = _box(st.g * p, r) - st.mean_g * mean_p
    a = cov_gp / (st.var_g + eps)
    b = mean_p - a * st.mean_g
    mean_a, mean_b = _box(a, r), _box(b, r)
    if st.subsample > 1:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return mean_a, mean_b


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float, subsample: int = 1) -> np.ndarray:
    """Fast guided filter (He & Sun 2015) with a single-channel float32 guide."""
    mean_a, mean_b = _guided_coeffs(guide_stats(guide, radius, subsample), src, eps)
    return mean_a * guide + mean_b


def _refine_geometry(long_side: int, opts: "RefineOptions") -> tuple[int, int]:
    radius = max(2, round(long_side * opts.radius_frac))
    subsample = max(1, min(radius // 2, long_side // 2048))
    return radius, subsample


def guided_refine(
    gray: np.ndarray,
    alpha: np.ndarray,
    opts: RefineOptions = RefineOptions(),
    reference_size: int | None = None,
    stats: GuideStats | None = None,
) -> np.ndarray:
    """``gray``: float32 HxW luminance in [0, 1] at full resolution, used as the guide.

    ``reference_size`` is the long side of the whole image when ``gray``/``alpha`` are a
    crop of it (the touch-up tools refine only the area they change), so the filter
    radius matches what the full-image refine would use. ``stats`` is the guide's
    precomputed half of the filter (``prepare_image``); it is used only if it matches.
    """
    h, w = alpha.shape
    long_side = reference_size or max(h, w)
    radius, subsample = _refine_geometry(long_side, opts)

    uncertain = np.empty((h, w), np.uint8)
    flags = parallel.map_rows(lambda y0, y1: _mark_uncertain(alpha, uncertain, y0, y1, opts), h, w)
    if not any(flags):
        return alpha

    st = stats
    if st is None or st.guide is not gray or st.radius != radius or st.subsample != subsample:
        st = guide_stats(gray, radius, subsample)
    mean_a, mean_b = _guided_coeffs(st, alpha, opts.eps)

    k = 2 * radius + 1
    band = cv2.dilate(uncertain, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    band = cv2.blur(band.astype(np.float32), (k, k))  # feather so there is no visible seam

    out = np.empty_like(alpha)

    def blend(y0: int, y1: int) -> None:
        filtered = mean_a[y0:y1] * gray[y0:y1] + mean_b[y0:y1]
        np.clip(filtered, 0.0, 1.0, out=filtered)
        a = alpha[y0:y1]
        np.clip(a + band[y0:y1] * (filtered - a), 0.0, 1.0, out=out[y0:y1])

    parallel.map_rows(blend, h, w)
    return out


def _mark_uncertain(alpha, out, y0, y1, opts) -> bool:
    a = alpha[y0:y1]
    np.logical_and(a > opts.band_low, a < opts.band_high, out=out[y0:y1].view(bool))
    return bool(out[y0:y1].any())


# ------------------------------------------------------------ colour clean-up
def _fg_geometry(h: int, w: int) -> tuple[int, int, int, int]:
    # The paper's radii (90, 6) are tuned for ~1 MP images; grow them with resolution, never shrink.
    scale = max(1.0, max(h, w) / 1024)
    r1, r2 = round(90 * scale), round(6 * scale)
    return r1, r2, max(1, r1 // 4), max(1, r2 // 4)


def _grid(x: np.ndarray, step: int) -> np.ndarray:
    if step == 1:
        return x
    h, w = x.shape[:2]
    return cv2.resize(x, (max(1, w // step), max(1, h // step)), interpolation=cv2.INTER_AREA)


def _to_size(x: np.ndarray, height: int, width: int) -> np.ndarray:
    if x.shape[:2] == (height, width):
        return x
    return cv2.resize(x, (width, height), interpolation=cv2.INTER_LINEAR)


@dataclass
class ForegroundPrep:
    """The photo on the two coarse grids Blur-Fusion works on (matte-independent)."""

    height: int
    width: int
    img1: np.ndarray
    img2: np.ndarray


def foreground_prep(image01: np.ndarray) -> ForegroundPrep:
    h, w = image01.shape[:2]
    _, _, s1, s2 = _fg_geometry(h, w)
    return ForegroundPrep(h, w, _grid(image01, s1), _grid(image01, s2))


def _smooth_fb(prep: ForegroundPrep, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blur-Fusion's smooth foreground / background estimates, at full resolution."""
    h, w = alpha.shape
    r1, r2, s1, s2 = _fg_geometry(h, w)
    # Pass 1: large radius, image used as both initial F and B.
    img1, a1 = prep.img1, _grid(alpha, s1)
    bf1, bb1 = _blurred_fb(img1, img1, a1, max(1, round(r1 / s1)))

    # Pass 2: small radius, seeded with pass-1 estimates, on a finer grid.
    img2, a2 = prep.img2, _grid(alpha, s2)
    bf1, bb1 = _to_size(bf1, *a2.shape), _to_size(bb1, *a2.shape)
    fg1 = _fuse(img2, a2[:, :, None], bf1, bb1)
    bf2, bb2 = _blurred_fb(fg1, bb1, a2, max(1, round(r2 / s2)))

    fg = _to_size(bf2, h, w)
    if fg is bf2:
        fg = fg.copy()
    return fg, _to_size(bb2, h, w)


def estimate_foreground(image01: np.ndarray, alpha: np.ndarray, prep: ForegroundPrep | None = None) -> np.ndarray:
    """Blur-Fusion foreground colour estimation. Returns float32 HxWx3 in [0, 1].

    Exact values are computed for the partially transparent pixels (the only
    ones the output uses); elsewhere the smooth estimate is returned. The box
    blurs are computed on grids downsampled by ~radius/4, which is visually
    identical and keeps a 24 MP photo at well under a second.
    """
    h, w = alpha.shape
    if prep is None or (prep.height, prep.width) != (h, w):
        prep = foreground_prep(image01)
    fg, bb = _smooth_fb(prep, alpha)

    def fuse(y0: int, y1: int) -> None:
        a = alpha[y0:y1]
        band = (a > 0.0) & (a < 1.0)
        if band.any():
            f = fg[y0:y1]
            f[band] = _fuse(image01[y0:y1][band], a[band][:, None], f[band], bb[y0:y1][band])

    parallel.map_rows(fuse, h, w)
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


# ------------------------------------------------------------------ pipeline
@dataclass
class ImagePrep:
    """Everything the refine and compose steps need from the photo alone.

    ``prepare_image`` builds it on the CPU while the GPU runs the network, so the
    full-resolution steps afterwards only do the matte-dependent half.
    """

    image: LoadedImage
    gray: np.ndarray
    guide: GuideStats | None
    foreground: ForegroundPrep | None

    def matches(self, image: LoadedImage) -> bool:
        return self.image is image


def _gray(image: LoadedImage) -> np.ndarray:
    g8 = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2GRAY)
    out = np.empty(g8.shape, np.float32)
    maxv = np.float32(image.max_value)

    def conv(y0: int, y1: int) -> None:
        np.divide(g8[y0:y1], maxv, out=out[y0:y1], dtype=np.float32)

    parallel.map_rows(conv, *g8.shape)
    return out


def image_as_float(image: LoadedImage) -> np.ndarray:
    """``image.as_float()`` computed on all cores."""
    out = np.empty(image.rgb.shape, np.float32)
    maxv = np.float32(image.max_value)

    def conv(y0: int, y1: int) -> None:
        np.divide(image.rgb[y0:y1], maxv, out=out[y0:y1], dtype=np.float32)

    parallel.map_rows(conv, image.height, image.width)
    return out


def prepare_image(image: LoadedImage, opts: RefineOptions = RefineOptions()) -> ImagePrep:
    """The matte-independent half of refine + compose (run it while the GPU is busy)."""
    gray = _gray(image) if opts.guided else None
    guide = None
    if gray is not None:
        guide = guide_stats(gray, *_refine_geometry(max(image.height, image.width), opts))
    fg = foreground_prep(image_as_float(image)) if opts.decontaminate else None
    return ImagePrep(image=image, gray=gray, guide=guide, foreground=fg)


def refine_alpha(
    image: LoadedImage,
    coarse_alpha: np.ndarray,
    opts: RefineOptions = RefineOptions(),
    prep: ImagePrep | None = None,
) -> np.ndarray:
    """Coarse network matte -> float32 matte at the exact input resolution."""
    alpha = upsample_alpha(coarse_alpha, image.height, image.width)
    if opts.guided:
        usable = prep is not None and prep.matches(image) and prep.gray is not None
        gray = prep.gray if usable else _gray(image)
        alpha = guided_refine(gray, alpha, opts, stats=prep.guide if usable else None)
    return alpha


def compose_rgba(
    image: LoadedImage,
    alpha: np.ndarray,
    decontaminate: bool = True,
    prep: ImagePrep | None = None,
) -> np.ndarray:
    """Build the output RGBA array, same dtype and size as the input.

    This is also what the editor calls after every brush stroke / click, with the
    user-edited alpha, so it must stay a pure function of (image, alpha).
    ``prep`` (from ``prepare_image`` for this image) only saves time.
    """
    h, w = image.height, image.width
    if alpha.shape != (h, w):
        raise ValueError(f"alpha {alpha.shape} does not match image {(h, w)}")
    maxv = image.max_value
    dtype = image.rgb.dtype
    out = np.empty((h, w, 4), dtype)
    alpha01 = np.empty((h, w), np.float32) if decontaminate else None

    def quantise(y0: int, y1: int) -> bool:
        a = np.clip(alpha[y0:y1], 0.0, 1.0)
        a *= maxv
        aq = np.round(a).astype(dtype)
        out[y0:y1, :, :3] = image.rgb[y0:y1]
        out[y0:y1, :, 3] = aq
        if alpha01 is not None:
            np.divide(aq, np.float32(maxv), out=alpha01[y0:y1], dtype=np.float32)
        return bool((aq < maxv).any())

    any_transparent = any(parallel.map_rows(quantise, h, w))
    if not (decontaminate and any_transparent):
        return out

    fprep = prep.foreground if prep is not None and prep.matches(image) else None
    if fprep is None:
        fprep = foreground_prep(image_as_float(image))
    fg, bb = _smooth_fb(fprep, alpha01)
    fmax = np.float32(maxv)

    def write(y0: int, y1: int) -> None:
        a = alpha01[y0:y1]
        rgb = image.rgb[y0:y1]
        f = fg[y0:y1]
        band = (a > 0.0) & (a < 1.0)
        if band.any():
            img01 = rgb[band].astype(np.float32) / fmax
            f[band] = _fuse(img01, a[band][:, None], f[band], bb[y0:y1][band])
        not_opaque = out[y0:y1, :, 3] < maxv
        q = np.round(f * maxv).astype(dtype)
        np.copyto(out[y0:y1, :, :3], q, where=not_opaque[:, :, None])

    parallel.map_rows(write, h, w)
    return out
