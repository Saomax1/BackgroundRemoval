import numpy as np

from cutout.core.imageio import LoadedImage
from cutout.core.refine import (
    RefineOptions,
    compose_rgba,
    guided_filter,
    refine_alpha,
    upsample_alpha,
)
from conftest import disc_alpha


def test_upsample_exact_size():
    a = upsample_alpha(disc_alpha(100, 100), 3001, 1999)
    assert a.shape == (3001, 1999) and a.dtype == np.float32
    assert 0.0 <= a.min() and a.max() <= 1.0


def test_guided_filter_identity_on_constant():
    g = np.full((50, 60), 0.3, np.float32)
    p = np.full((50, 60), 0.7, np.float32)
    np.testing.assert_allclose(guided_filter(g, p, 4, 1e-3), 0.7, atol=1e-4)
    np.testing.assert_allclose(guided_filter(g, p, 4, 1e-3, subsample=2), 0.7, atol=1e-4)


def test_guided_refine_snaps_to_image_edge():
    # Image has a hard vertical edge at x=60; coarse alpha has a blurry edge there.
    h, w = 64, 128
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :60] = 230
    coarse = np.clip((60 - np.arange(w)) / 16.0 + 0.5, 0, 1).astype(np.float32)
    coarse = np.tile(coarse, (h, 1))
    li = LoadedImage(rgb=img)
    refined = refine_alpha(li, coarse, RefineOptions(radius_frac=0.04, eps=1e-4))
    # The guided filter puts a hard step exactly on the image edge (x=59 -> 60),
    # where the coarse matte only had a gentle ramp.
    step = np.abs(np.diff(refined[h // 2]))
    assert int(np.argmax(step)) == 59
    assert step[59] > 5 * np.abs(np.diff(coarse[h // 2])).max()


def test_compose_keeps_opaque_pixels_bit_exact(rng):
    rgb = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    li = LoadedImage(rgb=rgb)
    alpha = disc_alpha(40, 40)
    rgba = compose_rgba(li, alpha)
    assert rgba.shape == (40, 40, 4) and rgba.dtype == np.uint8
    opaque = rgba[..., 3] == 255
    assert opaque.any()
    np.testing.assert_array_equal(rgba[opaque][:, :3], rgb[opaque])


def test_compose_16bit():
    rgb = np.full((10, 10, 3), 50000, np.uint16)
    rgba = compose_rgba(LoadedImage(rgb=rgb), np.full((10, 10), 0.5, np.float32))
    assert rgba.dtype == np.uint16
    assert rgba[0, 0, 3] == 32768


def test_decontamination_removes_background_tint():
    # Red subject on a green background; semi-transparent edge pixels are a red/green mix.
    h, w = 32, 96
    alpha = np.tile(np.clip((48 - np.arange(w)) / 8.0 + 0.5, 0, 1).astype(np.float32), (h, 1))
    fg = np.array([1.0, 0.0, 0.0], np.float32)
    bg = np.array([0.0, 1.0, 0.0], np.float32)
    mixed = alpha[..., None] * fg + (1 - alpha[..., None]) * bg
    rgb = np.round(mixed * 255).astype(np.uint8)
    rgba = compose_rgba(LoadedImage(rgb=rgb), alpha)
    edge = (rgba[..., 3] > 20) & (rgba[..., 3] < 235)
    assert edge.any()
    green_before = rgb[edge][:, 1].mean()
    green_after = rgba[edge][:, 1].mean()
    assert green_after < green_before * 0.3
