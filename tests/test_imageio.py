import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from cutout.core.imageio import _apply_orientation, load_image, save_png


def test_8bit_roundtrip_exact(tmp_path, rng):
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    src = tmp_path / "in.png"
    Image.fromarray(rgb).save(src)
    img = load_image(src)
    assert img.rgb.dtype == np.uint8 and img.rgb.shape == (37, 53, 3)
    np.testing.assert_array_equal(img.rgb, rgb)

    alpha = rng.integers(0, 256, (37, 53), dtype=np.uint8)
    out = tmp_path / "out.png"
    save_png(out, np.dstack([rgb, alpha]))
    back = np.asarray(Image.open(out))
    assert back.shape == (37, 53, 4)
    np.testing.assert_array_equal(back[..., :3], rgb)
    np.testing.assert_array_equal(back[..., 3], alpha)


def test_16bit_png_kept_16bit(tmp_path, rng):
    rgb16 = rng.integers(0, 65536, (20, 30, 3), dtype=np.uint16)
    src = tmp_path / "deep.png"
    src.write_bytes(cv2.imencode(".png", cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))[1].tobytes())
    img = load_image(src)
    assert img.rgb.dtype == np.uint16
    np.testing.assert_array_equal(img.rgb, rgb16)
    assert img.as_uint8().dtype == np.uint8

    out = tmp_path / "deep_out.png"
    rgba = np.dstack([rgb16, np.full((20, 30), 40000, np.uint16)])
    save_png(out, rgba)
    back = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16
    np.testing.assert_array_equal(cv2.cvtColor(back, cv2.COLOR_BGRA2RGBA), rgba)


def test_exif_orientation_applied(tmp_path, rng):
    rgb = rng.integers(0, 256, (10, 16, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    src = tmp_path / "rot.jpg"
    Image.fromarray(rgb).save(src, quality=100, exif=exif)
    img = load_image(src)
    assert img.rgb.shape == (16, 10, 3)  # what the user sees, not the raw sensor grid


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orientation_helper_matches_pillow(orientation, rng):
    rgb = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, "PNG", exif=exif)
    expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(buf.getvalue()))))
    np.testing.assert_array_equal(_apply_orientation(rgb, orientation), expected)


def test_unicode_path_and_icc(tmp_path, rng):
    rgb = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    src = tmp_path / "фото_写真.jpg"
    Image.fromarray(rgb).save(src, icc_profile=icc, quality=95)
    img = load_image(src)
    assert img.icc_profile == icc
    out = tmp_path / "出力.png"
    save_png(out, np.dstack([img.rgb, np.full((8, 8), 255, np.uint8)]), img.icc_profile)
    assert Image.open(out).info.get("icc_profile") == icc


def test_palette_and_grayscale_inputs(tmp_path):
    gray = tmp_path / "g.png"
    Image.new("L", (9, 4), 128).save(gray)
    assert load_image(gray).rgb.shape == (4, 9, 3)
    pal = tmp_path / "p.gif"
    Image.new("P", (6, 3)).save(pal)
    assert load_image(pal).rgb.shape == (3, 6, 3)
