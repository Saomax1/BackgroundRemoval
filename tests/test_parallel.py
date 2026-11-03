"""CPU and GPU in tandem: strip-parallel full-resolution work must give the same bytes."""

import io
import zlib

import numpy as np
import pytest
from PIL import Image

from cutout.core import parallel, png
from cutout.core.engine import CutoutEngine
from cutout.core.imageio import LoadedImage, load_image, save_png
from cutout.core.models import PROMPT_REGISTRY, REGISTRY
from cutout.core.refine import compose_rgba, prepare_image, refine_alpha
from conftest import FakeSegmenter, disc_alpha


@pytest.fixture
def small_strips(monkeypatch):
    """Force many strips on small test images, so the parallel paths really run."""
    monkeypatch.setattr(parallel, "MIN_STRIP_PIXELS", 4096)
    monkeypatch.setattr(png, "_STRIP_PIXELS", 4096)


def _photo(rng, h=533, w=721, dtype=np.uint8):
    hi = 65535 if dtype == np.uint16 else 255
    return LoadedImage(rgb=rng.integers(0, hi + 1, (h, w, 3), dtype=np.int64).astype(dtype))


def test_strips_cover_every_row_once():
    for h, w in [(1, 1), (7, 3), (1000, 1000), (4000, 6000), (5, 10**7)]:
        parts = parallel.strips(h, w, n=8)
        assert parts[0][0] == 0 and parts[-1][1] == h
        assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
        assert all(y0 < y1 for y0, y1 in parts)
    assert len(parallel.strips(4000, 6000, n=4, max_pixels=100_000)) >= 200


def test_map_rows_keeps_order_and_nests(small_strips):
    def outer(y0, y1):
        inner = parallel.map_rows(lambda a, b: (a, b), 50, 1000)  # nested: runs serially
        return y0, y1, len(inner)

    out = parallel.map_rows(outer, 300, 1000)
    assert [o[0] for o in out] == sorted(o[0] for o in out)
    assert out[-1][1] == 300 and all(o[2] >= 1 for o in out)


def test_thread_count_env(monkeypatch):
    monkeypatch.setenv("CUTOUT_CPU_THREADS", "3")
    assert parallel.cpu_threads() == 3
    monkeypatch.setenv("CUTOUT_CPU_THREADS", "nonsense")
    assert parallel.cpu_threads() >= 1


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_parallel_refine_and_compose_match_single_thread(small_strips, monkeypatch, rng, dtype):
    img = _photo(rng, dtype=dtype)
    coarse = disc_alpha(256, 256)
    monkeypatch.setenv("CUTOUT_CPU_THREADS", "1")
    a1 = refine_alpha(img, coarse)
    o1 = compose_rgba(img, a1)
    monkeypatch.setenv("CUTOUT_CPU_THREADS", "4")
    a4 = refine_alpha(img, coarse)
    o4 = compose_rgba(img, a4)
    prep = prepare_image(img)
    ap = refine_alpha(img, coarse, prep=prep)
    op = compose_rgba(img, ap, prep=prep)
    assert np.array_equal(a1, a4) and np.array_equal(a1, ap)
    assert np.array_equal(o1, o4) and np.array_equal(o1, op)
    assert np.array_equal(compose_rgba(img, a1, False), compose_rgba(img, a1, False, prep))


def test_prep_for_another_image_is_ignored(rng):
    img, other = _photo(rng), _photo(rng)
    a = refine_alpha(img, disc_alpha(256, 256))
    assert np.array_equal(compose_rgba(img, a, prep=prepare_image(other)), compose_rgba(img, a))


def test_adler32_combine():
    a, b = b"cutout" * 999, bytes(range(256)) * 300
    assert png.adler32_combine(zlib.adler32(a), zlib.adler32(b), len(b)) == zlib.adler32(a + b)
    assert png.adler32_combine(1, zlib.adler32(b), len(b)) == zlib.adler32(b)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (3, 5000), (431, 257), (64, 64)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_png_round_trip_exact(small_strips, rng, shape, dtype):
    hi = 65535 if dtype == np.uint16 else 255
    rgba = rng.integers(0, hi + 1, shape + (4,), dtype=np.int64).astype(dtype)
    rgba[: shape[0] // 2] //= 7  # some smooth, compressible area too
    data = png.encode_png(rgba)
    with Image.open(io.BytesIO(data)) as im:
        im.load()  # Pillow checks every chunk CRC and the zlib stream
    if dtype == np.uint8:
        assert np.array_equal(np.asarray(Image.open(io.BytesIO(data))), rgba)
    import cv2

    back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert back.dtype == dtype
    assert np.array_equal(cv2.cvtColor(back, cv2.COLOR_BGRA2RGBA), rgba)


def test_png_compresses_like_pillow(rng):
    yy, xx = np.mgrid[0:1500, 0:1200]
    smooth = np.dstack([(xx // 5) % 256, (yy // 7) % 256, ((xx + yy) // 9) % 256, np.full_like(xx, 255)])
    rgba = np.clip(smooth + rng.integers(-3, 4, smooth.shape), 0, 255).astype(np.uint8)
    ours = len(png.encode_png(rgba))
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=6)
    assert ours <= len(buf.getvalue()) * 1.05


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_save_png_keeps_icc_on_both_depths(tmp_path, rng, dtype):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    hi = 65535 if dtype == np.uint16 else 255
    rgba = rng.integers(0, hi + 1, (40, 30, 4), dtype=np.int64).astype(dtype)
    out = tmp_path / "x.png"
    save_png(out, rgba, icc)
    with Image.open(out) as im:
        assert im.info.get("icc_profile") == icc
    img = load_image(out)
    assert img.rgb.dtype == dtype and np.array_equal(img.rgb, rgba[:, :, :3])


def test_engine_result_carries_prep_and_matches(tmp_path, rng):
    (tmp_path / REGISTRY["birefnet-general"].filename).write_bytes(b"x")
    engine = CutoutEngine(models_dir=tmp_path, segmenter_factory=FakeSegmenter)
    img = _photo(rng)
    result = engine.remove_background(img)
    assert result.prep is not None and result.prep.gray is None  # the big guide is not kept
    assert np.array_equal(result.alpha, refine_alpha(img, disc_alpha(256, 256)))
    assert np.array_equal(result.to_rgba(), compose_rgba(img, result.alpha))


def test_click_to_select_runs_on_cpu_by_default(tmp_path, monkeypatch):
    spec = PROMPT_REGISTRY["sam2.1-hiera-base-plus"]
    (tmp_path / spec.encoder_filename).write_bytes(b"x")
    (tmp_path / spec.decoder_filename).write_bytes(b"x")
    seen = []

    def factory(spec, enc, dec, device):
        seen.append(device)
        return object()

    monkeypatch.delenv("CUTOUT_SELECT_DEVICE", raising=False)
    CutoutEngine(models_dir=tmp_path, device="auto", prompt_segmenter_factory=factory).prompt_segmenter()
    CutoutEngine(models_dir=tmp_path, device="cuda", prompt_segmenter_factory=factory).prompt_segmenter()
    CutoutEngine(models_dir=tmp_path, device="cuda", select_device="cuda",
                 prompt_segmenter_factory=factory).prompt_segmenter()
    monkeypatch.setenv("CUTOUT_SELECT_DEVICE", "auto")
    CutoutEngine(models_dir=tmp_path, prompt_segmenter_factory=factory).prompt_segmenter()
    CutoutEngine(models_dir=tmp_path, device="cpu", prompt_segmenter_factory=factory).prompt_segmenter()
    assert seen == ["cpu", "cpu", "cuda", "auto", "cpu"]


def test_debug_report_lists_step_timings(tmp_path, rng):
    from cutout.core import debug

    (tmp_path / REGISTRY["birefnet-general"].filename).write_bytes(b"x")
    CutoutEngine(models_dir=tmp_path, segmenter_factory=FakeSegmenter).remove_background(_photo(rng))
    last = debug.recent_timings()[-1]
    assert "cutout 721x533" in last and "inference" in last and "cpu_prep_alongside_model" in last
    report = debug.build_report(models_dir=tmp_path)
    assert "Timings (most recent last)" in report and "threads for full-size steps" in report
