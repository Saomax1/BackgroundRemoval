"""Enhance: tiled upscaling, face repair, the engine's GPU turn and the CLI (no real models)."""

import dataclasses

import numpy as np
import pytest
from PIL import Image

from cutout.core import enhance
from cutout.core.engine import CutoutEngine
from cutout.core.enhance import EnhanceTooLarge, pick_scale, restore_faces, tiles, upscale_image
from cutout.core.head import Face
from cutout.core.imageio import LoadedImage, load_image, save_rgb_png
from cutout.core.models import (
    ENHANCE_REGISTRY,
    ModelNotInstalled,
    Subject,
    enhance_model_path,
    resolve_face_model,
    resolve_upscaler,
)


class BlurUpscaler:
    """A 4x "network" with a 5x5 receptive field: blur, then repeat each pixel 4x4.

    Its output depends on neighbours, so tiling is only exact if each tile sees enough of them.
    """

    name = "fake-up"
    scale = 4

    def __init__(self, *args, **kwargs):
        self.calls = 0
        self.device = args[2] if len(args) > 2 else "cpu"

    def upscale(self, tile):
        assert tile.dtype == np.float32 and tile.max() <= 1.0
        self.calls += 1
        h, w = tile.shape[:2]
        p = np.pad(tile, ((2, 2), (2, 2), (0, 0)), mode="edge")
        blur = sum(p[dy:dy + h, dx:dx + w] for dy in range(5) for dx in range(5)) / 25.0
        return np.repeat(np.repeat(blur, 4, axis=0), 4, axis=1).astype(np.float32)


class WhiteFace:
    name = "fake-face"
    face_size = 512

    def __init__(self, *args, **kwargs):
        pass

    def restore(self, face):
        assert face.shape == (512, 512, 3)
        return np.ones_like(face)


def _img(rng, h, w, dtype=np.uint8):
    hi = 65536 if dtype == np.uint16 else 256
    return rng.integers(0, hi, (h, w, 3), dtype=dtype)


# --------------------------------------------------------------------- tiling


def test_tiles_cover_every_pixel_once():
    for h, w in ((1, 1), (255, 257), (600, 401), (1024, 768)):
        seen = np.zeros((h, w), np.int32)
        for y0, y1, x0, x1 in tiles(h, w, 256):
            assert 0 < y1 - y0 <= 256 and 0 < x1 - x0 <= 256
            seen[y0:y1, x0:x1] += 1
        assert (seen == 1).all()


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_tiled_upscale_matches_one_big_run(rng, scale):
    rgb = _img(rng, 97, 131)
    whole = upscale_image(rgb, BlurUpscaler(), scale, tile=1000, pad=0)
    tiled = upscale_image(rgb, BlurUpscaler(), scale, tile=32, pad=2)  # pad >= receptive radius
    assert tiled.shape == (97 * scale, 131 * scale, 3) and tiled.dtype == np.uint8
    np.testing.assert_array_equal(tiled, whole)


def test_scale_1_is_the_4x_result_averaged_back():
    flat = np.full((40, 50, 3), 123, np.uint8)
    out = upscale_image(flat, BlurUpscaler(), 1, tile=16, pad=2)
    assert out.shape == flat.shape
    np.testing.assert_array_equal(out, flat)


def test_16_bit_stays_16_bit(rng):
    rgb = _img(rng, 30, 20, np.uint16)
    out = upscale_image(rgb, BlurUpscaler(), 2, tile=16, pad=2)
    assert out.dtype == np.uint16 and out.shape == (60, 40, 3)
    assert out.max() > 255  # really 16-bit values, not squashed to 8 bits


def test_progress_is_reported_per_tile(rng):
    seen = []
    upscale_image(_img(rng, 70, 70), BlurUpscaler(), 2, tile=32, progress=lambda d, n: seen.append((d, n)))
    assert seen == [(i, 9) for i in range(1, 10)]


def test_pick_scale_lowers_to_fit_and_refuses_huge():
    assert pick_scale(1000, 800, 4) == 4
    assert pick_scale(2000, 2000, 4) == 2  # 4x would be 64 MP
    assert pick_scale(4000, 4000, 2) == 1
    with pytest.raises(EnhanceTooLarge):
        pick_scale(7000, 7000, 2)
    with pytest.raises(ValueError):
        pick_scale(10, 10, 3)


# ---------------------------------------------------------------- face repair


def _face_at(cx, cy, eyes):
    """A face whose landmarks sit like the template's, centred at (cx, cy), eyes ``eyes`` apart."""
    t = enhance.FACE_TEMPLATE_512
    k = eyes / np.linalg.norm(t[1] - t[0])
    pts = (t - t.mean(0)) * k + (cx, cy)
    return Face(cx - eyes, cy - eyes, 2 * eyes, 2 * eyes, 0.9, 0.0, tuple(map(tuple, pts.tolist())))


def test_restore_faces_changes_only_the_face():
    rgb = np.full((600, 800, 3), 100, np.uint8)
    fixes = restore_faces(rgb, [_face_at(400, 300, 60)], WhiteFace(), strength=1.0)
    assert len(fixes) == 1
    x0, y0, x1, y1 = fixes[0].box
    assert x0 < 400 < x1 and y0 < 300 < y1
    assert rgb[300, 400].min() > 240  # the middle of the face is the repaired (white) face
    assert (rgb[:, : x0] == 100).all() and (rgb[: y0] == 100).all()  # outside untouched
    assert (rgb[y1:] == 100).all() and (rgb[:, x1:] == 100).all()


def test_restore_faces_strength_mixes():
    rgb = np.full((600, 800, 3), 100, np.uint8)
    restore_faces(rgb, [_face_at(400, 300, 60)], WhiteFace(), strength=0.5)
    assert abs(int(rgb[300, 400, 0]) - round(100 + 0.5 * 155)) <= 2


def test_restore_faces_skips_big_faces_and_drawn_boxes():
    rgb = np.full((2000, 2000, 3), 100, np.uint8)
    big = _face_at(1000, 1000, 400)  # far bigger than the model's 512 face: would get softer
    drawn = Face.from_box(10, 10, 200, 200)  # no landmarks
    assert restore_faces(rgb, [big, drawn], WhiteFace()) == []
    assert (rgb == 100).all()


# --------------------------------------------------------------------- engine


def _models(tmp_path, *ids):
    d = tmp_path / "models"
    d.mkdir(exist_ok=True)
    for mid in ids:
        enhance_model_path(ENHANCE_REGISTRY[mid], d).write_bytes(b"x")
    return d


def test_resolve_prefers_the_subjects_upscaler(tmp_path):
    d = _models(tmp_path, "realesrgan-x4plus", "realesrgan-x4plus-anime-6b")
    assert resolve_upscaler(Subject.PERSON, d).id == "realesrgan-x4plus"
    assert resolve_upscaler(Subject.ANIME, d).id == "realesrgan-x4plus-anime-6b"
    (tmp_path / "x").mkdir()
    d2 = _models(tmp_path / "x", "realesrgan-x4plus")
    assert resolve_upscaler(Subject.ANIME, d2).id == "realesrgan-x4plus"  # any upscaler beats none
    with pytest.raises(ModelNotInstalled):
        resolve_face_model(d)


def test_engine_enhance_keeps_depth_and_profile(tmp_path, rng):
    d = _models(tmp_path, "realesrgan-x4plus")
    engine = CutoutEngine(models_dir=d, device="cpu", upscaler_factory=BlurUpscaler)
    img = LoadedImage(_img(rng, 50, 40, np.uint16), icc_profile=b"icc", source_path=tmp_path / "a.png")
    res = engine.enhance(img, Subject.PERSON, 2)
    assert res.image.rgb.shape == (100, 80, 3) and res.image.rgb.dtype == np.uint16
    assert res.image.icc_profile == b"icc" and res.image.source_path == img.source_path
    assert res.scale == 2 and res.upscaler_id == "realesrgan-x4plus"
    assert res.face_model_id is None and res.notice is None
    assert not engine.can_fix_faces() and engine.can_enhance(Subject.ANIME)


def test_engine_enhance_fixes_faces_on_photos_only(tmp_path, rng, monkeypatch):
    d = _models(tmp_path, "realesrgan-x4plus", "gfpgan-1.4")
    engine = CutoutEngine(models_dir=d, device="cpu", upscaler_factory=BlurUpscaler,
                          face_restorer_factory=WhiteFace)
    found = []
    monkeypatch.setattr(engine.face_finder, "detect",
                        lambda rgb, thr: found.append(rgb.shape) or [_face_at(200, 150, 40)])
    img = LoadedImage(np.full((150, 200, 3), 90, np.uint8))
    res = engine.enhance(img, Subject.PERSON, 2)
    assert res.faces_fixed == 1 and res.face_model_id == "gfpgan-1.4"
    assert found == [(300, 400, 3)]  # faces are found on the enhanced picture
    assert res.image.rgb[150, 200].min() > 200
    res = engine.enhance(img, Subject.ANIME, 2)
    assert res.face_model_id is None and res.faces_fixed == 0  # GFPGAN is for real faces


def test_engine_enhance_lowers_scale_with_a_notice(tmp_path, monkeypatch):
    d = _models(tmp_path, "realesrgan-x4plus")
    monkeypatch.setattr(enhance, "MAX_OUTPUT_PIXELS", 100 * 100 * 4)
    monkeypatch.setattr("cutout.core.engine.pick_scale",
                        lambda w, h, s: pick_scale(w, h, s, max_pixels=100 * 100 * 4))
    engine = CutoutEngine(models_dir=d, device="cpu", upscaler_factory=BlurUpscaler)
    res = engine.enhance(LoadedImage(np.zeros((100, 100, 3), np.uint8)), Subject.PERSON, 4)
    assert res.scale == 2 and res.requested_scale == 4
    assert "2x instead of 4x" in res.notice


def test_engine_enhance_falls_back_to_cpu_after_oom(tmp_path):
    d = _models(tmp_path, "realesrgan-x4plus")
    devices = []

    def factory(spec, path, device):
        devices.append(device)
        up = BlurUpscaler()
        if device != "cpu":
            def boom(tile):
                raise RuntimeError("BFCArena: Failed to allocate memory")
            up.upscale = boom
        return up

    engine = CutoutEngine(models_dir=d, device="auto", upscaler_factory=factory)
    res = engine.enhance(LoadedImage(np.zeros((20, 20, 3), np.uint8)), Subject.PERSON, 2)
    assert devices == ["auto", "auto", "cpu"]
    assert res.notice and "processor" in res.notice
    assert res.image.rgb.shape == (40, 40, 3)


def test_engine_enhance_without_upscaler(tmp_path):
    engine = CutoutEngine(models_dir=_models(tmp_path), device="cpu")
    assert not engine.can_enhance()
    with pytest.raises(ModelNotInstalled):
        engine.enhance(LoadedImage(np.zeros((8, 8, 3), np.uint8)))


# ------------------------------------------------------------------- PNG + CLI


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_rgb_png_round_trip(tmp_path, rng, dtype):
    rgb = _img(rng, 33, 21, dtype)
    p = tmp_path / "e.png"
    save_rgb_png(p, rgb, None)
    back = load_image(p)
    np.testing.assert_array_equal(back.rgb, rgb)


def test_cli_enhance_only(tmp_path, rng, monkeypatch):
    from cutout import cli

    d = _models(tmp_path, "realesrgan-x4plus")
    monkeypatch.setattr(CutoutEngine, "__init__", _engine_init_with(BlurUpscaler))
    src = tmp_path / "small.png"
    Image.fromarray(_img(rng, 30, 40)).save(src)
    assert cli.main([str(src), "--enhance-only", "--enhance", "4", "--models-dir", str(d), "--device", "cpu"]) == 0
    out = tmp_path / "small_enhanced_4x.png"
    with Image.open(out) as im:
        assert im.size == (160, 120) and im.mode == "RGB"


def _engine_init_with(factory):
    original = CutoutEngine.__init__

    def init(self, *args, **kwargs):
        kwargs.setdefault("upscaler_factory", factory)
        original(self, *args, **kwargs)

    return init


def test_registry_entries_are_consistent():
    for spec in ENHANCE_REGISTRY.values():
        assert spec.filename.endswith(".onnx")
        assert spec.urls or spec.export_source
        if spec.role.value == "upscale":
            assert spec.scale == 4 and spec.export_source.endswith(".pth")
        else:
            assert spec.face_size == 512
    assert dataclasses.is_dataclass(next(iter(ENHANCE_REGISTRY.values())))


# ------------------------------------------------- real ONNX Runtime plumbing


def _save(graph, path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)


def test_onnx_upscaler_on_a_tiny_graph(tmp_path, rng):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from cutout.core.enhance import OnnxUpscaler

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h4", "w4"])
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 4, 4])
    node = helper.make_node("Resize", ["input", "", "scales"], ["output"], mode="nearest")
    path = tmp_path / "up.onnx"
    _save(helper.make_graph([node], "g", [x], [y], initializer=[scales]), path)
    del onnx

    up = OnnxUpscaler(ENHANCE_REGISTRY["realesrgan-x4plus"], path, device="cpu")
    rgb = _img(rng, 45, 70)
    out = upscale_image(rgb, up, 4, tile=32, pad=4)
    np.testing.assert_array_equal(out, np.repeat(np.repeat(rgb, 4, 0), 4, 1))
    np.testing.assert_array_equal(upscale_image(rgb, up, 1, tile=32), rgb)


def test_onnx_face_restorer_on_a_tiny_graph(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from cutout.core.enhance import OnnxFaceRestorer

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 512, 512])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 512, 512])
    path = tmp_path / "face.onnx"
    _save(helper.make_graph([helper.make_node("Neg", ["input"], ["output"])], "g", [x], [y]), path)

    fr = OnnxFaceRestorer(ENHANCE_REGISTRY["gfpgan-1.4"], path, device="cpu")
    face = np.full((512, 512, 3), 0.25, np.float32)
    # [0, 1] -> [-1, 1] in, negated, back to [0, 1]: 0.25 -> -0.5 -> 0.5 -> 0.75
    np.testing.assert_allclose(fr.restore(face), 0.75, atol=1e-6)
