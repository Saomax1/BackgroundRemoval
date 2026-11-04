"""Head only: the cut at the neck, hair and arms, several faces, the bundled face detector."""

import math

import numpy as np
import pytest

from conftest import FakeSegmenter
from cutout.core import debug
from cutout.core.engine import CutoutEngine
from cutout.core.head import (
    FACE_MODEL,
    Face,
    FaceFinder,
    HeadOptions,
    NoFaceFound,
    head_only,
    keep_head,
)
from cutout.core.imageio import LoadedImage
from cutout.core.models import REGISTRY, Subject


def _disc(a, cx, cy, r, value=1.0):
    yy, xx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
    a[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = value


def person(h=800, w=600, arm=False):
    """A matte of a simple person: round head (face box 100x120 at 250..350 x 200..320),
    hair on top, a neck and a wide body. ``arm``: a raised arm beside the head that
    touches the hair."""
    a = np.zeros((h, w), np.float32)
    _disc(a, 300, 250, 85)  # head and hair
    a[300:420, 270:330] = 1  # neck
    a[380:, 150:450] = 1  # shoulders and body
    if arm:
        a[60:400, 395:440] = 1  # upper arm going up from the shoulder ...
        a[180:230, 375:400] = 0.6  # ... touching the hair
    face = Face(250, 200, 100, 120, 0.9)
    return a, face


def test_keeps_head_drops_body_same_size():
    a, face = person()
    cut = keep_head(a, [face])
    out = cut.alpha
    assert out.shape == a.shape and out.dtype == np.float32
    assert cut.faces == [face]
    assert out[250, 300] == 1.0  # face
    assert out[175, 300] == 1.0  # hair above the face
    assert out[700, 300] == 0.0 and out[500, 200] == 0.0  # body
    # Never adds anything: the head is a part of the cutout.
    assert np.all(out <= a + 1e-6)


def test_cut_is_below_the_chin_and_feathered():
    a, face = person()
    out = keep_head(a, [face]).alpha
    o = HeadOptions()
    chin = face.y + face.h  # 320
    cut = chin + o.neck * face.h  # 344
    column = out[:, 300]
    assert column[int(chin) - 1] == 1.0  # the chin and some neck are kept
    assert column[int(cut) + 10] == 0.0
    soft = column[int(chin) : int(cut) + 10]
    assert np.any((soft > 0.05) & (soft < 0.95))  # a soft edge, not a scissor line
    # Beside the neck the cut rises to chin level (hair at the side is cut there).
    assert out[int(chin) + 15, 230] == 0.0


def test_raised_arm_touching_hair_is_dropped():
    a, face = person(arm=True)
    out = keep_head(a, [face]).alpha
    assert out[250, 300] == 1.0
    assert out[100, 417] == 0.0  # the raised arm
    assert not out[:, 400:].any()  # all of it, up to where it touches the hair


def test_hair_and_hat_sticking_out_are_kept():
    a, face = person()
    a[60:170, 290:310] = 1  # a tall hat / bun on top of the head
    a[150:300, 150:220] = 1  # big hair at the side, above the chin
    a[150:200, 210:230] = 1  # joined to the head
    out = keep_head(a, [face]).alpha
    assert out[70, 300] == 1.0
    assert out[200, 160] == 1.0


def test_tilted_head_cut_follows_the_tilt():
    a = np.zeros((800, 800), np.float32)
    _disc(a, 400, 400, 300)  # one big blob: only the cut decides
    roll = math.radians(30)
    face = Face(350, 340, 100, 120, roll=roll)
    out = keep_head(a, [face]).alpha
    cx, cy = face.center
    dx, dy = -math.sin(roll), math.cos(roll)  # chin-wards
    fw, fh = face.size
    near = fh / 2 - 5
    far = fh / 2 + HeadOptions().neck * fh + 15
    assert out[round(cy + near * dy), round(cx + near * dx)] == 1.0
    assert out[round(cy + far * dy), round(cx + far * dx)] == 0.0


def test_two_people_both_heads_kept():
    a1, f1 = person(w=600)
    a = np.concatenate([a1, a1], axis=1)
    f2 = Face(f1.x + 600, f1.y, f1.w, f1.h, 0.8)
    cut = keep_head(a, [f1, f2])
    assert cut.alpha[250, 300] == 1.0 and cut.alpha[250, 900] == 1.0
    assert cut.alpha[700, 300] == 0.0 and cut.alpha[700, 900] == 0.0
    assert len(cut.faces) == 2


def test_face_off_the_cutout_raises():
    a, _ = person()
    with pytest.raises(NoFaceFound):
        keep_head(a, [Face(20, 600, 60, 70)])  # in the empty background
    with pytest.raises(NoFaceFound):
        keep_head(a, [])


def test_user_box_any_corner_order():
    f = Face.from_box(350, 320, 250, 200)
    assert (f.x, f.y, f.w, f.h, f.roll) == (250, 200, 100, 120, 0.0)


def test_finder_filters_off_matte_and_small_faces(monkeypatch):
    a, face = person()
    rgb = np.zeros(a.shape + (3,), np.uint8)
    finder = FaceFinder()
    seen = {}

    def fake_detect(crop, threshold):
        seen.setdefault("shapes", []).append(crop.shape)
        # Coordinates are relative to the crop around the cutout.
        ox, oy = seen["offset"]
        return [
            Face(face.x - ox, face.y - oy, face.w, face.h, 0.9),
            Face(5 - ox + 400, 5 - oy, 50, 60, 0.95),  # off the cutout (background)
            Face(face.x - ox + 30, face.y - oy + 30, 20, 24, 0.7),  # tiny, on the matte
        ]

    ys, xs = np.nonzero(a > 0.5)
    m = int(0.05 * max(xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())) + 1
    seen["offset"] = (max(0, xs.min() - m), max(0, ys.min() - m))
    monkeypatch.setattr(finder, "detect", fake_detect)
    faces = finder.find(rgb, a, Subject.PERSON)
    assert len(faces) == 1 and abs(faces[0].x - face.x) < 1e-6 and abs(faces[0].y - face.y) < 1e-6
    assert seen["shapes"][0][0] < a.shape[0]  # looked only around the cutout


def test_finder_retries_lower_for_drawings(monkeypatch):
    a, face = person()
    finder = FaceFinder()
    thresholds = []

    def fake_detect(crop, threshold):
        thresholds.append(threshold)
        return []

    monkeypatch.setattr(finder, "detect", fake_detect)
    assert finder.find(np.zeros(a.shape + (3,), np.uint8), a, Subject.ANIME) == []
    o = HeadOptions()
    assert thresholds == [o.anime_threshold, o.retry_threshold]
    with pytest.raises(NoFaceFound):
        head_only(np.zeros(a.shape + (3,), np.uint8), a, Subject.ANIME, finder=finder)


def test_bundled_face_detector_loads_and_runs():
    assert FACE_MODEL.is_file() and FACE_MODEL.stat().st_size < 1_000_000
    finder = FaceFinder()
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert finder.detect(noise, 0.9) == []
    flat = np.full((3000, 2000, 3), 128, np.uint8)  # big input is shrunk first
    assert finder.detect(flat, 0.6) == []


def test_engine_head_only_uses_cpu_and_records_timing(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / REGISTRY["birefnet-general"].filename).write_bytes(b"x")
    engine = CutoutEngine(models_dir=models, segmenter_factory=FakeSegmenter)
    a, face = person()
    image = LoadedImage(rgb=np.zeros(a.shape + (3,), np.uint8))
    before = a.copy()
    cut = engine.head_only(image, a, Subject.PERSON, faces=[face])
    assert np.array_equal(a, before)  # the input matte is not changed
    assert cut.alpha[700, 300] == 0.0 and cut.alpha[250, 300] == 1.0
    assert engine.loaded_models == []  # no model was loaded for this
    assert any("head only" in line for line in debug.recent_timings())
