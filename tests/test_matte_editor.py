import numpy as np
import pytest

from cutout.core.imageio import LoadedImage
from cutout.editing import Brush, BrushMode, EditHistory, MaskOp, MatteEditor, Rect
from cutout.editing.brush import dab_centres, stamp
from cutout.editing.matte import TILE


def _alpha(h=600, w=900, value=0.0):
    return np.full((h, w), value, np.float32)


def test_stamp_profile():
    r = Rect(0, 0, 41, 41)
    hard = stamp(20.3, 20.5, 10, 1.0, r)
    assert hard[20, 20] == 1.0 and hard[20, 35] == 0.0
    assert 0.0 < hard[20, 29] < 1.0  # 1 px anti-aliased rim
    soft = stamp(20.5, 20.5, 10, 0.0, r)
    assert soft[20, 20] > 0.95 and 0.2 < soft[20, 25] < 0.8


def test_dab_spacing_is_continuous_across_segments():
    a, carry = dab_centres(0, 0, 10, 0, 4.0, 0.0)
    b, carry = dab_centres(10, 0, 20, 0, 4.0, carry)
    xs = [x for x, _, _ in a + b]
    assert np.allclose(np.diff(xs), 4.0) and xs[0] == 4.0


def test_keep_and_erase_strokes_and_undo():
    alpha = _alpha()
    original = alpha.copy()
    ed = MatteEditor(alpha)
    dirty = ed.begin_stroke(100, 100, Brush(BrushMode.KEEP, radius=20, hardness=1.0))
    assert not dirty.empty and dirty.x <= 80 and dirty.x1 >= 120
    ed.continue_stroke(700, 100)
    patch = ed.end_stroke()
    assert alpha is ed.alpha  # edited in place
    assert alpha[100, 100] == 1.0 and alpha[100, 400] == 1.0 and alpha[100, 700] == 1.0
    assert alpha[150, 400] == 0.0 and alpha[500, 500] == 0.0
    # only tiles the stroke crossed are stored, not the whole matte
    assert patch.nbytes <= 4 * TILE * TILE * 4
    assert ed.can_undo and ed.history.undo_label == "Keep brush"

    ed.begin_stroke(400, 50, Brush(BrushMode.ERASE, radius=10, hardness=1.0))
    ed.continue_stroke(400, 150)
    ed.end_stroke()
    assert alpha[100, 400] == 0.0 and alpha[100, 300] == 1.0

    ed.undo()
    assert alpha[100, 400] == 1.0
    ed.undo()
    assert np.array_equal(alpha, original)
    assert not ed.can_undo and ed.can_redo
    ed.redo()
    assert alpha[100, 400] == 1.0
    ed.redo()
    assert alpha[100, 400] == 0.0


def test_overlapping_dabs_do_not_build_up():
    alpha = _alpha()
    ed = MatteEditor(alpha)
    b = Brush(BrushMode.KEEP, radius=30, hardness=0.0, opacity=0.5)
    ed.begin_stroke(300, 300, b)
    for _ in range(20):
        ed.continue_stroke(301, 300)
        ed.continue_stroke(300, 300)
    ed.end_stroke()
    assert alpha.max() <= 0.5 + 1e-6
    assert alpha[300, 300] == pytest.approx(0.5, abs=0.02)


def test_patch_before_after_for_external_undo_stack():
    alpha = _alpha(300, 300, 0.25)
    ed = MatteEditor(alpha, history=None)
    ed.begin_stroke(150, 150, Brush(BrushMode.ERASE, radius=40, hardness=1.0))
    patch = ed.end_stroke()
    assert not ed.can_undo  # caller owns history
    assert patch.before.shape == (patch.rect.h, patch.rect.w)
    assert np.all(patch.before == 0.25)
    assert np.array_equal(patch.after, alpha[patch.rect.slices])
    assert patch.after.min() == 0.0


def test_cancel_stroke_restores():
    alpha = _alpha()
    ed = MatteEditor(alpha)
    ed.begin_stroke(10, 10, Brush(radius=50))
    ed.continue_stroke(500, 500)
    ed.cancel_stroke()
    assert alpha.max() == 0.0 and not ed.can_undo


def test_strokes_clip_at_image_edge():
    alpha = _alpha(100, 100)
    ed = MatteEditor(alpha)
    ed.begin_stroke(-20, -20, Brush(radius=40, hardness=1.0))
    ed.continue_stroke(120, 50)
    ed.end_stroke()
    assert alpha[0, 0] == 1.0 and alpha.shape == (100, 100)


def test_apply_mask_ops():
    alpha = _alpha(400, 400, 0.5)
    ed = MatteEditor(alpha)
    m = np.ones((50, 50), np.float32)
    ed.apply_mask(m, MaskOp.ADD, Rect(10, 10, 50, 50))
    assert alpha[20, 20] == 1.0 and alpha[100, 100] == 0.5
    ed.apply_mask(m, MaskOp.SUBTRACT, Rect(380, 380, 50, 50))  # partly off-image
    assert alpha[390, 390] == 0.0 and alpha[370, 370] == 0.5
    ed.restore_original()
    assert np.all(alpha == 0.5)
    ed.undo()
    assert alpha[20, 20] == 1.0 and alpha[390, 390] == 0.0


def test_history_memory_cap_drops_oldest():
    alpha = _alpha(512, 512)
    hist = EditHistory(max_bytes=3 * TILE * TILE * 4)
    ed = MatteEditor(alpha, history=hist)
    for x in (10, 300, 10, 300):
        ed.begin_stroke(x, 10, Brush(BrushMode.KEEP, radius=5))
        ed.end_stroke()
        ed.begin_stroke(x, 10, Brush(BrushMode.ERASE, radius=5))
        ed.end_stroke()
    assert hist.nbytes <= hist.max_bytes


def test_refine_brush_snaps_hard_edge_to_image():
    h, w = 200, 200
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, 100:] = 255  # true edge at x = 100
    alpha = np.zeros((h, w), np.float32)
    alpha[:, 90:] = 1.0  # matte edge 10 px off
    ed = MatteEditor(alpha, LoadedImage(rgb))
    ed.begin_stroke(95, 20, Brush(BrushMode.REFINE, radius=30, hardness=1.0))
    ed.continue_stroke(95, 180)
    ed.end_stroke()
    row = alpha[100]
    assert row[110] > 0.9 and row[80] < 0.1
    assert row[92] < 0.7 and row[98] < 0.95  # the wrongly kept strip fades out...
    assert row[101] > 0.97  # ...and the matte now has its sharp edge on the real one
    assert np.all(alpha[:, 150:] == 1.0)  # outside the brush untouched


def test_editor_rejects_bad_alpha():
    with pytest.raises(ValueError):
        MatteEditor(np.zeros((10, 10), np.float64))
    with pytest.raises(ValueError):
        MatteEditor(np.zeros((10, 10), np.float32), LoadedImage(np.zeros((5, 5, 3), np.uint8)))
