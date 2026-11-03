import numpy as np
import pytest

from cutout.core.engine import CutoutEngine
from cutout.core.imageio import LoadedImage
from cutout.core.models import PROMPT_REGISTRY, ModelNotInstalled, resolve_prompt_model
from cutout.core.prompt_segmenter import ImageEmbedding, PromptPrediction, to_model_prompt
from cutout.editing import ClickSelector, MaskOp, MatteEditor, Rect


class FakeSam:
    """Each positive click selects a disc of ``r`` low-res cells around it; negative clicks carve one out.

    Works in SAM's stretched square space, like the real decoder, so the selector's coordinate
    mapping and full-res upsampling are exercised for non-square images.
    """

    name = "fake-sam"

    def __init__(self, *args, low_res=256, input_size=1024, r=20, **kwargs):
        self.low_res, self.input_size, self.r = low_res, input_size, r
        self.encodes = 0
        self.last_mask_input = None

    def encode(self, rgb):
        self.encodes += 1
        return ImageEmbedding(height=rgb.shape[0], width=rgb.shape[1])

    def decode(self, embedding, points, labels, box=None, mask_input=None):
        self.last_mask_input = mask_input
        coords, labs = to_model_prompt(embedding.height, embedding.width, self.input_size, points, labels, box)
        n = self.low_res
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5
        sel = np.zeros((n, n), bool)
        f = n / self.input_size
        for (x, y), lab in zip(coords, labs):
            d = np.hypot(xx - (x + 0.5) * f, yy - (y + 0.5) * f)
            if lab == 1:
                sel |= d < self.r
            elif lab == 0:
                sel &= d >= self.r / 2
        logits = np.where(sel, 8.0, -8.0).astype(np.float32)
        return PromptPrediction(
            logits=np.stack([logits] * 4), scores=np.array([0.9, 0.5, 0.95, 0.6], np.float32),
            single_index=0, multi_indices=(1, 2, 3),
        )


def _photo(h=900, w=1600):
    """Dark background with a bright disc centred at (800, 450), radius 200 px."""
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.full((h, w, 3), 30, np.uint8)
    rgb[np.hypot(xx - 800, yy - 450) < 200] = 220
    return LoadedImage(rgb)


def test_to_model_prompt_maps_scene_coords_and_pads():
    coords, labs = to_model_prompt(500, 2000, 1024, np.array([[1000.5, 250.5]]), np.array([1]))
    assert np.allclose(coords[0], [1000 * 1024 / 2000, 250 * 1024 / 500])
    assert list(labs) == [1, -1] and np.allclose(coords[1], 0)
    coords, labs = to_model_prompt(100, 100, 1024, np.zeros((0, 2)), np.zeros(0), box=(50.5, 60.5, 10.5, 20.5))
    assert list(labs) == [2, 3] and coords[0][0] < coords[1][0]


def test_click_selects_full_res_mask_on_real_edges():
    img = _photo()
    fake = FakeSam()
    sel = ClickSelector(fake, img)
    sel.prepare()
    sel.prepare()
    assert fake.encodes == 1 and sel.ready

    s = sel.add_click(800.5, 450.5)
    assert s.score == pytest.approx(0.95)  # one click: best multimask candidate
    assert not s.empty and s.mask.shape == (s.rect.h, s.rect.w)
    assert s.rect.intersect(Rect(0, 0, 1600, 900)) == s.rect
    full = s.full_mask(900, 1600)
    assert full.dtype == np.float32 and full.shape == (900, 1600)
    assert full[450, 800] == 1.0 and full[50, 50] == 0.0
    # Fake disc: 20 cells of 256 -> 125 px wide, 70 px tall (stretched). Mask stays near that,
    # and nothing far outside it is selected.
    assert full[450, 900] == 1.0 and full[450, 1000] == 0.0
    assert s.preview.shape == (256, 256)


def test_negative_click_and_undo_click():
    sel = ClickSelector(FakeSam(), _photo())
    first = sel.add_click(800.5, 450.5)
    second = sel.add_click(800.5, 450.5, positive=False)
    assert second.full_mask(900, 1600)[450, 800] == 0.0
    assert len(sel.clicks) == 2
    back = sel.undo_click()
    assert back is first and len(sel.clicks) == 1
    sel.clear()
    assert sel.selection is None and not sel.clicks


def test_previous_logits_fed_back():
    fake = FakeSam()
    sel = ClickSelector(fake, _photo())
    sel.add_click(100.5, 100.5)
    assert fake.last_mask_input is None
    sel.add_click(200.5, 100.5)
    assert fake.last_mask_input is not None and fake.last_mask_input.shape == (256, 256)


def test_empty_selection_when_nothing_found():
    sel = ClickSelector(FakeSam(), _photo())
    sel.add_click(800.5, 450.5, positive=False)
    assert sel.selection.empty


def test_selection_merges_into_matte():
    img = _photo()
    alpha = np.zeros((900, 1600), np.float32)
    ed = MatteEditor(alpha, img)
    sel = ClickSelector(FakeSam(), img)
    patch = ed.apply_mask(sel.add_click(800.5, 450.5), MaskOp.ADD)
    assert alpha[450, 800] == 1.0 and alpha[0, 0] == 0.0
    assert patch.rect.intersect(sel.selection.rect) == patch.rect
    sel.clear()
    ed.apply_mask(sel.add_click(800.5, 450.5), MaskOp.SUBTRACT)
    assert alpha[450, 800] == 0.0 and alpha.max() <= 0.5  # soft rim: min(m, 1 - m)
    ed.undo()
    assert alpha[450, 800] == 1.0


def test_engine_click_selector(tmp_path):
    with pytest.raises(ModelNotInstalled):
        resolve_prompt_model(tmp_path)
    spec = PROMPT_REGISTRY["sam2.1-hiera-base-plus"]
    (tmp_path / spec.encoder_filename).write_bytes(b"x")
    with pytest.raises(ModelNotInstalled):  # needs both halves
        resolve_prompt_model(tmp_path)
    (tmp_path / spec.decoder_filename).write_bytes(b"x")
    assert resolve_prompt_model(tmp_path).id == spec.id

    engine = CutoutEngine(models_dir=tmp_path, prompt_segmenter_factory=FakeSam)
    assert engine.prompt_segmenter() is engine.prompt_segmenter()
    selector = engine.click_selector(_photo())
    assert selector.add_click(800.5, 450.5).full_mask(900, 1600)[450, 800] == 1.0
