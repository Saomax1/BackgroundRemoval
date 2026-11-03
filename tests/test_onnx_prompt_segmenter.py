"""Tiny generated encoder/decoder graphs with the SAM 2 I/O contract, run through real ONNX Runtime."""

import dataclasses

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

from cutout.core.imageio import LoadedImage  # noqa: E402
from cutout.core.models import PROMPT_REGISTRY  # noqa: E402
from cutout.core.prompt_segmenter import OnnxSamSegmenter  # noqa: E402
from cutout.editing import ClickSelector  # noqa: E402

S, R = 64, 16  # encoder input side, low-res mask side


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _encoder(path):
    """image_embed = brightness pooled to R x R; high-res feats are pass-through placeholders."""
    f = TensorProto.FLOAT
    x = helper.make_tensor_value_info("image", f, [1, 3, S, S])
    outs = [
        helper.make_tensor_value_info("high_res_feats_0", f, [1, 3, S, S]),
        helper.make_tensor_value_info("high_res_feats_1", f, [1, 3, S, S]),
        helper.make_tensor_value_info("image_embed", f, [1, 1, R, R]),
    ]
    k = S // R
    nodes = [
        helper.make_node("Identity", ["image"], ["high_res_feats_0"]),
        helper.make_node("Identity", ["image"], ["high_res_feats_1"]),
        helper.make_node("AveragePool", ["image"], ["pooled"], kernel_shape=[k, k], strides=[k, k]),
        helper.make_node("ReduceMean", ["pooled", "axes"], ["image_embed"], keepdims=1),
    ]
    inits = [helper.make_tensor("axes", TensorProto.INT64, [1], [1])]
    _save(helper.make_graph(nodes, "enc", [x], outs, initializer=inits), path)


def _decoder(path):
    """masks = 4 copies of (brightness - 0.5) * 20 (bright = selected); ignores the prompt."""
    f = TensorProto.FLOAT
    ins = [
        helper.make_tensor_value_info("image_embed", f, [1, 1, R, R]),
        helper.make_tensor_value_info("high_res_feats_0", f, [1, 3, S, S]),
        helper.make_tensor_value_info("high_res_feats_1", f, [1, 3, S, S]),
        helper.make_tensor_value_info("point_coords", f, [1, "n", 2]),
        helper.make_tensor_value_info("point_labels", f, [1, "n"]),
        helper.make_tensor_value_info("mask_input", f, [1, 1, R, R]),
        helper.make_tensor_value_info("has_mask_input", f, [1]),
    ]
    outs = [
        helper.make_tensor_value_info("masks", f, [1, 4, R, R]),
        helper.make_tensor_value_info("iou_predictions", f, [1, 4]),
    ]
    nodes = [
        helper.make_node("Sub", ["image_embed", "half"], ["d"]),
        helper.make_node("Mul", ["d", "k"], ["m"]),
        helper.make_node("Concat", ["m", "m", "m", "m"], ["masks"], axis=1),
        helper.make_node("Identity", ["iou"], ["iou_predictions"]),
    ]
    inits = [
        helper.make_tensor("half", f, [], [0.5]),
        helper.make_tensor("k", f, [], [20.0]),
        helper.make_tensor("iou", f, [1, 4], [0.9, 0.3, 0.8, 0.4]),
    ]
    _save(helper.make_graph(nodes, "dec", ins, outs, initializer=inits), path)


def test_onnx_sam_plumbing(tmp_path):
    enc, dec = tmp_path / "e.onnx", tmp_path / "d.onnx"
    _encoder(enc)
    _decoder(dec)
    spec = dataclasses.replace(
        PROMPT_REGISTRY["sam2.1-hiera-small"], input_size=S, low_res_size=R, mean=(0, 0, 0), std=(1, 1, 1)
    )
    seg = OnnxSamSegmenter(spec, enc, dec, device="cpu")
    assert seg.session_info.provider == "CPUExecutionProvider"

    rgb = np.zeros((300, 500, 3), np.uint8)
    rgb[:, 250:] = 255  # right half bright -> "selected"
    emb = seg.encode(rgb)
    assert emb.height == 300 and emb.width == 500 and set(emb.features) >= {"image_embed"}
    pred = seg.decode(emb, np.array([[400.5, 150.5]]), np.array([1]))
    assert pred.logits.shape == (4, R, R) and pred.best(1) == 2 and pred.best(2) == 0
    pred2 = seg.decode(emb, np.array([[400.5, 150.5]]), np.array([1]), mask_input=pred.logits[0])
    assert np.allclose(pred2.logits, pred.logits)
    seg.warmup()

    sel = ClickSelector(seg, LoadedImage(rgb)).add_click(400.5, 150.5)
    full = sel.full_mask(300, 500)
    assert full[:, 260:].min() == 1.0 and full[:, :240].max() == 0.0
    # the edge sits on the image's real edge at x = 250, at full resolution
    assert 0.0 <= full[150, 249] < 0.5 < full[150, 250] <= 1.0
