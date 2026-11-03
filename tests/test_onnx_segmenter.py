"""Runs real ONNX Runtime sessions on tiny generated graphs, so the I/O plumbing is tested in CI."""

import dataclasses

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

from cutout.core.models import REGISTRY, ResizeMode  # noqa: E402
from cutout.core.segmenter import OnnxSegmenter  # noqa: E402


def _mean_channel_model(path, size, logits: bool):
    """Output = mean over RGB of the (normalised) input; optionally scaled to look like logits."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, size, size])
    nodes = [helper.make_node("ReduceMean", ["input", "axes"], ["m"], keepdims=1)]
    inits = [helper.make_tensor("axes", TensorProto.INT64, [1], [1])]
    if logits:
        inits.append(helper.make_tensor("k", TensorProto.FLOAT, [], [20.0]))
        inits.append(helper.make_tensor("c", TensorProto.FLOAT, [], [0.5]))
        nodes += [helper.make_node("Sub", ["m", "c"], ["d"]), helper.make_node("Mul", ["d", "k"], ["out"])]
    else:
        nodes[0].output[0] = "out"
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)


@pytest.mark.parametrize("mode", [ResizeMode.STRETCH, ResizeMode.LETTERBOX])
@pytest.mark.parametrize("logits", [False, True])
def test_onnx_segmenter_runs(tmp_path, mode, logits):
    size = 64
    path = tmp_path / "m.onnx"
    _mean_channel_model(path, size, logits)
    spec = dataclasses.replace(
        REGISTRY["isnet-anime"], input_size=size, resize_mode=mode, activation="auto", output_index=0
    )
    seg = OnnxSegmenter(spec, path, device="cpu")
    assert seg.session_info.provider == "CPUExecutionProvider"

    img = np.zeros((90, 160, 3), np.uint8)
    img[:, 80:] = 255  # right half white -> foreground
    alpha = seg.predict(img)
    if mode is ResizeMode.LETTERBOX:
        assert alpha.shape == (36, 64)  # padding cropped away, aspect kept
    else:
        assert alpha.shape == (64, 64)
    assert alpha.min() >= 0 and alpha.max() <= 1
    assert alpha[:, -5:].mean() > 0.9 and alpha[:, :5].mean() < 0.1
    seg.warmup()
