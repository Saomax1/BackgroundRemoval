"""The optimised-graph cache: later loads skip ONNX Runtime's graph optimisation, same output."""

import dataclasses
import logging

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from cutout.core import runtime  # noqa: E402
from cutout.core.models import REGISTRY  # noqa: E402
from cutout.core.segmenter import OnnxSegmenter  # noqa: E402

SIZE = 32


def _conv_model(path, seed=0, layers=6):
    """A small conv net with foldable constants and Conv+Mul+Relu chains ORT fuses."""
    rng = np.random.default_rng(seed)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, SIZE, SIZE])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, SIZE, SIZE])
    nodes, inits, prev, cin = [], [], "input", 3
    for i in range(layers):
        cout = 1 if i == layers - 1 else 16
        inits += [numpy_helper.from_array((rng.standard_normal((cout, cin, 3, 3)) * 0.3).astype(np.float32), f"w{i}"),
                  numpy_helper.from_array(rng.standard_normal(cout).astype(np.float32), f"b{i}"),
                  numpy_helper.from_array(rng.random((1, cout, 1, 1)).astype(np.float32) + 0.5, f"s{i}")]
        nodes += [helper.make_node("Conv", [prev, f"w{i}", f"b{i}"], [f"c{i}"], pads=[1, 1, 1, 1]),
                  helper.make_node("Mul", [f"s{i}", f"s{i}"], [f"ss{i}"]),  # constant: folded away
                  helper.make_node("Mul", [f"c{i}", f"ss{i}"], [f"m{i}"])]
        prev = f"m{i}"
        if i < layers - 1:
            nodes.append(helper.make_node("Relu", [prev], [f"r{i}"]))
            prev = f"r{i}"
        cin = cout
    nodes.append(helper.make_node("Sigmoid", [prev], ["out"]))
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y], initializer=inits),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)


@pytest.fixture
def spec():
    return dataclasses.replace(REGISTRY["isnet-anime"], input_size=SIZE)


@pytest.fixture
def image():
    return np.random.default_rng(1).integers(0, 256, (40, 50, 3), dtype=np.uint8)


def _cache_dirs():
    return sorted(p.name for p in runtime.graph_cache_root().iterdir())


def test_second_load_uses_the_cached_graph_and_gives_identical_output(tmp_path, spec, image):
    _conv_model(tmp_path / "m.onnx")
    first = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    assert first.info.graph_cache is None
    assert len(_cache_dirs()) == 1 and _cache_dirs()[0].startswith("m-")
    second = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    assert second.info.graph_cache is not None
    plain_opts = runtime._session_options()
    import onnxruntime as ort

    plain = ort.InferenceSession(str(tmp_path / "m.onnx"), plain_opts, providers=["CPUExecutionProvider"])
    from cutout.core.segmenter import postprocess, preprocess

    tensor, crop = preprocess(image, spec)
    reference = postprocess(plain.run(None, {"input": tensor})[0], crop, spec)
    np.testing.assert_array_equal(first.predict(image), reference)
    np.testing.assert_array_equal(second.predict(image), reference)  # bit-identical


def test_a_changed_model_gets_a_new_graph_and_the_old_one_is_removed(tmp_path, spec):
    _conv_model(tmp_path / "m.onnx", seed=0)
    OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    old = _cache_dirs()
    _conv_model(tmp_path / "m.onnx", seed=1, layers=7)  # an updated model file
    seg = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    assert seg.info.graph_cache is None  # not the stale graph
    assert len(_cache_dirs()) == 1 and _cache_dirs() != old


def test_a_broken_cached_graph_is_rebuilt(tmp_path, spec, image, caplog):
    _conv_model(tmp_path / "m.onnx")
    good = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu").predict(image)
    (runtime.graph_cache_root() / _cache_dirs()[0] / runtime.GRAPH_FILE).write_bytes(b"not a model")
    with caplog.at_level(logging.WARNING):
        seg = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    assert seg.info.graph_cache is None and "rebuilding" in caplog.text
    np.testing.assert_array_equal(seg.predict(image), good)
    assert OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu").info.graph_cache is not None


def test_a_cached_graph_that_fails_to_run_falls_back_to_the_model(tmp_path, spec, image):
    _conv_model(tmp_path / "m.onnx")
    good = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu").predict(image)
    seg = OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    assert seg.info.graph_cache is not None

    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("kernel failed")

    seg.session = Broken()
    np.testing.assert_array_equal(seg.predict(image), good)
    assert seg.info.graph_cache is None and _cache_dirs() == []


def test_cache_can_be_turned_off(tmp_path, spec, monkeypatch):
    monkeypatch.setenv("CUTOUT_GRAPH_CACHE", "0")
    _conv_model(tmp_path / "m.onnx")
    OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu")
    assert OnnxSegmenter(spec, tmp_path / "m.onnx", device="cpu").info.graph_cache is None
