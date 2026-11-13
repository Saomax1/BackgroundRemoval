"""Export a Real-ESRGAN upscaler (RRDBNet .pth) to ONNX (build machine only, needs torch).

    python scripts/export_realesrgan_onnx.py realesrgan-x4plus --models-dir build/enhance-models
    python scripts/export_realesrgan_onnx.py realesrgan-x4plus --weights RealESRGAN_x4plus.pth

Downloads the official weights from the Real-ESRGAN GitHub release (``export_source`` in
the registry) unless ``--weights`` is given, rebuilds the network (RRDBNet, the same
layers and parameter names as basicsr, so no basicsr install is needed), and exports it
with dynamic height and width so the app can feed tiles of any size. Then checks the
ONNX output against PyTorch on a random tile.

Input ``input``: 1x3xHxW float32 RGB in [0, 1]. Output ``output``: 1x3x(4H)x(4W).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cutout.core.models import ENHANCE_REGISTRY, EnhanceRole, default_models_dir, enhance_model_path  # noqa: E402


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """basicsr's RRDBNet for scale 4 (Real-ESRGAN x4plus and x4plus anime 6B)."""

    def __init__(self, num_block: int = 23, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_weights(net: nn.Module, path: Path) -> None:
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    net.load_state_dict(state, strict=True)


def export(net: nn.Module, dest: Path, opset: int = 17) -> None:
    net.eval()
    dummy = torch.rand(1, 3, 64, 64)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.onnx")
    kwargs = {}
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        kwargs["dynamo"] = False  # the TorchScript exporter: stable across torch versions
    with torch.no_grad():
        torch.onnx.export(
            net, dummy, str(tmp), opset_version=opset, input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height4", 3: "width4"}},
            **kwargs,
        )
    tmp.replace(dest)


def check(net: nn.Module, dest: Path, size: tuple[int, int] = (37, 53)) -> float:
    """Largest difference between PyTorch and ONNX Runtime on a random odd-sized tile."""
    import onnxruntime as ort

    x = torch.rand(1, 3, *size)
    with torch.no_grad():
        want = net(x).numpy()
    sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    if got.shape != want.shape:
        raise RuntimeError(f"ONNX output {got.shape} != PyTorch {want.shape}")
    return float(np.abs(got - want).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ids = sorted(k for k, s in ENHANCE_REGISTRY.items() if s.role is EnhanceRole.UPSCALE and s.export_source)
    ap.add_argument("model", choices=ids)
    ap.add_argument("--models-dir", type=Path, default=default_models_dir())
    ap.add_argument("--weights", type=Path, help="local .pth instead of downloading export_source")
    ap.add_argument("--random", action="store_true", help="random weights (tests the export only)")
    args = ap.parse_args()
    spec = ENHANCE_REGISTRY[args.model]
    net = RRDBNet(num_block=spec.num_block)
    if not args.random:
        weights = args.weights
        if weights is None:
            from fetch_models import download

            weights = Path(tempfile.gettempdir()) / Path(spec.export_source).name
            if not weights.is_file():
                print(f"{spec.id}: downloading {spec.export_source}")
                download(spec.export_source, weights)
        load_weights(net, weights)
    dest = enhance_model_path(spec, args.models_dir)
    export(net, dest)
    err = check(net, dest)
    print(f"{spec.id}: exported {dest} ({dest.stat().st_size / 1e6:.0f} MB), max diff vs PyTorch {err:.2e}")
    if err > 1e-3:
        dest.unlink()
        print(f"{spec.id}: ONNX output differs from PyTorch by {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
