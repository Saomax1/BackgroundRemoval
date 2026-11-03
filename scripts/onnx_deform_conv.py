"""ONNX-exportable drop-in for ``torchvision.ops.deform_conv2d`` (build machine only).

BiRefNet's decoder uses modulated deformable convolution. ONNX has no such op,
and the ``deform_conv2d_onnx_exporter`` symbolic needs static shapes that the
newer BiRefNet code (HR-matting, ToonOut) does not give the tracer, so it fails
with ``NoneType + int``. This module re-expresses the op with ``grid_sample``
(ONNX GridSample, opset 16+) and a matmul, which every ONNX Runtime execution
provider runs. Numerically it matches torchvision's bilinear sampling with zero
padding; ``tests``-style check: ``python scripts/onnx_deform_conv.py``.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def deform_conv2d(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), mask=None):
    b, c, h, w = input.shape
    out_ch, cg, kh, kw = weight.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    ho, wo = offset.shape[-2:]
    k = kh * kw
    og = offset.shape[1] // (2 * k)  # offset groups
    groups = c // cg  # weight groups
    dt = input.dtype

    # Sampling position of kernel tap (i, j) for output pixel (y, x), before the learned offset.
    ys = torch.arange(ho, dtype=dt) * sh - ph
    xs = torch.arange(wo, dtype=dt) * sw - pw
    ki = (torch.arange(kh, dtype=dt) * dh).repeat_interleave(kw)
    kj = (torch.arange(kw, dtype=dt) * dw).repeat(kh)
    base_y = ys.view(1, 1, 1, ho, 1) + ki.view(1, 1, k, 1, 1)
    base_x = xs.view(1, 1, 1, 1, wo) + kj.view(1, 1, k, 1, 1)

    off = offset.reshape(b, og, k, 2, ho, wo)  # torchvision order: (dy, dx) per tap
    py = base_y + off[:, :, :, 0]
    px = base_x + off[:, :, :, 1]
    # grid_sample with align_corners=True maps -1/+1 to the first/last pixel centre.
    gx = px * (2.0 / max(w - 1, 1)) - 1.0
    gy = py * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(b * og, k * ho, wo, 2)
    sampled = F.grid_sample(
        input.reshape(b * og, c // og, h, w), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (b*og, c/og, k*ho, wo)
    sampled = sampled.reshape(b, og, c // og, k, ho, wo)
    if mask is not None:
        sampled = sampled * mask.reshape(b, og, 1, k, ho, wo)
    cols = sampled.reshape(b, groups, cg * k, ho * wo)
    wmat = weight.reshape(groups, out_ch // groups, cg * k)
    out = torch.matmul(wmat.unsqueeze(0), cols).reshape(b, out_ch, ho, wo)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def install() -> int:
    """Swap the torchvision op for this one everywhere it was imported.

    Safe to call repeatedly: call it before loading the model (for code that
    imports the op at call time) and again after (for ``from torchvision.ops
    import deform_conv2d`` at module import). Returns how many model modules now
    use the replacement, so the caller can tell whether the swap took.
    """
    import torchvision.ops
    import torchvision.ops.deform_conv as tv_dc

    original = tv_dc.__dict__.get("_cutout_original", tv_dc.deform_conv2d)
    tv_dc._cutout_original = original
    torchvision.ops.deform_conv2d = deform_conv2d
    tv_dc.deform_conv2d = deform_conv2d
    using = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or name.startswith("torchvision") or name == __name__:
            continue
        current = getattr(mod, "deform_conv2d", None)
        if current is original:
            mod.deform_conv2d = deform_conv2d
            current = deform_conv2d
        if current is deform_conv2d:
            using += 1
    return using


if __name__ == "__main__":  # self-check against torchvision
    from torchvision.ops import deform_conv2d as reference

    torch.manual_seed(0)
    worst = 0.0
    for kh, stride, pad, dil, groups, og, use_mask in [
        (3, 1, 1, 1, 1, 1, True),
        (7, 1, 3, 1, 1, 1, True),
        (1, 1, 0, 1, 1, 1, True),
        (3, 2, 1, 2, 2, 2, False),
    ]:
        x = torch.randn(2, 8, 13, 11)
        wgt = torch.randn(6, 8 // groups, kh, kh)
        bias = torch.randn(6)
        ho = (13 + 2 * pad - dil * (kh - 1) - 1) // stride + 1
        wo = (11 + 2 * pad - dil * (kh - 1) - 1) // stride + 1
        offset = torch.randn(2, 2 * og * kh * kh, ho, wo) * 3  # big enough to leave the image
        mask = torch.rand(2, og * kh * kh, ho, wo) if use_mask else None
        ref = reference(x, offset, wgt, bias, stride=stride, padding=pad, dilation=dil, mask=mask)
        got = deform_conv2d(x, offset, wgt, bias, stride=stride, padding=pad, dilation=dil, mask=mask)
        worst = max(worst, (ref - got).abs().max().item())
    print(f"max abs difference vs torchvision: {worst:.2e}")
    raise SystemExit(0 if worst < 1e-3 else 1)
