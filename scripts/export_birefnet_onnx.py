"""Export BiRefNet-family PyTorch checkpoints to ONNX (build machine only, needs internet + torch).

    pip install -r packaging/requirements-export.txt
    python scripts/export_birefnet_onnx.py birefnet-hr-matting
    python scripts/export_birefnet_onnx.py toonout   # downloads birefnet_finetuned_toonout.pth

The exported graph ends in a sigmoid, so it outputs alpha in [0, 1] directly.
BiRefNet uses deformable convolutions, which plain ``torch.onnx`` cannot export;
``deform_conv2d_onnx_exporter`` registers a symbolic for them (this is the route
BiRefNet's own ONNX tutorial uses).

Status: written for the build pipeline, not yet run end-to-end (the dev sandbox
that produced it has no access to Hugging Face). Validate the exported file with
``python -m cutout sample.jpg --model <id>`` before shipping it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutout.core.models import REGISTRY, default_models_dir, model_path  # noqa: E402

BASE_ARCH_REPO = "ZhengPeng7/BiRefNet"  # architecture used for fine-tunes that ship only a .pth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", choices=[s.id for s in REGISTRY.values() if s.export_source])
    ap.add_argument("--weights", type=Path, help="local .pth state dict (for fine-tunes such as ToonOut)")
    ap.add_argument("--models-dir", type=Path, default=default_models_dir())
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    from deform_conv2d_onnx_exporter import register_deform_conv2d_onnx_op
    from transformers import AutoModelForImageSegmentation

    register_deform_conv2d_onnx_op()
    spec = REGISTRY[args.model_id]
    repo = spec.export_source.removeprefix("hf:")
    if repo.endswith((".pth", ".pt")) and not args.weights:
        # "owner/repo/file.pth": a fine-tuned state dict published on the Hub (ToonOut).
        from huggingface_hub import hf_hub_download

        owner, name, filename = repo.split("/", 2)
        args.weights = Path(hf_hub_download(f"{owner}/{name}", filename))

    if args.weights:
        net = AutoModelForImageSegmentation.from_pretrained(BASE_ARCH_REPO, trust_remote_code=True)
        state = torch.load(args.weights, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"warning: {len(missing)} missing / {len(unexpected)} unexpected keys")
    else:
        net = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
    net.eval().requires_grad_(False)

    class Final(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            return torch.sigmoid(out)

    s = spec.input_size
    dummy = torch.randn(1, 3, s, s)
    dest = model_path(spec, args.models_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # no_grad: without it the trace keeps every activation for autograd, and the
    # 2048 px HR model runs a 16 GB CI runner out of memory.
    with torch.no_grad():
        torch.onnx.export(
            Final(net),
            dummy,
            str(dest),
            input_names=["input_image"],
            output_names=["alpha"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
