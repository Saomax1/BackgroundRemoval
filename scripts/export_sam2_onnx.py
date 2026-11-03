"""Export SAM 2.1 checkpoints to the encoder + decoder ONNX pair the click-to-select tool uses.

Build machine only (needs internet + torch); the app itself never downloads anything.

    pip install torch "git+https://github.com/facebookresearch/sam2.git" huggingface_hub onnx
    python scripts/export_sam2_onnx.py sam2.1-hiera-large
    python scripts/export_sam2_onnx.py sam2.1-hiera-base-plus --check photo.jpg

The graphs follow the contract in ``src/cutout/core/prompt_segmenter.py``. The
decoder re-implements SAM's prompt encoder so that "no previous mask" is an
input (``has_mask_input``) rather than Python control flow, the same trick as
SAM 1's official ONNX export, and returns all four mask tokens (0 = single-mask
head, 1-3 = multimask) so the app picks the right one per click.

``--check IMAGE`` runs the exported pair through ONNX Runtime next to the
PyTorch predictor on one centre click and prints the mask IoU between them
(expect > 0.98).

Status: written for the build pipeline, not yet run end-to-end (the dev sandbox
that produced it has no access to Hugging Face). Run it with ``--check`` once
and pin the printed SHA-256 hashes in ``PROMPT_REGISTRY``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cutout.core.models import PROMPT_REGISTRY, default_models_dir, prompt_model_paths  # noqa: E402

FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]  # SAM 2 backbone levels at 1024 input


def build_wrappers(sam):
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            backbone_out = self.model.forward_image(image)
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
            if self.model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            feats = [
                f.permute(1, 2, 0).reshape(1, -1, *size)
                for f, size in zip(vision_feats[::-1], FEAT_SIZES[::-1])
            ][::-1]
            return feats[0], feats[1], feats[2]  # high_res_feats_0, high_res_feats_1, image_embed

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.pe = model.sam_prompt_encoder
            self.img_size = model.image_size

        def _embed_points(self, coords, labels):
            pe = self.pe
            coords = (coords + 0.5) / self.img_size
            emb = pe.pe_layer._pe_encoding(coords)
            labels = labels.unsqueeze(-1).expand_as(emb)
            emb = emb * (labels != -1)
            emb = emb + pe.not_a_point_embed.weight * (labels == -1)
            for i in range(pe.num_point_embeddings):
                emb = emb + pe.point_embeddings[i].weight * (labels == i)
            return emb

        def _embed_masks(self, mask_input, has_mask_input):
            pe = self.pe
            dense = has_mask_input * pe.mask_downscaling(mask_input)
            return dense + (1 - has_mask_input) * pe.no_mask_embed.weight.reshape(1, -1, 1, 1)

        def forward(
            self, image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input
        ):
            sparse = self._embed_points(point_coords, point_labels)
            dense = self._embed_masks(mask_input, has_mask_input)
            masks, iou, _, _ = self.model.sam_mask_decoder.predict_masks(
                image_embeddings=image_embed,
                image_pe=self.pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                repeat_image=False,
                high_res_features=[high_res_feats_0, high_res_feats_1],
            )
            return masks, iou

    return Encoder(sam).eval(), Decoder(sam).eval()


def export(model_id: str, models_dir: Path, opset: int) -> tuple[Path, Path]:
    import torch
    from sam2.build_sam import build_sam2_hf

    spec = PROMPT_REGISTRY[model_id]
    repo = spec.export_source.removeprefix("hf:")
    sam = build_sam2_hf(repo, device="cpu")
    sam.eval()
    encoder, decoder = build_wrappers(sam)
    enc_path, dec_path = prompt_model_paths(spec, models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    s = spec.input_size
    image = torch.randn(1, 3, s, s)
    with torch.no_grad():
        hr0, hr1, embed = encoder(image)
        print(f"{model_id}: exporting encoder -> {enc_path}")
        torch.onnx.export(
            encoder,
            (image,),
            str(enc_path),
            input_names=["image"],
            output_names=["high_res_feats_0", "high_res_feats_1", "image_embed"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

        r = spec.low_res_size
        dummy = (
            embed,
            hr0,
            hr1,
            torch.randint(0, s, (1, 3, 2), dtype=torch.float32),
            torch.tensor([[1, 0, -1]], dtype=torch.float32),
            torch.zeros(1, 1, r, r),
            torch.zeros(1),
        )
        print(f"{model_id}: exporting decoder -> {dec_path}")
        torch.onnx.export(
            decoder,
            dummy,
            str(dec_path),
            input_names=[
                "image_embed",
                "high_res_feats_0",
                "high_res_feats_1",
                "point_coords",
                "point_labels",
                "mask_input",
                "has_mask_input",
            ],
            output_names=["masks", "iou_predictions"],
            dynamic_axes={"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return enc_path, dec_path


def check(model_id: str, models_dir: Path, image_path: Path) -> None:
    import numpy as np
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    from cutout.core.engine import CutoutEngine
    from cutout.core.imageio import load_image

    img = load_image(image_path)
    rgb = img.as_uint8()
    h, w = rgb.shape[:2]
    click = np.array([[w / 2, h / 2]])

    predictor = SAM2ImagePredictor.from_pretrained(PROMPT_REGISTRY[model_id].export_source.removeprefix("hf:"))
    predictor.set_image(rgb)
    ref, ref_scores, _ = predictor.predict(point_coords=click, point_labels=np.array([1]), multimask_output=True)
    ref_mask = ref[int(np.argmax(ref_scores))] > 0.5

    engine = CutoutEngine(models_dir=models_dir, device="cpu")
    sel = engine.click_selector(img, model_id)
    out = sel.add_click(w / 2 + 0.5, h / 2 + 0.5).full_mask(h, w) > 0.5
    iou = (ref_mask & out).sum() / max(1, (ref_mask | out).sum())
    print(f"{model_id}: IoU vs PyTorch predictor on centre click = {iou:.4f}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", choices=sorted(PROMPT_REGISTRY))
    ap.add_argument("--models-dir", type=Path, default=default_models_dir())
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", type=Path, metavar="IMAGE", help="compare against the PyTorch predictor")
    args = ap.parse_args()

    enc, dec = export(args.model_id, args.models_dir, args.opset)
    for p in (enc, dec):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.0f} MB sha256={sha256_of(p)}")
    if args.check:
        check(args.model_id, args.models_dir, args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
