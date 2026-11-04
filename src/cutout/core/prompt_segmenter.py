"""Promptable segmenters: turn clicks on an image into a mask (click-to-select).

This is the sibling of ``segmenter.Segmenter`` for the touch-up tools. A prompt
model is split in two, so clicking feels instant:

* ``encode(rgb)`` runs the heavy image encoder once per image (hundreds of ms on
  GPU, seconds on CPU). Call it in a worker thread as soon as the auto cutout is
  done, so it is ready before the user reaches for the tool.
* ``decode(embedding, points, labels, ...)`` runs the small prompt decoder on
  every click (a few ms on GPU).

Masks come back as low-resolution *logits* covering the whole image (SAM 2:
256x256, stretched like the 1024x1024 encoder input). ``editing.select`` turns
them into a full-resolution, edge-refined mask; this module never resizes to
the photo's size.

Coordinates are image pixels in the same convention as a Qt scene: pixel
``(i, j)`` covers ``[i, i+1) x [j, j+1)`` and its centre is ``(i+0.5, j+0.5)``.

ONNX contract (what ``scripts/export_sam2_onnx.py`` produces)::

    encoder  in : image            float32 [1, 3, 1024, 1024]  RGB, ImageNet-normalised
             out: high_res_feats_0 [1, 32, 256, 256]
                  high_res_feats_1 [1, 64, 128, 128]
                  image_embed      [1, 256, 64, 64]
    decoder  in : image_embed, high_res_feats_0, high_res_feats_1  (from the encoder)
                  point_coords     float32 [1, N, 2]   (x, y) in 1024-space
                  point_labels     float32 [1, N]      1 add, 0 remove, 2/3 box corners, -1 padding
                  mask_input       float32 [1, 1, 256, 256]  previous logits
                  has_mask_input   float32 [1]
             out: masks            float32 [1, 4, 256, 256]  logits; 0 = single-mask head, 1-3 = multimask
                  iou_predictions  float32 [1, 4]

Encoder outputs are matched to decoder inputs by name, and the decoder is only
fed the inputs it declares, so a decoder exported without e.g. ``mask_input``
still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .models import PromptModelSpec
from .runtime import SessionInfo, create_session, run_options


@dataclass
class ImageEmbedding:
    """Encoder output for one image. Opaque to callers except for the size it was made for."""

    height: int
    width: int
    features: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class PromptPrediction:
    logits: np.ndarray  # float32 K x r x r, whole image stretched to a square; > 0 means "selected"
    scores: np.ndarray  # float32 K, the model's own quality estimate per candidate
    single_index: int = 0  # candidate to use when several clicks are given
    multi_indices: tuple[int, ...] = ()  # candidates to choose from for a single click

    def best(self, n_clicks: int) -> int:
        """Index of the candidate to use. SAM's advice: with one click the prompt is ambiguous
        (shirt vs. whole person), so take the best-scoring multimask candidate; with more
        clicks the single-mask head is better."""
        if n_clicks <= 1 and self.multi_indices:
            idx = list(self.multi_indices)
            return idx[int(np.argmax(self.scores[idx]))]
        return self.single_index


class PromptSegmenter(Protocol):
    @property
    def name(self) -> str: ...

    def encode(self, rgb: np.ndarray) -> ImageEmbedding:
        """``rgb``: HxWx3 uint8, the whole image."""
        ...

    def decode(
        self,
        embedding: ImageEmbedding,
        points: np.ndarray,
        labels: np.ndarray,
        box: tuple[float, float, float, float] | None = None,
        mask_input: np.ndarray | None = None,
    ) -> PromptPrediction:
        """``points``: N x 2 (x, y) image pixels; ``labels``: N ints, 1 = add here, 0 = not this.
        ``box``: optional (x0, y0, x1, y1). ``mask_input``: previous low-res logits (r x r) to
        refine iteratively, as SAM does between clicks."""
        ...


class OnnxSamSegmenter:
    """SAM 2 / 2.1 exported as encoder + decoder ONNX (see module docstring for the contract)."""

    def __init__(self, spec: PromptModelSpec, encoder_file: Path, decoder_file: Path, device: str = "auto"):
        self.spec = spec
        self.encoder, self.info = create_session(encoder_file, device)
        self.decoder, self.decoder_info = create_session(decoder_file, device)
        enc_in = self.encoder.get_inputs()[0]
        self._enc_input = enc_in.name
        self._enc_dtype = _np_dtype(enc_in.type)
        self._enc_outputs = [o.name for o in self.encoder.get_outputs()]
        self._dec_inputs = {i.name: _np_dtype(i.type) for i in self.decoder.get_inputs()}
        self._dec_outputs = [o.name for o in self.decoder.get_outputs()]
        self._enc_run_opts = run_options(self.info)  # free the encoder's activations after each image

    @property
    def name(self) -> str:
        return self.spec.id

    @property
    def session_info(self) -> SessionInfo:
        return self.info

    def encode(self, rgb: np.ndarray) -> ImageEmbedding:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("prompt segmenter input must be HxWx3 uint8")
        h, w = rgb.shape[:2]
        s = self.spec.input_size
        interp = cv2.INTER_AREA if h * w > s * s else cv2.INTER_CUBIC
        x = cv2.resize(rgb, (s, s), interpolation=interp).astype(np.float32) / 255.0
        x = (x - np.asarray(self.spec.mean, np.float32)) / np.asarray(self.spec.std, np.float32)
        tensor = np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(self._enc_dtype)
        outs = self.encoder.run(None, {self._enc_input: tensor}, self._enc_run_opts)
        return ImageEmbedding(height=h, width=w, features=dict(zip(self._enc_outputs, outs)))

    def decode(self, embedding, points, labels, box=None, mask_input=None) -> PromptPrediction:
        coords, labs = to_model_prompt(
            embedding.height, embedding.width, self.spec.input_size, points, labels, box
        )
        r = self.spec.low_res_size
        feeds: dict[str, np.ndarray] = {
            "point_coords": coords[None],
            "point_labels": labs[None],
        }
        if mask_input is None:
            feeds["mask_input"] = np.zeros((1, 1, r, r), np.float32)
            feeds["has_mask_input"] = np.zeros((1,), np.float32)
        else:
            feeds["mask_input"] = np.asarray(mask_input, np.float32).reshape(1, 1, r, r)
            feeds["has_mask_input"] = np.ones((1,), np.float32)
        feeds.update(embedding.features)

        missing = [n for n in self._dec_inputs if n not in feeds]
        if missing:
            raise RuntimeError(f"{self.spec.id} decoder needs inputs {missing} this app does not provide")
        run_feeds = {n: np.asarray(feeds[n]).astype(dt, copy=False) for n, dt in self._dec_inputs.items()}
        outs = dict(zip(self._dec_outputs, self.decoder.run(None, run_feeds)))
        masks = np.asarray(outs.get("masks", next(iter(outs.values()))), np.float32)
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        scores = np.asarray(outs.get("iou_predictions", np.ones(masks.shape[0])), np.float32).reshape(-1)
        k = masks.shape[0]
        multi = tuple(range(1, k)) if k == 4 else tuple(range(k)) if k == 3 else ()
        return PromptPrediction(logits=masks, scores=scores, single_index=0, multi_indices=multi)

    def warmup(self) -> None:
        s = self.spec.input_size
        emb = self.encode(np.zeros((s, s, 3), np.uint8))
        self.decode(emb, np.array([[s / 2, s / 2]]), np.array([1]))


def to_model_prompt(
    height: int,
    width: int,
    input_size: int,
    points: np.ndarray,
    labels: np.ndarray,
    box: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Image-pixel clicks -> SAM prompt arrays in the encoder's square input space.

    SAM indexes points by pixel (it adds 0.5 itself), so scene coordinates lose their
    half-pixel first. Without a box, SAM expects a trailing padding point labelled -1.
    """
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    labs = np.asarray(labels, np.float32).reshape(-1)
    if pts.shape[0] != labs.shape[0]:
        raise ValueError("points and labels must have the same length")
    scale = np.array([input_size / width, input_size / height], np.float32)
    coords = (pts - 0.5) * scale
    if box is not None:
        x0, y0, x1, y1 = box
        corners = (np.array([[min(x0, x1), min(y0, y1)], [max(x0, x1), max(y0, y1)]], np.float32) - 0.5) * scale
        coords = np.concatenate([corners, coords])
        labs = np.concatenate([np.array([2, 3], np.float32), labs])
    else:
        coords = np.concatenate([coords, np.zeros((1, 2), np.float32)])
        labs = np.concatenate([labs, np.array([-1], np.float32)])
    return coords.astype(np.float32), labs.astype(np.float32)


def _np_dtype(onnx_type: str):
    return np.float16 if "float16" in onnx_type else np.float32
