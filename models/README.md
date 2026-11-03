Model weights live here (or in `%CUTOUT_MODELS_DIR%`). They are not committed.

Populate with `python scripts/fetch_models.py` (downloadable ONNX files) and
`python scripts/export_birefnet_onnx.py ...` (models that ship only as PyTorch weights) and
`python scripts/export_sam2_onnx.py ...` (the click-to-select encoder/decoder pairs).
See ARCHITECTURE.md, "Models".
