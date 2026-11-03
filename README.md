# BackgroundRemoval

The best app for background removal: offline, GPU-accelerated, and lossless.
Drop in a photo of a person or an anime character and get a transparent PNG at
exactly the same resolution, with clean hair and soft edges.

Status: the core engine and a command-line tool are in place. The desktop UI,
touch-up tools (brush, click-to-select) and the Windows installer are next.
See [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start (development)

```
pip install -e ".[gpu,ui,dev]"
python scripts/fetch_models.py
python -m cutout photo.jpg                 # writes photo_cutout.png
python -m cutout art.png --subject anime
python -m cutout --list-models
python -m cutout.ui                        # editor window: open or drop an image
```
