"""Small, one-click updates for the installed Windows app.

The installed folder is split into *packs* (see ``packs.py``): the app code, the
third-party libraries, Qt, the NVIDIA GPU runtime, and one pack per AI model.
Every GitHub release publishes ``cutout-release.json`` (``manifest.py``), which
lists each pack's content hash and where to download it. An update downloads
only the packs whose hash changed, so a code-only update is tens of MB instead
of the full 3 GB (``client.py``). The same code finishes a fresh install: the
small ``Setup.exe`` leaves out the GPU runtime and the models, and the app
fetches them the first time it starts.

Nothing here goes online unless the user asks (Help > Check for Updates) or
turns on the startup check.
"""
