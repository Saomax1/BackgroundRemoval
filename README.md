# Cutout

Cutout removes the background from a photo of a person or an anime/cartoon
character and saves a transparent PNG. It runs entirely on your own PC, uses
your NVIDIA graphics card to go fast, and never shrinks your picture: what comes
out is exactly the same size as what went in.

**Where things stand:** the background-removal engine works from a command
window and in the editor window, which also has touch-up brushes and
click-to-select. A one-click `Setup.exe` is built automatically on GitHub: download it
from the [latest release](https://github.com/Saomax1/BackgroundRemoval/releases/latest)
(no sign-in needed). The first time Cutout starts it downloads its AI models
(about 3 GB, once). To update later, choose **Help > Check for Updates** in the
app: it downloads only what changed, usually a few tens of MB, and restarts.
Your settings are kept. Details are in [packaging/README.md](packaging/README.md).
If you'd rather run from source, use the steps below. You only do them once.

---

## What you need

* A Windows 10 or 11 PC with an NVIDIA graphics card.
* About 10 GB of free disk space during setup (most of it can be cleaned up afterwards).
* An internet connection **for setup only**. After that, Cutout works fully offline.

## One-time setup

You'll type a few commands into **PowerShell**. Open it by pressing the Windows
key, typing `PowerShell`, and pressing Enter. Copy each command below, paste it
with a right-click, and press Enter. Wait for each one to finish before the next.

### 1. Install Python

Download **Python 3.12** from <https://www.python.org/downloads/windows/> and
run the installer. On the first screen, **tick "Add python.exe to PATH"**, then
click *Install Now*.

Check it worked (close and reopen PowerShell first):

```powershell
py -3.12 --version
```

You should see `Python 3.12.something`.

### 2. Get Cutout

On the GitHub page, click the green **Code** button, then **Download ZIP**.
Unzip it to a simple place, for example `C:\Cutout`. Then go into that folder:

```powershell
cd C:\Cutout
```

(If you unzipped somewhere else, use that folder instead. It should contain
`README.md` and a `src` folder.)

### 3. Set up Cutout's own Python space

This keeps Cutout's pieces separate from anything else on your PC.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[gpu,ui]"
```

After the second line your prompt starts with `(.venv)`. **Whenever you open a
new PowerShell window to use Cutout, run `cd C:\Cutout` and
`.venv\Scripts\Activate.ps1` again.**

### 4. Download the AI models

```powershell
python scripts\fetch_models.py
```

This downloads two ready-made models (one for people, one for anime) and shows
a percentage as it goes. Each finished model prints a line ending in `ok`.

### 5. Build the two best models (optional, but recommended)

The highest-quality models for hair and soft edges only come in a format that
needs converting once. This step downloads about 1 to 2 GB of extra tools.

```powershell
pip install torch torchvision transformers timm kornia einops deform_conv2d_onnx_exporter
python scripts\export_birefnet_onnx.py birefnet-hr-matting
python scripts\export_birefnet_onnx.py toonout
```

Each one finishes with `wrote ...\models\...onnx`. It can take several minutes
and look frozen; that's normal.

If the `toonout` line fails, download the file `birefnet_finetuned_toonout.pth`
from the ToonOut page on Hugging Face (<https://huggingface.co/joelseytre/toonout>),
save it in `C:\Cutout`, and run:

```powershell
python scripts\export_birefnet_onnx.py toonout --weights birefnet_finetuned_toonout.pth
```

Once both are built you can free the space the converter used:

```powershell
pip uninstall -y torch torchvision transformers timm kornia einops deform_conv2d_onnx_exporter
```

> Skipping step 5 is fine. Cutout simply uses the models from step 4, which
> are good but a little less precise on fine hair.

### 6. Check everything is in place

```powershell
python -m cutout --list-models
```

Each model you set up shows `[installed]`.

## Using the editor window

```powershell
python -m cutout.ui
```

Drop a picture onto the window (or click **Open image…**). The background is
removed automatically. Pick **Person** or **Anime** at the top to match the
picture; switching re-runs the removal. Scroll to zoom, drag to move around,
and press **O** to compare with the original. **Save PNG** writes a
transparent PNG at the exact size of your picture, next to it as
`name_cutout.png` by default.

To fix anything the automatic step missed, pick a tool on the left:

| Tool | Key | What it does |
|---|---|---|
| **Keep** | K | Paint back parts of the subject that were removed. |
| **Erase** | E | Paint away leftover background. Hold **Alt** to switch between Keep and Erase. |
| **Refine Edge** | R | Brush over hair or fur so the edge follows the photo. |
| **Select** | S | Click a whole part (a sleeve, a gap between arms). Right-click excludes. **Enter** keeps the blue area, **Delete** removes it, **Esc** clears. |

The brush size is in pixels of your full-size picture, from **1 px** (exactly
one pixel, no feathering) up to 1000 px. Change it with the Size slider, the
number box, the **1 px** button, `[` and `]`, or Ctrl+scroll. Zoom in (scroll,
up to 6400%) to work on single pixels; from 800% a grid shows every pixel.
**Ctrl+Z** undoes each stroke or selection.

## Removing a background from the command line

```powershell
python -m cutout "C:\Pictures\me.jpg"
python -m cutout "C:\Pictures\character.png" --subject anime
```

The result is saved next to the original as `me_cutout.png`. The last line it
prints ends with `device=cuda` when your graphics card was used.

Tip: you can drag a picture from File Explorer into the PowerShell window to
paste its full path instead of typing it.

---

## If something goes wrong

**Sending a problem report.** Turn on **Help > Debug Mode** (it stays on until
you turn it off), or start Cutout from the **Cutout (debug mode)** Start menu
shortcut. Do the thing that failed again, then choose **Help > Copy Debug
Report** and paste it into the chat. Error messages also have a **Copy Report**
button, and if Cutout closed unexpectedly it offers a report the next time it
starts. The report holds your PC and graphics card details, which model ran,
memory use, and the recent log. Your user folder name is replaced with
`%USERPROFILE%`. Log files are kept in `%LOCALAPPDATA%\Cutout\logs`
(**Help > Open Log Folder**).

**"running ... on CPU (this will be slow)" or `device=cpu`**
Cutout couldn't use your graphics card. It still works, just slower. To fix it:

1. Update your NVIDIA driver from <https://www.nvidia.com/Download/index.aspx>
   (or the NVIDIA app). Version 528 or newer is needed. Restart the PC afterwards.
2. Check Windows can see the card: run `nvidia-smi`. It should list your GPU.
3. Make sure only the GPU version of the AI runtime is installed:
   ```powershell
   pip uninstall -y onnxruntime onnxruntime-gpu
   pip install -e ".[gpu,ui]"
   ```
4. To see the exact reason, force the graphics card and read the error:
   `python -m cutout "C:\Pictures\me.jpg" --device cuda`

**"Activate.ps1 cannot be loaded because running scripts is disabled"**
Run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**"'py' is not recognized" or "'python' is not recognized"**
Python isn't on the PATH. Re-run the Python installer, choose *Modify*, and make
sure "Add Python to environment variables" is ticked. Then reopen PowerShell.

**"DLL load failed" when Cutout starts**
Install the Microsoft Visual C++ Redistributable (x64) from
<https://aka.ms/vs/17/release/vc_redist.x64.exe>, then restart PowerShell.

**"out of memory" errors**
Cutout keeps one model on the graphics card at a time and unloads it when you
switch between Person and Anime. If the card still runs out (other programs, such
as games or video editors, are using it), Cutout retries once and then makes that
cutout on the processor instead, which is slower but gives the same result. The
photo's size doesn't matter: the models always run at a fixed size. If it keeps
happening, the smaller people model needs much less graphics memory:
`python -m cutout photo.jpg --model birefnet-general`

**"no person model installed" (or "no anime model installed")**
The models aren't in the `models` folder yet. Re-run step 4, and check with
`python -m cutout --list-models`.

**A download in step 4 stopped halfway**
Just run `python scripts\fetch_models.py` again. Half-finished files are
thrown away and started fresh.

---

## For developers

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the engine is put together.
Install with `pip install -e ".[gpu,ui,dev]"` (or `".[cpu,ui,dev]"` without an NVIDIA
card) and run the tests with `pytest`. The tests need no models or GPU.
