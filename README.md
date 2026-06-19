<div align="center">

<h1>RVC GUI<br><br>
  
For audio file inference only

  <br>

  

</div>

  

 

  

## GUI

![GUI](https://github.com/Tiger14n/RVC-GUI/raw/main/docs/GUI.JPG)
 <br><br>
  
## Direct setup for Windows users
## [Windows-pkg](https://github.com/Tiger14n/RVC-GUI/releases/tag/Windows-pkg)
  
<br><br>
## Preparing the environment

This project has been migrated to [`uv`](https://docs.astral.sh/uv/) for dependency management and now ships a pinned, reproducible environment based on **Python 3.10** and **PyTorch 2.12.1 + CUDA 13.2**.

### Quick start (recommended)

1. **Install `uv`** (one-time, per machine):

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and sync** the project — `uv` will automatically download the correct CPython 3.10 interpreter, resolve the lockfile, and install every dependency (including the CUDA 13.2-enabled `torch` / `torchvision` wheels from the dedicated PyTorch index):

   ```bash
   git clone https://github.com/FranzFischer78/RVC-GUI.git
   cd RVC-GUI
   uv sync
   ```

3. **Download [`hubert_base.pt`](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)** and place it in the project root folder.

4. **Launch the GUI** through `uv` (no manual venv activation needed):

   ```bash
   uv run python rvcgui.py
   ```

   On Windows you can also run `RVC-GUI.bat` (it now detects the `uv`-managed `.venv` automatically).

### What `uv sync` does under the hood

- Creates a project-local `.venv` with CPython 3.10 (downloaded on demand by `uv`).
- Resolves dependencies against the lockfile `uv.lock` — every contributor gets byte-identical versions.
- Pulls `torch==2.12.1+cu132` and `torchvision==0.27.1+cu132` from the dedicated PyTorch index `https://download.pytorch.org/whl/cu132` (configured in `pyproject.toml` under `[[tool.uv.index]]` and `[tool.uv.sources]`).
- Installs all other dependencies from PyPI.

### Verifying the CUDA build

After `uv sync`, verify that the GPU-enabled torch build was installed:

```bash
uv run python -c "import torch; print(torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

Expected output on a CUDA 13.2-capable host:

```
2.12.1+cu132 CUDA: 13.2 available: True
```

If `torch.cuda.is_available()` returns `False`, make sure your NVIDIA driver supports CUDA 13.2 (driver version >= 580.x is recommended) and that no CPU-only `torch` wheel was accidentally installed.

### Switching to a different CUDA variant

To use a different CUDA build (e.g. `cu128`), edit `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"                    # rename
url = "https://download.pytorch.org/whl/cu128"   # change URL

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Then update the `torch` / `torchvision` version pins to a version published on that index and re-run `uv lock && uv sync`.

### CPU-only install (no NVIDIA GPU)

If you do not have an NVIDIA GPU, install the CPU-only torch wheels from PyPI:

1. Comment out the `[[tool.uv.index]]` block named `pytorch-cu132` and the `[tool.uv.sources]` block in `pyproject.toml`.
2. Change the `torch` / `torchvision` version pins to plain PyPI versions (e.g. `torch==2.12.1` without the `+cu132` suffix).
3. Run `uv lock && uv sync`.

### Apple silicon Macs

For Apple Silicon (MPS) support, install the nightly CPU torch build instead:

```bash
# After commenting out the cu132 index in pyproject.toml as above
uv pip install --pre torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/nightly/cpu
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python rvcgui.py
```

### PyTorch & CUDA notes

- **Why is `torchaudio` not a hard dependency?** RVC-GUI's source code never imports `torchaudio` — audio I/O goes through `soundfile`, `librosa`, and `scipy.io.wavfile`. The cu132 PyTorch index does not publish x86_64 / Windows `torchaudio` wheels, and PyPI's `torchaudio` wheels are tightly pinned to specific torch versions, so making `torchaudio` a hard dependency would either break the cu132 build or force a torch downgrade.
- **`torchcrepe` compatibility shim:** `torchcrepe` (used for the CREPE f0 estimator) imports `torchaudio` at module load time. Because no `torchaudio` wheel matches the cu132 build, RVC-GUI ships a small shim module (`_torchcrepe_compat.py`) that stubs out `torchaudio` and re-implements `torchcrepe.load.audio` using `soundfile`. This shim is imported automatically at the top of `rvcgui.py` — no user action required.
- **`weights_only=False`:** Under torch 2.6+, `torch.load` defaults to `weights_only=True`, which rejects RVC `.pth` checkpoints that contain arbitrary Python objects. All `torch.load` calls in this repo have been updated to pass `weights_only=False` explicitly.

### Legacy install (without `uv`)

The original `requirements.txt` is preserved for users who cannot adopt `uv`. The legacy flow still works:

```bash
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python rvcgui.py
```

Note, however, that the legacy flow is **not** the recommended path and may diverge from the versions pinned in `uv.lock`. The `uv` flow above is the only supported install path going forward.

<br>

# Loading models
use the import button to import a model from a zip file, 
* The .zip must contain the ".pth" weight file. 
* The .zip is recommended to contain the feature retrieval files ".index"

Or place the model manually in root/models
```
models
├───Person1
│   ├───xxxx.pth
│   ├───xxxx.index
│   └───xxxx.npy
└───Person2
    ├───xxxx.pth
    ├───...
    └───...
````

<br>


<br> 

### How to get models?.
* Join the[ AI Hub](https://discord.gg/aihub) Discord 
* [Community Models on HuggingFace](https://huggingface.co/QuickWick/Music-AI-Voices/tree/main) by Wicked aka QuickWick

<br>

K7#4523
