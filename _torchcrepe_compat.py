"""
Runtime compatibility shim for torchcrepe + fairseq under torch+cu132.

Background
----------
RVC-GUI depends on `torchcrepe` and `fairseq`. Under modern PyTorch
(2.6+, including the torch 2.12.1+cu132 wheel used by this project), both
libraries break at import or load time:

1. **torchcrepe**: the upstream `torchcrepe.load` module unconditionally
   `import torchaudio`s at module load time, and `torchaudio`'s C++ extension
   refuses to load when its compiled CUDA version does not match torch's
   CUDA version. As of torch 2.12.1 there is **no** `torchaudio` wheel on
   either PyPI or the cu132 PyTorch index that was compiled against CUDA 13.2.
   RVC-GUI never calls `torchcrepe.load.audio()` (it uses `soundfile`,
   `librosa`, and `scipy.io.wavfile` for all audio I/O), so the `torchaudio`
   import inside `torchcrepe.load` is dead code from RVC-GUI's perspective.

2. **fairseq**: `fairseq.checkpoint_utils.load_checkpoint_to_cpu` calls
   `torch.load(f, map_location=...)` without `weights_only=False`. Under
   torch 2.6+ `torch.load` defaults to `weights_only=True`, which rejects
   HuBERT checkpoints (they pickle `fairseq.data.dictionary.Dictionary`
   instances) with `_pickle.UnpicklingError`. fairseq is unmaintained and
   has not been updated to pass `weights_only=False` explicitly. We monkey-
   patch the function at import time so HuBERT loads cleanly.

This shim:
- Stubs out `torchaudio` in `sys.modules` and re-implements
  `torchcrepe.load.audio` using `soundfile`.
- Wraps `fairseq.checkpoint_utils.load_checkpoint_to_cpu` so that
  `weights_only=False` is passed through to `torch.load`.
- Provides `ensure_hubert_base_pt()` which auto-downloads `hubert_base.pt`
  from HuggingFace if it is missing from the project root.

Usage
-----
This module MUST be imported before any module that imports `torchcrepe` or
`fairseq` (i.e. before `vc_infer_pipeline` and before
`from fairseq import checkpoint_utils`). `rvcgui.py` does this at the very
top of the file.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# HuggingFace URL for the HuBERT base checkpoint used by RVC.
# This is the canonical URL referenced in the original RVC README.
# ---------------------------------------------------------------------------
HUBERT_BASE_URL = (
    "https://huggingface.co/lj1995/VoiceConversionWebUI/"
    "resolve/main/hubert_base.pt"
)
HUBERT_BASE_FILENAME = "hubert_base.pt"


def _install_torchcrepe_shim() -> None:
    """Replace ``torchcrepe.load.audio`` with a soundfile-based loader and
    prevent ``import torchaudio`` from running inside ``torchcrepe.load``.

    This is idempotent — calling it more than once is a no-op.
    """
    # Step 1: insert a stub `torchaudio` module into ``sys.modules`` so that
    # the bare ``import torchaudio` at the top of ``torchcrepe.load`` resolves
    # to our stub instead of the real (CUDA-mismatched) extension build.
    if "torchaudio" not in sys.modules:
        stub = types.ModuleType("torchaudio")
        # ``torchcrepe.load`` only references ``torchaudio.load``, so a stub
        # that raises a clear error if anything else is touched is enough.
        def _stub_load(*_args, **_kwargs):  # pragma: no cover - defensive
            raise RuntimeError(
                "torchaudio.load() is not available in this build. RVC-GUI "
                "uses soundfile/librosa for audio I/O; torchcrepe.load.audio "
                "should not be called. If you see this error, please file a "
                "bug report."
            )

        stub.load = _stub_load  # type: ignore[attr-defined]
        # ``torchcrepe`` may access ``torchaudio.__version__`` for logging.
        stub.__version__ = "0.0.0-shim"  # type: ignore[attr-defined]
        sys.modules["torchaudio"] = stub

    # Step 2: import torchcrepe now that the stub is in place, then replace
    # its ``load.audio`` with a real soundfile-based implementation so that
    # any future caller (or test) gets a working loader instead of the stub.
    try:
        import torchcrepe
        import torchcrepe.load as _tc_load
    except Exception:  # pragma: no cover - import errors will surface elsewhere
        return

    if getattr(_tc_load, "_rvcgui_shim_installed", False):
        return

    def _soundfile_audio(filename: str):
        """Drop-in replacement for ``torchaudio.load`` using ``soundfile``.

        Returns ``(waveform, sample_rate)`` with the same shapes and dtypes
        as ``torchaudio.load``: waveform is a FloatTensor of shape
        ``(channels, num_samples)``, sample_rate is a Python int.
        """
        import soundfile as sf

        data, sr = sf.read(filename, always_2d=True, dtype="float32")
        # soundfile returns shape (num_samples, channels); torchaudio returns
        # (channels, num_samples), so transpose.
        waveform = torch.from_numpy(data).T
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sr

    _tc_load.audio = _soundfile_audio  # type: ignore[assignment]
    _tc_load._rvcgui_shim_installed = True  # type: ignore[attr-defined]


def _install_fairseq_shim() -> None:
    """Monkey-patch ``fairseq.checkpoint_utils.load_checkpoint_to_cpu`` so
    that it passes ``weights_only=False`` to ``torch.load``.

    Under torch 2.6+ the default for ``weights_only`` flipped from ``False``
    to ``True``, which breaks loading of HuBERT checkpoints (they pickle
    ``fairseq.data.dictionary.Dictionary`` instances). fairseq is unmaintained
    and has not been updated to pass ``weights_only=False`` explicitly, so we
    patch it here.

    The patch is idempotent — calling it more than once is a no-op.
    """
    try:
        import fairseq.checkpoint_utils as _fs_ckpt
    except Exception:  # pragma: no cover - import errors surface elsewhere
        return

    if getattr(_fs_ckpt, "_rvcgui_weights_only_patch_installed", False):
        return

    _original_load_checkpoint_to_cpu = _fs_ckpt.load_checkpoint_to_cpu

    def _patched_load_checkpoint_to_cpu(path, arg_overrides=None, **kwargs):
        # Temporarily wrap torch.load so that any call without an explicit
        # `weights_only` argument defaults to `weights_only=False` instead of
        # the torch 2.6+ default of `True`. We do this by intercepting
        # torch.load inside fairseq.checkpoint_utils only (not globally).
        _orig_torch_load = _fs_ckpt.torch.load

        def _torch_load_with_weights_only_false(*args, **kw):
            if "weights_only" not in kw:
                kw["weights_only"] = False
            return _orig_torch_load(*args, **kw)

        try:
            _fs_ckpt.torch.load = _torch_load_with_weights_only_false
            return _original_load_checkpoint_to_cpu(path, arg_overrides, **kwargs)
        finally:
            _fs_ckpt.torch.load = _orig_torch_load

    _fs_ckpt.load_checkpoint_to_cpu = _patched_load_checkpoint_to_cpu
    _fs_ckpt._rvcgui_weights_only_patch_installed = True  # type: ignore[attr-defined]


def _download_file(url: str, dest: Path, *, chunk_size: int = 1 << 20) -> None:
    """Stream-download `url` to `dest` with a simple progress indicator.

    Uses urllib (stdlib) so we don't add a `requests` dependency.
    """
    import urllib.request
    import urllib.error

    # HuggingFace serves files behind a 302 redirect; urllib follows
    # redirects by default. We add a User-Agent so the CDN doesn't 403 us.
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "rvc-gui/0.2 (auto-downloader)"},
    )
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = resp.getheader("Content-Length")
            total_int = int(total) if total else None
            downloaded = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_int:
                        mb_done = downloaded / (1 << 20)
                        mb_total = total_int / (1 << 20)
                        pct = (downloaded / total_int) * 100 if total_int else 0
                        print(
                            f"\r  downloading {dest.name}: "
                            f"{mb_done:.1f} / {mb_total:.1f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        mb_done = downloaded / (1 << 20)
                        print(
                            f"\r  downloading {dest.name}: {mb_done:.1f} MB",
                            end="",
                            flush=True,
                        )
            print()  # newline after progress
        # Atomic-ish rename: .part -> final name
        tmp_path.replace(dest)
    except (urllib.error.URLError, OSError) as e:
        # Clean up partial download so we don't leave a corrupt file behind.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download {url} to {dest}: {e}. "
            f"Please download it manually and place it at {dest}."
        ) from e


def ensure_hubert_base_pt(base_dir: str | os.PathLike | None = None) -> Path:
    """Ensure that `hubert_base.pt` exists in the project root.

    If the file is missing, it is auto-downloaded from HuggingFace
    (`HUBERT_BASE_URL`). If the file is already present, this is a no-op.

    Parameters
    ----------
    base_dir : str or PathLike, optional
        The directory in which `hubert_base.pt` should live. Defaults to the
        current working directory (which is where RVC-GUI expects it).

    Returns
    -------
    Path
        The absolute path to `hubert_base.pt`.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    dest = base / HUBERT_BASE_FILENAME

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    # File missing (or zero-byte): download it.
    print(
        f"[rvc-gui] {HUBERT_BASE_FILENAME} not found in {base}. "
        f"Auto-downloading from HuggingFace..."
    )
    print(f"[rvc-gui]   URL: {HUBERT_BASE_URL}")
    _download_file(HUBERT_BASE_URL, dest)
    print(f"[rvc-gui]   saved to: {dest}")
    return dest


# ---------------------------------------------------------------------------
# Install all shims at import time so that any subsequent import of
# `torchcrepe` or `fairseq.checkpoint_utils` picks up the patched versions.
# ---------------------------------------------------------------------------
_install_torchcrepe_shim()
_install_fairseq_shim()
