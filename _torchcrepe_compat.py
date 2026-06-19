"""
Runtime compatibility shim for torchcrepe under torch+cu132.

Background
----------
RVC-GUI depends on `torchcrepe` for the CREPE f0 estimator. The upstream
`torchcrepe.load` module unconditionally `import torchaudio`s at module load
time, and `torchaudio`'s C++ extension refuses to load when its compiled CUDA
version does not match torch's CUDA version.

As of torch 2.12.1 there is **no** `torchaudio` wheel on either PyPI or the
`cu132` PyTorch index that was compiled against CUDA 13.2 — the closest
(`torchaudio==2.11.0` on PyPI) was compiled against CUDA 13.0 and raises:

    RuntimeError: Detected that PyTorch and TorchAudio were compiled with
    different CUDA versions. PyTorch has CUDA version 13.2 whereas TorchAudio
    has CUDA version 13.0.

RVC-GUI never calls `torchcrepe.load.audio()` (it uses `soundfile`,
`librosa`, and `scipy.io.wavfile` for all audio I/O), so the `torchaudio`
import inside `torchcrepe.load` is dead code from RVC-GUI's perspective.

This shim replaces `torchcrepe.load.audio` with a `soundfile`-based loader
and prevents the unconditional `import torchaudio` from running, so that
`import torchcrepe` works cleanly under torch+cu132.

Usage
-----
This module MUST be imported before any module that imports `torchcrepe`
(i.e. before `vc_infer_pipeline`). `rvcgui.py` does this at the very top
of the file.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch


def _install_shim() -> None:
    """Replace ``torchcrepe.load.audio`` with a soundfile-based loader and
    prevent ``import torchaudio`` from running inside ``torchcrepe.load``.

    This is idempotent — calling it more than once is a no-op.
    """
    # Step 1: insert a stub `torchaudio` module into ``sys.modules`` so that
    # the bare ``import torchaudio`` at the top of ``torchcrepe.load`` resolves
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


# Install the shim at import time so that any subsequent `import torchcrepe`
# (e.g. inside vc_infer_pipeline.py) picks up the patched loader.
_install_shim()
