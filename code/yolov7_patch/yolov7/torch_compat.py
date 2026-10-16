"""Restore the pre-2.6 torch.load default so the 2022-era YOLOv7 code runs unchanged."""
import torch as _torch

if not getattr(_torch, "_wo_patched", False):
    _orig = _torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    _torch.load = _load
    _torch._wo_patched = True
