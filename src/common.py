"""Helpers shared by the training, evaluation and inference scripts."""
from __future__ import annotations

import torch

AMPERE_MAJOR = 8


def compute_dtype() -> torch.dtype:
    """bf16 on Ampere or newer, fp16 on older GPUs such as the T4, fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= AMPERE_MAJOR else torch.float16
