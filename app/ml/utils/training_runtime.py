from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch


def smooth_binary_targets(
    target: torch.Tensor,
    target_min: float,
    target_max: float,
) -> torch.Tensor:
    return target * (target_max - target_min) + target_min


def resolve_amp_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return dtypes[name.lower()]


def configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def cuda_runtime_info(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "cuda_device_name": props.name,
        "cuda_capability": f"{props.major}.{props.minor}",
        "cuda_total_memory_gib": round(props.total_memory / (1024**3), 3),
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
    }


def set_seed(seed: int, seed_cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn
