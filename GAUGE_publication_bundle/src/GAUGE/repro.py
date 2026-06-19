from __future__ import annotations

import os
import random
from typing import Any

from .runtime import (
    STRICT_REPRO_CPU_THREADS,
    STABLE_FLOAT3_CPU_THREADS,
    THROUGHPUT_16C_FLOAT3_CPU_THREADS,
    configure_cpu_runtime,
)

import numpy as np
import torch

_TORCH_INTEROP_THREADS_CONFIGURED = False
RUNTIME_PROFILE_STABLE = "stable_float3"
RUNTIME_PROFILE_STRICT = "strict_repro"
RUNTIME_PROFILE_THROUGHPUT = "throughput_16c_float3"


def _configure_torch_threads(cpu_threads: int) -> None:
    global _TORCH_INTEROP_THREADS_CONFIGURED
    try:
        torch.set_num_threads(int(cpu_threads))
    except RuntimeError:
        pass
    if not _TORCH_INTEROP_THREADS_CONFIGURED:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _TORCH_INTEROP_THREADS_CONFIGURED = True


def _normalize_runtime_profile(profile: str | None, deterministic: bool | None) -> str:
    if deterministic is not None:
        return RUNTIME_PROFILE_STRICT if deterministic else RUNTIME_PROFILE_STABLE
    normalized = str(profile or RUNTIME_PROFILE_STABLE).strip().lower()
    if normalized not in {RUNTIME_PROFILE_STABLE, RUNTIME_PROFILE_STRICT, RUNTIME_PROFILE_THROUGHPUT}:
        raise ValueError(
            "Unsupported runtime profile: "
            f"{profile!r}. Use {RUNTIME_PROFILE_STABLE!r}, {RUNTIME_PROFILE_STRICT!r}, "
            f"or {RUNTIME_PROFILE_THROUGHPUT!r}."
        )
    return normalized


def default_runtime_profile(device: str | None) -> str:
    if device is not None and str(device).startswith("cuda"):
        return RUNTIME_PROFILE_THROUGHPUT
    return RUNTIME_PROFILE_STABLE


def _runtime_profile_config(profile: str) -> dict[str, Any]:
    if profile == RUNTIME_PROFILE_STRICT:
        return {
            "cpu_threads": STRICT_REPRO_CPU_THREADS,
            "deterministic": True,
            "allow_tf32": False,
            "matmul_precision": "highest",
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
        }
    if profile == RUNTIME_PROFILE_THROUGHPUT:
        return {
            "cpu_threads": THROUGHPUT_16C_FLOAT3_CPU_THREADS,
            "deterministic": False,
            "allow_tf32": True,
            "matmul_precision": "high",
            "cublas_workspace_config": "",
            "cudnn_benchmark": False,
        }
    return {
        "cpu_threads": STABLE_FLOAT3_CPU_THREADS,
        "deterministic": False,
        "allow_tf32": True,
        "matmul_precision": "high",
        "cublas_workspace_config": "",
        "cudnn_benchmark": False,
    }


def set_reproducible_runtime(
    seed: int,
    *,
    device: str | None = None,
    profile: str | None = None,
    deterministic: bool | None = None,
    cublas_workspace_config: str | None = None,
) -> dict[str, Any]:
    runtime_profile = _normalize_runtime_profile(profile or default_runtime_profile(device), deterministic)
    profile_cfg = _runtime_profile_config(runtime_profile)
    cpu_runtime = configure_cpu_runtime(int(profile_cfg["cpu_threads"]))
    _configure_torch_threads(int(cpu_runtime["cpu_threads"]))
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    deterministic_enabled = bool(profile_cfg["deterministic"])
    resolved_cublas_workspace = cublas_workspace_config if cublas_workspace_config is not None else str(profile_cfg["cublas_workspace_config"])
    if deterministic_enabled and resolved_cublas_workspace:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = resolved_cublas_workspace
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic_enabled:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = bool(profile_cfg["cudnn_benchmark"])
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(profile_cfg["cudnn_benchmark"])
    use_cuda = str(device).startswith("cuda") if device is not None else torch.cuda.is_available()
    tf32_enabled = False
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = bool(profile_cfg["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(profile_cfg["allow_tf32"])
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(str(profile_cfg["matmul_precision"]))
        tf32_enabled = bool(torch.backends.cuda.matmul.allow_tf32)
    return {
        "seed": int(seed),
        "runtime_profile": runtime_profile,
        "deterministic": deterministic_enabled,
        "cpu_runtime": cpu_runtime,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32": tf32_enabled,
        "precision": "float32",
    }
