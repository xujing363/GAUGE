from __future__ import annotations

import gc
import os
from typing import Any

import torch

STABLE_FLOAT3_CPU_THREADS = 8
STRICT_REPRO_CPU_THREADS = 16
THROUGHPUT_16C_FLOAT3_CPU_THREADS = 16
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _parse_positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(1, parsed)


def get_cpu_thread_limit(default_threads: int = STABLE_FLOAT3_CPU_THREADS) -> int:
    return _parse_positive_int(os.environ.get("BASLIN_CPU_THREADS"), default_threads)


def configure_cpu_runtime(default_threads: int = STABLE_FLOAT3_CPU_THREADS, *, force: bool = True) -> dict[str, Any]:
    threads = get_cpu_thread_limit(default_threads)
    for name in THREAD_ENV_VARS:
        if force:
            os.environ[name] = str(threads)
        else:
            os.environ.setdefault(name, str(threads))
    if force:
        os.environ["NUMEXPR_MAX_THREADS"] = str(threads)
    else:
        os.environ.setdefault("NUMEXPR_MAX_THREADS", str(threads))
    return {
        "cpu_threads": threads,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
    }


def release_gpu_memory(*objects: Any, empty_cache: bool = True, collect_garbage: bool = True) -> None:
    for obj in objects:
        if obj is None:
            continue
        if hasattr(obj, "train_tensor_cache") and hasattr(obj, "eval_tensor_cache"):
            for name in (
                "tensor_banks",
                "grouped_sampler_index",
                "train_tensor_cache",
                "val_tensor_cache",
                "eval_tensor_cache",
                "eval_kg_payload_cache",
                "kg_drug_idx_bank_cache",
                "pairwise_workspace",
            ):
                if hasattr(obj, name):
                    current = getattr(obj, name)
                    if hasattr(current, "clear") and callable(getattr(current, "clear")):
                        try:
                            current.clear()
                        except TypeError:
                            pass
                    setattr(obj, name, None)
            continue
        if hasattr(obj, "clear") and callable(getattr(obj, "clear")):
            try:
                obj.clear()
            except TypeError:
                pass
    if collect_garbage:
        gc.collect()
    if empty_cache and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
