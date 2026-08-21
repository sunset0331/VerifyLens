"""
verifylens_kaggle/hardware.py
-------------------------------
Hardware detection and reporting for Kaggle GPU environment.

Detects:
  - CUDA availability and GPU name
  - GPU total memory
  - CPU info
  - System RAM

Fails clearly (SystemExit) if CUDA is unavailable, since running the
200-sample VLM benchmark on CPU is not feasible.
"""

from __future__ import annotations

import platform
import sys
from typing import Dict, Any


def detect_hardware(require_cuda: bool = True) -> Dict[str, Any]:
    """
    Detect hardware environment and print a formatted summary.

    Parameters
    ----------
    require_cuda : bool
        If True (default), exits with an informative error if CUDA is
        not available. The heavy VLM benchmark MUST run on GPU.

    Returns
    -------
    dict with keys: cuda_available, gpu_name, gpu_memory_gb,
                    gpu_memory_total_bytes, cpu, ram_gb
    """
    info: Dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "gpu_memory_total_bytes": None,
        "gpu_count": 0,
        "cuda_version": None,
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # ── CPU / RAM ────────────────────────────────────────────────────────────
    try:
        import psutil
        ram = psutil.virtual_memory()
        info["ram_gb"] = round(ram.total / (1024 ** 3), 1)
    except ImportError:
        # psutil may not be installed — try /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024 ** 2), 1)
                        break
        except Exception:
            info["ram_gb"] = "unknown"

    # ── CUDA / GPU ───────────────────────────────────────────────────────────
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()

            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_total_bytes"] = props.total_memory
            info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        else:
            info["cuda_available"] = False

    except ImportError:
        info["cuda_available"] = False
        info["gpu_name"] = "torch not installed"

    # ── Print formatted block ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HARDWARE DETECTION")
    print("=" * 60)
    print(f"  CUDA available : {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"  CUDA version   : {info['cuda_version']}")
        print(f"  GPU count      : {info['gpu_count']}")
        print(f"  GPU name       : {info['gpu_name']}")
        print(f"  GPU memory     : {info['gpu_memory_gb']} GB")
    else:
        print(f"  GPU            : not available")
    print(f"  CPU            : {info['cpu']}")
    print(f"  RAM            : {info['ram_gb']} GB")
    print(f"  Python         : {info['python_version']}")
    print(f"  Platform       : {info['platform']}")
    print("=" * 60 + "\n")

    # ── Fail clearly if no CUDA ──────────────────────────────────────────────
    if require_cuda and not info["cuda_available"]:
        print("ERROR: CUDA is not available on this machine.")
        print()
        print("This benchmark runs heavy VLM and OCR+LLM inference that")
        print("requires a CUDA GPU. Running on CPU would be extremely slow")
        print("and is not supported.")
        print()
        print("If you are on Kaggle, ensure:")
        print("  Settings > Accelerator > GPU T4 x2  (or P100/V100/A100)")
        print()
        print("The benchmark will not proceed without CUDA.")
        sys.exit(1)

    return info


def reset_peak_memory_stats() -> None:
    """Reset CUDA peak memory tracking before a benchmark run."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_memory_stats() -> Dict[str, Any]:
    """
    Return current CUDA memory allocation stats.
    Safe to call even if torch is not imported yet.
    """
    stats: Dict[str, Any] = {
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "current_allocated_gb": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            stats["peak_allocated_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 3
            )
            stats["peak_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / (1024 ** 3), 3
            )
            stats["current_allocated_gb"] = round(
                torch.cuda.memory_allocated() / (1024 ** 3), 3
            )
    except Exception:
        pass
    return stats
