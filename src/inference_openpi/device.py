"""Detect CUDA / Jetson and set JAX runtime env before importing JAX."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TEGRA_RELEASE = Path("/etc/nv_tegra_release")


def detect_accelerator() -> str:
    """Return 'jetson', 'cuda', or 'cpu'."""
    if TEGRA_RELEASE.is_file():
        return "jetson"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            subprocess.run(
                [nvidia_smi],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return "cuda"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            logger.warning("nvidia-smi is present but failed; falling back to CPU")
    return "cpu"


def _parse_nvidia_query(output: str) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows


def pick_cuda_device() -> str | None:
    """Pick a visible GPU with the most free memory. Honors existing CUDA_VISIBLE_DEVICES."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in ("", "-1"):
        return os.environ["CUDA_VISIBLE_DEVICES"]

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None

    rows = _parse_nvidia_query(result.stdout)
    if not rows:
        return None
    index, used, total = max(rows, key=lambda row: row[2] - row[1])
    free = total - used
    if free < 4096:
        logger.warning("Best GPU %s has only %s MiB free", index, free)
    return str(index)


def apply_inference_env(device: str | None = None) -> str:
    """Set JAX GPU flags. Must run before importing jax / openpi."""
    device = device or detect_accelerator()
    if device in ("cuda", "jetson"):
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        selected = pick_cuda_device()
        if selected is not None:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", selected)
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        logger.warning("No GPU detected; JAX will use CPU (slow and may OOM)")

    if device == "jetson":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
    elif device == "cuda":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

    logger.info(
        "Accelerator=%s CUDA_VISIBLE_DEVICES=%s JAX_PLATFORMS=%s PREALLOCATE=%s MEM_FRACTION=%s",
        device,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("JAX_PLATFORMS"),
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    )
    return device
