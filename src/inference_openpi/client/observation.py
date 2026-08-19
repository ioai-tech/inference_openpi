"""Build the ALOHA observation dict expected by AlohaInputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from inference_openpi.config import CAMERA_KEYS, STATE_DIM

try:
    from openpi_client import image_tools
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openpi-client is required on the robot machine. "
        "Install with: pip install -e openpi/packages/openpi-client"
    ) from exc


def _as_hwc_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = image_tools.convert_to_uint8(image)
    return np.ascontiguousarray(image)


def build_observation(
    state: np.ndarray,
    images_hwc: Mapping[str, np.ndarray],
    prompt: str,
    *,
    image_size: int = 224,
) -> dict:
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_arr.shape[0] != STATE_DIM:
        raise ValueError(f"state must have {STATE_DIM} dims, got {state_arr.shape}")

    images_chw: dict[str, np.ndarray] = {}
    for name in CAMERA_KEYS:
        if name not in images_hwc:
            raise KeyError(f"Missing camera {name!r}; need {CAMERA_KEYS}")
        resized = image_tools.resize_with_pad(_as_hwc_uint8(images_hwc[name]), image_size, image_size)
        images_chw[name] = np.transpose(resized, (2, 0, 1))

    observation = {
        "state": state_arr,
        "images": images_chw,
    }
    if prompt:
        observation["prompt"] = prompt
    return observation


def random_aloha_state(rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(-1.0, 1.0, size=(STATE_DIM,)).astype(np.float32)


def random_aloha_images_hwc(
    rng: np.random.Generator | None = None,
    *,
    height: int = 240,
    width: int = 320,
) -> dict[str, np.ndarray]:
    """Fake camera frames in HWC uint8, matching the ROS2 client before resize."""
    rng = np.random.default_rng() if rng is None else rng
    return {
        name: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) for name in CAMERA_KEYS
    }


def random_aloha_observation(prompt: str = "do something", *, image_size: int = 224) -> dict:
    rng = np.random.default_rng()
    return build_observation(
        random_aloha_state(rng),
        random_aloha_images_hwc(rng),
        prompt,
        image_size=image_size,
    )
