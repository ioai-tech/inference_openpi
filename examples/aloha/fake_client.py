#!/usr/bin/env python3
"""Fake ALOHA client: random images + joints, no ROS2 / robot required."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference_openpi.client.observation import (
    CAMERA_KEYS,
    STATE_DIM,
    build_observation,
    random_aloha_images_hwc,
    random_aloha_state,
)
from inference_openpi.config import load_profile

logger = logging.getLogger(__name__)
DEFAULT_PROFILE = _REPO_ROOT / "configs" / "robots" / "aloha.yaml"


def _require_openpi_client():
    try:
        from openpi_client import action_chunk_broker
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise SystemExit(
            "openpi-client is required. Prefer the openpi venv:\n"
            "  openpi/.venv/bin/python examples/aloha/fake_client.py --port 18000"
        ) from exc
    return action_chunk_broker, websocket_client_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Send random ALOHA observations to a policy server")
    parser.add_argument("--host", default=None, help="Policy server host (default: profile server.host)")
    parser.add_argument("--port", type=int, default=None, help="Policy server port (default: profile server.port)")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--prompt", default="do something")
    parser.add_argument("--steps", type=int, default=3, help="How many server infer() calls to send")
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Use ActionChunkBroker (one server call per action_horizon steps). Default: one real request per step.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    profile = load_profile(args.profile)
    host = args.host or profile.server.host
    port = args.port if args.port is not None else profile.server.port

    action_chunk_broker, websocket_client_policy = _require_openpi_client()
    logger.info("Connecting to ws://%s:%s", host, port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    logger.info("Server metadata: %s", policy.get_server_metadata())
    infer = policy.infer
    if args.chunk:
        broker = action_chunk_broker.ActionChunkBroker(
            policy=policy,
            action_horizon=profile.policy.action_horizon,
        )
        infer = broker.infer
        logger.info("Using ActionChunkBroker action_horizon=%s", profile.policy.action_horizon)
    else:
        logger.info("Each step is a real WebSocket infer() request")

    rng = np.random.default_rng(args.seed)
    logger.info(
        "Fake obs: state=%s cameras=%s raw_hwc=240x320 resized=%s prompt=%r",
        STATE_DIM,
        CAMERA_KEYS,
        profile.policy.image_size,
        args.prompt,
    )

    ok = 0
    failed = 0
    latencies: list[float] = []
    for step in range(args.steps):
        observation = build_observation(
            random_aloha_state(rng),
            random_aloha_images_hwc(rng),
            args.prompt,
            image_size=profile.policy.image_size,
        )
        t0 = time.monotonic()
        try:
            result = infer(observation)
            elapsed_ms = (time.monotonic() - t0) * 1000
            actions = np.asarray(result["actions"])
            if actions.size == 0 or actions.shape[-1] != STATE_DIM:
                raise ValueError(f"bad actions shape {actions.shape}")
            ok += 1
            latencies.append(elapsed_ms)
            logger.info(
                "req=%s/%s infer_ms=%.1f actions.shape=%s action[0,:4]=%s",
                step + 1,
                args.steps,
                elapsed_ms,
                actions.shape,
                np.round(np.asarray(actions, dtype=np.float32).reshape(-1)[:4], 4),
            )
        except Exception as exc:
            failed += 1
            logger.exception("req=%s/%s FAILED: %s", step + 1, args.steps, exc)

    logger.info(
        "SUMMARY requested=%s returned=%s failed=%s latency_ms mean=%.1f p50=%.1f p95=%.1f",
        args.steps,
        ok,
        failed,
        float(np.mean(latencies)) if latencies else 0.0,
        float(np.quantile(latencies, 0.50)) if latencies else 0.0,
        float(np.quantile(latencies, 0.95)) if latencies else 0.0,
    )
    if ok != args.steps:
        raise SystemExit(f"Expected {args.steps} results, got {ok} (failed={failed})")
    logger.info("Fake client finished %s request(s) OK", ok)


if __name__ == "__main__":
    main()
