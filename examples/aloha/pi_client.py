#!/usr/bin/env python3
"""ALOHA ROS2 client for a remote OpenPI policy server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataclasses import replace

from inference_openpi.client.loop import run_client
from inference_openpi.config import load_profile

DEFAULT_PROFILE = _REPO_ROOT / "configs" / "robots" / "aloha.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an OpenPI ALOHA policy server from ROS2")
    parser.add_argument("--host", default=None, help="Policy server IP (overrides YAML server.host)")
    parser.add_argument("--port", type=int, default=None, help="Policy server port (overrides YAML)")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="Robot YAML profile")
    parser.add_argument("--prompt", default=None, help="Override YAML policy.prompt")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Publish joint commands. Default is dry-run (infer only).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N control steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    profile = load_profile(args.profile)
    if args.prompt is not None:
        profile = replace(profile, policy=replace(profile.policy, prompt=args.prompt))

    host = args.host or profile.server.host
    port = args.port if args.port is not None else profile.server.port
    logging.info("Connecting to ws://%s:%s profile=%s execute=%s", host, port, profile.name, args.execute)
    run_client(profile, host=host, port=port, execute=args.execute, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
