"""GPU-aware WebSocket policy server. Does not modify openpi/scripts/serve_policy.py."""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

from inference_openpi.checkpoint import resolve_checkpoint
from inference_openpi.config import load_profile
from inference_openpi.device import apply_inference_env, detect_accelerator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "configs" / "robots" / "aloha.yaml"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"

logger = logging.getLogger(__name__)


def _reachable_ips() -> list[str]:
    """Prefer real NIC addresses over Ubuntu's 127.0.1.1 hostname mapping."""
    found: list[str] = []
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if getattr(addr, "family", None) != socket.AF_INET:
                    continue
                ip = addr.address
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                if ip not in found:
                    found.append(ip)
    except Exception:
        pass
    if not found:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                found.append(sock.getsockname()[0])
        except OSError:
            found.append("127.0.0.1")
    return found or ["127.0.0.1"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local ALOHA OpenPI checkpoint over WebSocket")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Orbax checkpoint directory. Default: auto-discover under ./checkpoints",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help=f"Robot YAML profile (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument("--port", type=int, default=None, help="Override profile server.port")
    parser.add_argument("--prompt", type=str, default=None, help="Override profile policy.prompt")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load the checkpoint, run one dummy ALOHA infer, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = _parse_args()
    device = detect_accelerator()
    apply_inference_env(device)

    from inference_openpi.policy_overlay import create_aloha_overlay_config
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server

    profile = load_profile(args.profile)
    checkpoint = resolve_checkpoint(args.checkpoint, search_root=DEFAULT_CHECKPOINT_ROOT)
    prompt = args.prompt if args.prompt is not None else profile.policy.prompt
    prompt = prompt or "do something"
    port = args.port if args.port is not None else profile.server.port

    logger.info("Profile=%s checkpoint=%s port=%s prompt=%r", profile.name, checkpoint, port, prompt)
    logger.info(
        "Overlay adapt_to_pi=%s use_delta_joint_actions=%s asset_id=%s",
        profile.policy.adapt_to_pi,
        profile.policy.use_delta_joint_actions,
        profile.policy.asset_id,
    )

    train_config = create_aloha_overlay_config(profile)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=prompt,
    )

    if args.smoke:
        from inference_openpi.client.observation import random_aloha_observation

        result = policy.infer(random_aloha_observation(prompt))
        actions = result["actions"]
        logger.info("Smoke infer OK actions.shape=%s dtype=%s", getattr(actions, "shape", None), getattr(actions, "dtype", None))
        return

    hostname = socket.gethostname()
    reachable = _reachable_ips()
    logger.info("Creating server (host=%s ips=%s bind=%s:%s)", hostname, ",".join(reachable), args.host, port)
    logger.info("Health check: curl http://127.0.0.1:%s/healthz", port)
    logger.info("Client example: python examples/aloha/pi_client.py --host %s --port %s", reachable[0], port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
