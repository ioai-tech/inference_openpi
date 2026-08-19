"""Check that the policy server is up; optionally run one dummy ALOHA infer."""

from __future__ import annotations

import argparse
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def check_healthz(host: str, port: int, timeout: float = 5.0) -> None:
    url = f"http://{host}:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8").strip()
            if response.status != 200:
                raise RuntimeError(f"{url} returned HTTP {response.status}")
            logger.info("%s -> %s", url, body or "OK")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Health check failed for {url}: {exc}") from exc


def check_infer(host: str, port: int, prompt: str) -> None:
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise SystemExit(
            "openpi-client is required for --infer. Install with:\n"
            "  pip install -e openpi/packages/openpi-client"
        ) from exc

    from inference_openpi.client.observation import random_aloha_observation

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    logger.info("Server metadata: %s", client.get_server_metadata())
    result = client.infer(random_aloha_observation(prompt))
    actions = result["actions"]
    logger.info("Infer OK actions.shape=%s dtype=%s", getattr(actions, "shape", None), getattr(actions, "dtype", None))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Health-check an OpenPI policy server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--infer", action="store_true", help="Send one random ALOHA observation")
    parser.add_argument("--prompt", default="do something")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    check_healthz(args.host, args.port)
    if args.infer:
        check_infer(args.host, args.port, args.prompt)
    logger.info("Health check passed")


if __name__ == "__main__":
    main()
