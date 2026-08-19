#!/usr/bin/env bash
# One-time setup for the GPU server and/or the ROS2 client machine.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./scripts/bootstrap.sh [--client]

  (default)  Init openpi submodule and sync the GPU server uv environment.
  --client   Install lightweight client deps only (no JAX). Use on the robot / Jetson client.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --client)
      CLIENT_ONLY=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"

install_client() {
  echo "Installing client package + openpi-client (no JAX)..."
  "${PYTHON}" -m pip install -e "${REPO_ROOT}"
  "${PYTHON}" -m pip install -e "${REPO_ROOT}/openpi/packages/openpi-client"
  echo "Client Python deps installed. Source ROS2 before running examples/aloha/pi_client.py"
}

if [[ "${CLIENT_ONLY}" -eq 1 ]]; then
  install_client
  exit 0
fi

if [[ ! -f "${REPO_ROOT}/.gitmodules" ]]; then
  echo "Not a git checkout? .gitmodules missing in ${REPO_ROOT}" >&2
  exit 1
fi

echo "Updating openpi submodule..."
git -C "${REPO_ROOT}" submodule update --init openpi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required for the GPU server. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ -n "${https_proxy:-}${HTTPS_PROXY:-}" ]]; then
  echo "Using proxy: ${https_proxy:-${HTTPS_PROXY}}"
  export http_proxy="${http_proxy:-${HTTP_PROXY:-${https_proxy:-${HTTPS_PROXY}}}}"
  export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
  export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
  export ALL_PROXY="${ALL_PROXY:-${https_proxy}}"
  export GIT_CONFIG_COUNT="${GIT_CONFIG_COUNT:-1}"
  export GIT_CONFIG_KEY_0="${GIT_CONFIG_KEY_0:-http.proxy}"
  export GIT_CONFIG_VALUE_0="${GIT_CONFIG_VALUE_0:-${https_proxy}}"
fi

echo "Syncing openpi uv environment (GIT_LFS_SKIP_SMUDGE=1)..."
(
  cd "${REPO_ROOT}/openpi"
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
)

install_client
echo "Bootstrap complete. Next: ./scripts/serve.sh"
