#!/usr/bin/env bash
# Start the ALOHA policy server with GPU/Jetson defaults. Does not edit openpi/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${REPO_ROOT}/configs/robots/aloha.yaml"
CHECKPOINT=""
PORT=""
PROMPT=""
EXTRA=()

usage() {
  cat <<'EOF'
Usage: ./scripts/serve.sh [--checkpoint DIR] [--profile YAML] [--port N] [--prompt TEXT] [--smoke]

Auto-detects CUDA / Jetson, finds an Orbax checkpoint under ./checkpoints,
and serves it over WebSocket (default port 8000).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

resolve_path() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  if [[ "${path}" != /* ]]; then
    path="${REPO_ROOT}/${path}"
  fi
  (cd "$(dirname "${path}")" && echo "$(pwd)/$(basename "${path}")")
}

PROFILE="$(resolve_path "${PROFILE}")"
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(resolve_path "${CHECKPOINT}")"
fi

detect_accelerator() {
  if [[ -f /etc/nv_tegra_release ]]; then
    echo jetson
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo cuda
    return
  fi
  echo cpu
}

pick_cuda_device() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F',' '{
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        free = $3 - $2;
        if (free > best) { best = free; idx = $1 }
      }
      END { if (idx != "") print idx }'
}

DEVICE="$(detect_accelerator)"
echo "Detected accelerator: ${DEVICE}"

case "${DEVICE}" in
  jetson)
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
    ;;
  cuda)
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
    GPU="$(pick_cuda_device || true)"
    if [[ -n "${GPU}" ]]; then
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
      echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
    ;;
  cpu)
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    echo "WARNING: no GPU detected; CPU inference is slow and may run out of memory." >&2
    ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the GPU server. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/openpi" ]]; then
  echo "openpi submodule missing. Run: ./scripts/bootstrap.sh" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(uv run --with pyyaml python -m inference_openpi.serve --profile "${PROFILE}")
if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${PORT}" ]]; then
  CMD+=(--port "${PORT}")
fi
if [[ -n "${PROMPT}" ]]; then
  CMD+=(--prompt "${PROMPT}")
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  CMD+=("${EXTRA[@]}")
fi

echo "JAX_PLATFORMS=${JAX_PLATFORMS:-}"
echo "Running: ${CMD[*]}"
cd "${REPO_ROOT}/openpi"
exec "${CMD[@]}"
