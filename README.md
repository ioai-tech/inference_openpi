# inference_openpi

Apache-2.0 wrapper around the [openpi](https://github.com/Physical-Intelligence/openpi) submodule. It serves a **local ALOHA-style** Orbax checkpoint over WebSocket and talks to a ROS2 robot client — **without modifying** `openpi/` sources.

Default overlay matches a fine-tuned checkpoint with:

- 14-D state / action (left 6 + left gripper + right 6 + right gripper)
- cameras: `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`
- `adapt_to_pi: false` (no upstream ALOHA joint-flip / gripper remap)
- norm stats at `assets/training_dataset/` (not the upstream `trossen` id)

Swap robots or cameras by copying [`configs/robots/aloha.yaml`](configs/robots/aloha.yaml).

## Layout

```
configs/robots/aloha.yaml     # only file you edit for a new robot / camera
src/inference_openpi/         # overlay TrainConfig, serve, ROS2 + fake client
examples/aloha/pi_client.py   # ROS2 client
examples/aloha/fake_client.py # random obs, no robot
scripts/bootstrap.sh          # submodule + uv (GPU) or pip (client)
scripts/serve.sh              # GPU / Jetson policy server
scripts/healthcheck.py
checkpoints/                  # put Orbax weights here (gitignored)
openpi/                       # git submodule, do not edit
```

## Clone

```bash
git clone --recurse-submodules https://github.com/ioai-tech/inference_openpi.git
cd inference_openpi
# if you already cloned without submodules:
# git submodule update --init openpi
```

## GPU server

```bash
# Optional, if GitHub / PyPI is slow:
# export https_proxy="http://127.0.0.1:7890"
# export http_proxy="http://127.0.0.1:7890"

./scripts/bootstrap.sh
./scripts/serve.sh --smoke          # load weights + one dummy infer, then exit
./scripts/serve.sh                  # default port 8000
# ./scripts/serve.sh --port 18000 --prompt "fold the cloth"
```

`serve.sh` picks a free GPU, sets JAX memory flags, and finds an Orbax dir under `checkpoints/`.

Do **not** use upstream `uv run scripts/serve_policy.py --env=ALOHA`: that turns on `adapt_to_pi=True` and looks up `assets/trossen`.

Health check (use the openpi venv, which already has `openpi-client`):

```bash
openpi/.venv/bin/python scripts/healthcheck.py --host 127.0.0.1 --port 8000 --infer
```

## Fake client (no ROS / robot)

Sends random 14-D joints and four camera images through the same observation packing as the real client. Each `--steps` value is one real `infer()` (not a local chunk slice).

```bash
openpi/.venv/bin/python examples/aloha/fake_client.py --host 127.0.0.1 --port 8000 --steps 3
```

One `infer()` returns an action chunk of shape `(horizon, 14)` (often ~0.4–1.0 s on a workstation GPU). For 10–50 Hz control, execute the chunk open-loop (`--chunk` or the ROS2 client’s `ActionChunkBroker`); do not expect 10 full model forwards per second on a V100. See the [π₀ paper Table I](https://arxiv.org/html/2410.24164v3) (≈73 ms on RTX 4090) and the [openpi DROID README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md) (0.5–1 s per chunk is normal for remote setups).

## ROS2 client

On the robot PC (no JAX required):

```bash
./scripts/bootstrap.sh --client
source /opt/ros/humble/setup.bash    # adjust distro

python examples/aloha/pi_client.py --host <GPU_SERVER_IP>   # dry-run
python examples/aloha/pi_client.py --host <GPU_SERVER_IP> --execute
```

## New robot / cameras

1. Copy `configs/robots/aloha.yaml`.
2. Set the four `sensor_msgs/Image` topics (`qos: best_effort` for RealSense).
3. Set `/joint_states` and the **14 joint names in training order**.
4. Set `command_type` to `joint_state` or `float64_multi_array`.
5. If motion looks like stacked deltas, set `policy.use_delta_joint_actions: false`.

Leave `adapt_to_pi` and `asset_id` alone unless the checkpoint convention changes.

## Jetson

- **Client only:** `./scripts/bootstrap.sh --client`, then point `--host` at the GPU server.
- **Server on Jetson:** `./scripts/serve.sh` detects `/etc/nv_tegra_release` and sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`. Override if needed:

  ```bash
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
  ./scripts/serve.sh
  ```

## License

Apache License 2.0, see [LICENSE](LICENSE). The `openpi/` submodule is also Apache-2.0; see [NOTICE](NOTICE).
