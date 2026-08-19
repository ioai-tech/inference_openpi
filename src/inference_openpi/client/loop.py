"""ROS2 control loop: cameras + joints -> OpenPI WebSocket -> joint command."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from inference_openpi.client.cameras_ros2 import Ros2Cameras
from inference_openpi.client.observation import build_observation
from inference_openpi.client.robot_ros2 import Ros2Robot

if TYPE_CHECKING:
    from inference_openpi.config import RobotProfile

logger = logging.getLogger(__name__)


def _require_ros2():
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ROS2 Python packages not found. Source your ROS2 distro first, e.g.\n"
            "  source /opt/ros/humble/setup.bash"
        ) from exc
    return rclpy, SingleThreadedExecutor


def _wait_ready(cameras: Ros2Cameras, robot: Ros2Robot, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last_log = 0.0
    while time.monotonic() < deadline:
        have_cams = cameras.received_names()
        have_joints = robot.has_state()
        now = time.monotonic()
        if len(have_cams) == 4 and have_joints:
            logger.info("ROS2 inputs ready: cameras=%s joints=ok", have_cams)
            return
        if now - last_log > 2.0:
            logger.info("Waiting for ROS2 inputs: cameras=%s joints=%s", have_cams, have_joints)
            last_log = now
        time.sleep(0.05)
    raise TimeoutError(
        f"Timed out after {timeout_sec:.0f}s waiting for 4 cameras and /joint_states. "
        f"cameras_seen={cameras.received_names()} joints={robot.has_state()}. "
        "Check configs/robots/aloha.yaml topic names."
    )


def run_client(
    profile: RobotProfile,
    *,
    host: str,
    port: int,
    execute: bool,
    max_steps: int | None = None,
) -> None:
    try:
        from openpi_client import action_chunk_broker
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise SystemExit(
            "openpi-client is required. Install with:\n"
            "  pip install -e openpi/packages/openpi-client"
        ) from exc

    rclpy, SingleThreadedExecutor = _require_ros2()
    rclpy.init()
    node = rclpy.create_node("openpi_aloha_client")
    cameras = Ros2Cameras(node, profile.cameras)
    robot = Ros2Robot(node, profile.joints)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    try:
        _wait_ready(cameras, robot, profile.control.ready_timeout_sec)
        policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        logger.info("Server metadata: %s", policy.get_server_metadata())
        broker = action_chunk_broker.ActionChunkBroker(
            policy=policy,
            action_horizon=profile.policy.action_horizon,
        )
        if not execute:
            logger.warning("Dry-run: inferences only, no joint commands. Pass --execute to publish.")

        period = 1.0 / profile.control.hz
        step = 0
        while rclpy.ok():
            if max_steps is not None and step >= max_steps:
                break
            t0 = time.monotonic()
            try:
                images = cameras.latest(now=t0)
                state = robot.latest_state(now=t0, timeout_sec=profile.cameras.timeout_sec)
            except TimeoutError as exc:
                logger.warning("%s", exc)
                time.sleep(period)
                continue
            observation = build_observation(
                state,
                images,
                profile.policy.prompt,
                image_size=profile.policy.image_size,
            )
            result = broker.infer(observation)
            action = np.asarray(result["actions"], dtype=np.float32).reshape(-1)
            if execute:
                robot.publish_action(action)
            elif step % int(max(profile.control.hz, 1)) == 0:
                logger.info("dry-run step=%s action[:4]=%s", step, np.round(action[:4], 4))
            step += 1
            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
