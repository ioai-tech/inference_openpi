"""Read / publish 14-D ALOHA joint state via ROS2."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from inference_openpi.config import STATE_DIM

if TYPE_CHECKING:
    from inference_openpi.config import JointSettings


def _require_ros2():
    try:
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, Header
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ROS2 Python packages not found. Source your ROS2 distro first, e.g.\n"
            "  source /opt/ros/humble/setup.bash"
        ) from exc
    return JointState, Float64MultiArray, Header


class Ros2Robot:
    def __init__(self, node, settings: JointSettings) -> None:
        JointState, Float64MultiArray, _Header = _require_ros2()
        self._node = node
        self._settings = settings
        self._lock = threading.Lock()
        self._latest: tuple[float, np.ndarray] | None = None

        node.create_subscription(JointState, settings.state_topic, self._on_joint_state, 10)
        if settings.command_type == "joint_state":
            self._publisher = node.create_publisher(JointState, settings.command_topic, 10)
        else:
            self._publisher = node.create_publisher(Float64MultiArray, settings.command_topic, 10)
        node.get_logger().info(
            f"Joints state <- {settings.state_topic}; command -> {settings.command_topic} "
            f"({settings.command_type})"
        )

    def _on_joint_state(self, msg) -> None:
        name_to_pos = {name: float(pos) for name, pos in zip(msg.name, msg.position)}
        missing = [name for name in self._settings.names if name not in name_to_pos]
        if missing:
            self._node.get_logger().error(
                f"{self._settings.state_topic} missing joints {missing}. "
                f"Available: {list(msg.name)}"
            )
            return
        state = np.asarray([name_to_pos[name] for name in self._settings.names], dtype=np.float32)
        with self._lock:
            self._latest = (time.monotonic(), state)

    def latest_state(self, *, now: float | None = None, timeout_sec: float = 1.0) -> np.ndarray:
        now = time.monotonic() if now is None else now
        with self._lock:
            latest = self._latest
        if latest is None:
            raise TimeoutError(f"No JointState received on {self._settings.state_topic}")
        stamp, state = latest
        if now - stamp > timeout_sec:
            raise TimeoutError(
                f"Stale JointState on {self._settings.state_topic} "
                f"(age={now - stamp:.2f}s timeout={timeout_sec}s)"
            )
        if state.shape[0] != STATE_DIM:
            raise ValueError(f"Expected {STATE_DIM}-D state, got {state.shape}")
        return state.copy()

    def has_state(self) -> bool:
        with self._lock:
            return self._latest is not None

    def publish_action(self, action: np.ndarray) -> None:
        JointState, Float64MultiArray, Header = _require_ros2()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != STATE_DIM:
            raise ValueError(f"Action must have {STATE_DIM} dims, got {action.shape}")
        if self._settings.command_type == "joint_state":
            msg = JointState()
            msg.header = Header()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.name = list(self._settings.names)
            msg.position = [float(value) for value in action]
            self._publisher.publish(msg)
            return
        msg = Float64MultiArray()
        msg.data = [float(value) for value in action]
        self._publisher.publish(msg)
