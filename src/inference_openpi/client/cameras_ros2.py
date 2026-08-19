"""Subscribe to ALOHA camera topics via ROS2."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from inference_openpi.config import CAMERA_KEYS

if TYPE_CHECKING:
    from inference_openpi.config import CameraSettings


def _require_ros2():
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ROS2 Python packages not found. Source your ROS2 distro first, e.g.\n"
            "  source /opt/ros/humble/setup.bash"
        ) from exc
    return rclpy, CvBridge, HistoryPolicy, QoSProfile, ReliabilityPolicy, Image


class Ros2Cameras:
    def __init__(self, node, settings: CameraSettings) -> None:
        _, CvBridge, HistoryPolicy, QoSProfile, ReliabilityPolicy, Image = _require_ros2()
        self._node = node
        self._settings = settings
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[float, np.ndarray]] = {}

        reliability = (
            ReliabilityPolicy.BEST_EFFORT if settings.qos == "best_effort" else ReliabilityPolicy.RELIABLE
        )
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=reliability)

        for name in CAMERA_KEYS:
            topic = settings.topics[name]
            node.create_subscription(
                Image,
                topic,
                lambda msg, camera=name: self._on_image(camera, msg),
                qos,
            )
            node.get_logger().info(f"Camera {name} <- {topic} (qos={settings.qos})")

    def _on_image(self, name: str, msg) -> None:
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        with self._lock:
            self._latest[name] = (time.monotonic(), np.asarray(image))

    def latest(self, *, now: float | None = None, require_fresh: bool = True) -> dict[str, np.ndarray]:
        now = time.monotonic() if now is None else now
        with self._lock:
            snapshot = dict(self._latest)
        missing = [name for name in CAMERA_KEYS if name not in snapshot]
        if missing:
            raise TimeoutError(f"No frames yet for cameras: {missing}")
        stale = [
            name
            for name, (stamp, _) in snapshot.items()
            if require_fresh and now - stamp > self._settings.timeout_sec
        ]
        if stale:
            raise TimeoutError(
                f"Stale cameras {stale} (timeout={self._settings.timeout_sec}s). "
                "Check topic names, QoS, and that the camera drivers are publishing."
            )
        return {name: image.copy() for name, (_, image) in snapshot.items()}

    def received_names(self) -> list[str]:
        with self._lock:
            return sorted(self._latest)
