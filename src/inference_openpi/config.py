"""Robot profile loaded from YAML. Defaults match the local ALOHA checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

CAMERA_KEYS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
STATE_DIM = 14
DEFAULT_ASSET_ID = "training_dataset"

DEFAULT_JOINT_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)


@dataclass(frozen=True)
class ServerSettings:
    """Client-side connection settings. The server always binds 0.0.0.0."""

    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class PolicySettings:
    prompt: str = ""
    adapt_to_pi: bool = False
    use_delta_joint_actions: bool = True
    asset_id: str = DEFAULT_ASSET_ID
    image_size: int = 224
    action_horizon: int = 25


@dataclass(frozen=True)
class ControlSettings:
    hz: float = 30.0
    ready_timeout_sec: float = 30.0


@dataclass(frozen=True)
class CameraSettings:
    topics: dict[str, str] = field(
        default_factory=lambda: {
            "cam_high": "/camera/high/color/image_raw",
            "cam_low": "/camera/low/color/image_raw",
            "cam_left_wrist": "/camera/left_wrist/color/image_raw",
            "cam_right_wrist": "/camera/right_wrist/color/image_raw",
        }
    )
    qos: str = "best_effort"
    timeout_sec: float = 1.0


@dataclass(frozen=True)
class JointSettings:
    state_topic: str = "/joint_states"
    command_topic: str = "/aloha/joint_command"
    command_type: str = "joint_state"
    names: tuple[str, ...] = DEFAULT_JOINT_NAMES


@dataclass(frozen=True)
class RobotProfile:
    name: str = "aloha"
    server: ServerSettings = field(default_factory=ServerSettings)
    policy: PolicySettings = field(default_factory=PolicySettings)
    control: ControlSettings = field(default_factory=ControlSettings)
    cameras: CameraSettings = field(default_factory=CameraSettings)
    joints: JointSettings = field(default_factory=JointSettings)

    def validate(self) -> None:
        missing = [key for key in CAMERA_KEYS if key not in self.cameras.topics]
        extra = [key for key in self.cameras.topics if key not in CAMERA_KEYS]
        if missing:
            raise ValueError(f"cameras.topics missing ALOHA keys: {missing}")
        if extra:
            raise ValueError(f"cameras.topics has unknown keys {extra}; expected {CAMERA_KEYS}")
        if self.cameras.qos not in ("best_effort", "reliable"):
            raise ValueError(f"cameras.qos must be best_effort or reliable, got {self.cameras.qos!r}")
        if self.joints.command_type not in ("joint_state", "float64_multi_array"):
            raise ValueError(
                "joints.command_type must be joint_state or float64_multi_array, "
                f"got {self.joints.command_type!r}"
            )
        if len(self.joints.names) != STATE_DIM:
            raise ValueError(f"joints.names must have {STATE_DIM} entries, got {len(self.joints.names)}")
        if self.policy.asset_id != DEFAULT_ASSET_ID:
            raise ValueError(
                f"policy.asset_id must be {DEFAULT_ASSET_ID!r} for this checkpoint, "
                f"got {self.policy.asset_id!r}"
            )
        if self.control.hz <= 0:
            raise ValueError("control.hz must be positive")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_raw() -> dict[str, Any]:
    return {
        "name": "aloha",
        "server": {"host": "127.0.0.1", "port": 8000},
        "policy": {
            "prompt": "",
            "adapt_to_pi": False,
            "use_delta_joint_actions": True,
            "asset_id": DEFAULT_ASSET_ID,
            "image_size": 224,
            "action_horizon": 25,
        },
        "control": {"hz": 30.0, "ready_timeout_sec": 30.0},
        "cameras": {
            "qos": "best_effort",
            "timeout_sec": 1.0,
            "topics": {
                "cam_high": "/camera/high/color/image_raw",
                "cam_low": "/camera/low/color/image_raw",
                "cam_left_wrist": "/camera/left_wrist/color/image_raw",
                "cam_right_wrist": "/camera/right_wrist/color/image_raw",
            },
        },
        "joints": {
            "state_topic": "/joint_states",
            "command_topic": "/aloha/joint_command",
            "command_type": "joint_state",
            "names": list(DEFAULT_JOINT_NAMES),
        },
    }


def profile_from_dict(raw: Mapping[str, Any]) -> RobotProfile:
    data = _merge(_default_raw(), raw)
    server = _as_dict(data.get("server"))
    policy = _as_dict(data.get("policy"))
    control = _as_dict(data.get("control"))
    cameras = _as_dict(data.get("cameras"))
    joints = _as_dict(data.get("joints"))
    profile = RobotProfile(
        name=str(data.get("name", "aloha")),
        server=ServerSettings(
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8000)),
        ),
        policy=PolicySettings(
            prompt=str(policy.get("prompt", "")),
            adapt_to_pi=bool(policy.get("adapt_to_pi", False)),
            use_delta_joint_actions=bool(policy.get("use_delta_joint_actions", True)),
            asset_id=str(policy.get("asset_id", DEFAULT_ASSET_ID)),
            image_size=int(policy.get("image_size", 224)),
            action_horizon=int(policy.get("action_horizon", 25)),
        ),
        control=ControlSettings(
            hz=float(control.get("hz", 30.0)),
            ready_timeout_sec=float(control.get("ready_timeout_sec", 30.0)),
        ),
        cameras=CameraSettings(
            topics={str(k): str(v) for k, v in _as_dict(cameras.get("topics")).items()},
            qos=str(cameras.get("qos", "best_effort")),
            timeout_sec=float(cameras.get("timeout_sec", 1.0)),
        ),
        joints=JointSettings(
            state_topic=str(joints.get("state_topic", "/joint_states")),
            command_topic=str(joints.get("command_topic", "/aloha/joint_command")),
            command_type=str(joints.get("command_type", "joint_state")),
            names=tuple(str(name) for name in joints.get("names", DEFAULT_JOINT_NAMES)),
        ),
    )
    profile.validate()
    return profile


def load_profile(path: str | Path) -> RobotProfile:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"Robot profile not found: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Profile {profile_path} must be a YAML mapping")
    return profile_from_dict(raw)
