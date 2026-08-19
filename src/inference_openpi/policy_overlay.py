"""ALOHA TrainConfig overlay. Imports openpi but never edits the submodule."""

from __future__ import annotations

from inference_openpi.config import RobotProfile

OVERLAY_NAME = "pi0_aloha_overlay"


def create_aloha_overlay_config(profile: RobotProfile):
    """Build a TrainConfig aligned with the local fine-tuned checkpoint."""
    from openpi.models import pi0_config
    from openpi.training.config import AssetsConfig, LeRobotAlohaDataConfig, TrainConfig

    return TrainConfig(
        name=OVERLAY_NAME,
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            adapt_to_pi=profile.policy.adapt_to_pi,
            use_delta_joint_actions=profile.policy.use_delta_joint_actions,
            assets=AssetsConfig(asset_id=profile.policy.asset_id),
            default_prompt=profile.policy.prompt or None,
        ),
        policy_metadata={
            "overlay": OVERLAY_NAME,
            "asset_id": profile.policy.asset_id,
            "adapt_to_pi": profile.policy.adapt_to_pi,
            "use_delta_joint_actions": profile.policy.use_delta_joint_actions,
        },
    )
