"""Configuration types for data conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig

_ROBOT_JOINT_NAMES_DEFAULT = {
    "g1": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    # CORRECTED: this must match the *retargeting MJCF's native joint order*
    # (models/r1/r1_26dof.xml, i.e. mujoco.MjModel joint declaration order), NOT holosoma's
    # training-side dof_names order. Proof: for g1, _ROBOT_JOINT_NAMES_DEFAULT["g1"] above is
    # byte-for-byte identical to models/g1/g1_29dof.xml's native mujoco joint order (verified),
    # which is what makes convert_data_format_mj.py's `dof_index_list` a no-op identity permutation
    # for g1. The previous version of this list used holosoma's IsaacLab dof_names order instead
    # (config_values/robot.py's r1_26dof, itself a *probed* PhysX order, unrelated to file
    # declaration order) which is a *different* permutation than the MJCF's native order for r1.
    # That mismatch made `motion_dof_pos[:, dof_index_list]` in convert_data_format_mj.py
    # scramble already-correct retargeted qpos into the wrong slots before replay/save --
    # this is what produced the visibly wrong joint targets in the MuJoCo viewer.
    # (holosoma's IsaacLab side is unaffected: it resolves joints by name via
    # `Articulation.find_joints(..., preserve_order=True)` and the WBT motion loader
    # remaps the saved npz by its own `joint_names` field, so dof_names order there is
    # arbitrary-but-consistent and does not need to match this list.)
    "r1": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "head_pitch_joint",
        "head_yaw_joint",
    ],
}


@dataclass(frozen=True)
class DataConversionConfig:
    """Configuration for data conversion.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with all conversion parameters.
    """

    input_file: str
    """Path to input motion file."""

    robot: str = "g1"
    """Robot model to use. Use str to allow dynamic robot types."""

    data_format: str = "smplh"
    """Motion data format. Use str to allow dynamic data formats."""

    object_name: str | None = None
    """Override object name (default depends on robot and data type)."""

    input_fps: int = 30
    """FPS of the input motion."""

    output_fps: int = 50
    """FPS of the output motion."""

    line_range: tuple[int, int] | None = None
    """Line range (start, end) for loading data (both inclusive)."""

    has_dynamic_object: bool = False
    """Whether the motion has a dynamic object."""

    output_name: str | None = None
    """Name of the output motion npz file."""

    once: bool = False
    """Run the motion once and exit."""

    headless: bool = False
    """Run without viewer (headless mode for batch processing)."""

    use_omniretarget_data: bool = False
    """Use OmniRetarget data format."""

    # --- Nested configs ---
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    """Robot configuration (nested - can override robot_urdf_file, robot_dof, etc.
    via --robot-config.robot-urdf-file)."""

    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplh", robot_type="g1")
    )
    """Motion data configuration (nested - can override data_format, robot_type, etc.
    via --motion-data-config.data-format)."""

    # --- Joint names, follow the pattern in holosoma_retargeting/config_types/robot.py ---
    joint_names: list[str] | None = None
    """Joint names to use."""

    def _joint_names(self) -> list[str]:
        """Get joint names - use override if provided, else use default."""
        if self.joint_names is not None:
            return self.joint_names
        if self.robot not in _ROBOT_JOINT_NAMES_DEFAULT:
            raise ValueError(f"No joint names found for robot: {self.robot}")
        return _ROBOT_JOINT_NAMES_DEFAULT[self.robot]

    JOINT_NAMES = property(
        _joint_names,
        doc="Get joint names - use override if provided, else use default.",
    )
