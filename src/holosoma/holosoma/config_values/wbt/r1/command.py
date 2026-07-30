"""Whole Body Tracking command presets for the R1 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    # Placeholder -- always overridden at the CLI via
    # --command.setup_terms.motion_command.params.motion_config.motion_file=<path>
    motion_file="holosoma/data/motions/r1_26dof/whole_body_tracking/placeholder.npz",
    body_names_to_track=[
        "pelvis_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_link",
    ],
    body_name_ref=["waist_yaw_link"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=init_pose_config,
)

r1_26dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

__all__ = ["r1_26dof_wbt_command"]
