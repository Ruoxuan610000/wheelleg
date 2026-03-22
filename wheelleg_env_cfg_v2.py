import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.user.wheelleg.mdp as mdp

from .wheelleg_env_cfg import EventCfg as BaseEventCfg
from .wheelleg_env_cfg import WheelLegEnvCfg as WheelLegEnvCfgBase
from .wheelleg_env_cfg import WheelLegSceneCfg


LEG_JOINT_NAMES = [
    "left_forw_joint",
    "left_back_joint",
    "right_forw_joint",
    "right_back_joint",
]
WHEEL_JOINT_NAMES = ["left_wheel_joint", "right_wheel_joint"]
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
DELAYED_ACTION_TERMS = ["leg_pos", "left_wheel_vel", "right_wheel_vel"]


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.8),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi),
        ),
    )

    height_command = mdp.HeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        debug_vis=False,
        ranges=mdp.HeightCommandCfg.Ranges(height=(0.30, 0.38)),
    )


@configclass
class ActionsCfg:
    leg_pos = mdp.JointPositionActionWithOffsetAndDelayCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=1.0,
        clip={".*": (-1.2, 1.2)},
        use_default_offset=True,
        preserve_order=True,
        min_delay=0,
        max_delay=1,
    )

    left_wheel_vel = mdp.JointVelocityActionWithDelayCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint"],
        scale=8.0,
        clip={".*": (-8.0, 8.0)},
        use_default_offset=False,
        min_delay=0,
        max_delay=1,
    )

    right_wheel_vel = mdp.JointVelocityActionWithDelayCfg(
        asset_name="robot",
        joint_names=["right_wheel_joint"],
        scale=-8.0,
        clip={".*": (-8.0, 8.0)},
        use_default_offset=False,
        min_delay=0,
        max_delay=1,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.08, n_max=0.08))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.16, n_max=0.16))
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "height_command"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
            },
            noise=Unoise(n_min=-1.0, n_max=1.0),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 4
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "height_command"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
            },
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
            },
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
            },
        )
        joint_offset = ObsTerm(
            func=mdp.default_joint_pos_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True),
            },
        )
        action_delay = ObsTerm(func=mdp.action_delay, params={"term_names": DELAYED_ACTION_TERMS})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg(BaseEventCfg):
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.5, 1.4),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    robot_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-0.5, 1.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.02, 0.02),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )

    robot_inertia = EventTerm(
        func=mdp.randomize_inertia_independent,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.9, 1.1),
        },
    )

    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    default_joint_offset = EventTerm(
        func=mdp.randomize_default_joint_pos_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "offset_range": (-0.02, 0.02),
        },
    )

    action_delay = EventTerm(
        func=mdp.randomize_action_delay,
        mode="reset",
        params={
            "min_delay": 0,
            "max_delay": 1,
            "term_names": DELAYED_ACTION_TERMS,
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class RewardsCfg:
    is_alive = RewTerm(func=mdp.is_alive, weight=0.25)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )

    track_yaw_rate_l2 = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )

    base_height_reward = RewTerm(
        func=mdp.rew_base_height_exp,
        weight=2.0,
        params={"command_name": "height_command", "std": 0.008},
    )

    balance_exp = RewTerm(func=mdp.flat_orientation_l2, weight=-4.0)
    vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)

    nominal_state = RewTerm(func=mdp.symmetry_state, weight=-0.5)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_acc_l2 = RewTerm(func=mdp.rew_action_acc_l2, weight=-0.01)

    leg_joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)},
    )

    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)},
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.02,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )

    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 1.0,
        },
    )

    joint_pos_l2 = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-2.0,
        params={"term_keys": ["bad_orientation", "bad_contact"]},
    )

    


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.9},
    )

    bad_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 12.0,
        },
    )


@configclass
class CurriculumsCfg:
    command_curriculum = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.update_command_curriculum,
            "modify_params": {
                "lin_vel_reward_term": "track_lin_vel_xy_exp",
                "ang_vel_reward_term": "track_yaw_rate_l2",
                "curriculum_threshold": 0.7,
                "expand_step": 0.1,
                "basic_max_curriculum": 2.0,
            },
        },
    )


@configclass
class WheelLegEnvV2Cfg(WheelLegEnvCfgBase):
    scene: WheelLegSceneCfg = WheelLegSceneCfg(num_envs=4096, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
