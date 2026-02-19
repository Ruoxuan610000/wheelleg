import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.envs.mdp as mdp
from .wheelleg import WHEELLEG_CFG

from .reward import *

@configclass
class WheelLegSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = WHEELLEG_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot/.*", history_length=3, track_air_time=True)

    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

@configclass
class CommandsCfg:

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.6, 0.6), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:

    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint",],
        scale=0.5,
        use_default_offset=True
        )
    
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint",],
        scale=8.0,
        use_default_offset=False
        )

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) #, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel) #, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity) #, noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel) #, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel) #, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:

    """
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 1.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    """
    reset_joints = EventTerm(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "position_range": (0.0, 0.0),   # 不随机：严格回默认姿态
        "velocity_range": (0.0, 0.0),
    },
)

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), 
                "y": (-0.5, 0.5), 
                "z": (0.291, 0.291),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14)},
                           
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    )

@configclass
class RewardsCfg:
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std":0.8} 
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std":0.8} 
    )
    

    base_contact = RewTerm(func=mdp.undesired_contacts, weight=-0.5, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},)
    #wheel_contact = RewTerm(func=mdp.undesired_contacts, weight=1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="left_wheel_link"), "threshold": 1.0},)
    leg_contact_p = RewTerm(
        func=mdp.undesired_contacts, weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_forw_link", "left_back_link", "right_forw_link", "right_back_link", "left_forw_p_link", "right_forw_p_link", "left_back_p_link", "right_back_p_link"]), "threshold": 1.0},
    )

    #lin_vel_xyz_exp = RewTerm(func=lin_vel_xyz_exp, weight=-0.5, params={"std": 0.5})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    #dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    #action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    balance_exp = RewTerm(func=balance_exp, weight=2.0, params={"std": 0.5})

@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},  # ~40deg
    )

'''
@configclass
class CurriculumsCfg:
    """Dictionary of curriculum terms."""

    # ---------- 通用的“按步数覆盖某个配置值”的函数 ----------
    @staticmethod
    def override_after(env, env_ids, old_value, value, num_steps: int):
        # common_step_counter 是全局累计步数（不是 episode 内步数）
        if env.common_step_counter > num_steps:
            return value
        return mdp.modify_term_cfg.NO_CHANGE

    # (A) 逐步放开命令范围：先小，再大
    cmd_lin_x_stage1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": override_after,
            "modify_params": {"value": (-0.4, 0.4), "num_steps": 0},
        },
    )
    cmd_lin_x_stage2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": override_after,
            "modify_params": {"value": (-0.6, 0.6), "num_steps": 200_000},
        },
    )
    cmd_lin_x_stage3 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": override_after,
            "modify_params": {"value": (-1.0, 1.0), "num_steps": 600_000},
        },
    )

    # (B) 侧向速度：先几乎不训，再慢慢加
    cmd_lin_y_stage1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": override_after,
            "modify_params": {"value": (-0.05, 0.05), "num_steps": 0},
        },
    )
    cmd_lin_y_stage2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_y",
            "modify_fn": override_after,
            "modify_params": {"value": (-0.2, 0.2), "num_steps": 300_000},
        },
    )

    # (C) yaw：先小，再大
    cmd_yaw_stage1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": override_after,
            "modify_params": {"value": (-0.4, 0.4), "num_steps": 0},
        },
    )
    cmd_yaw_stage2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": override_after,
            "modify_params": {"value": (-0.8, 0.8), "num_steps": 400_000},
        },
    )

    # (D) 先不启用 heading_command，后面再打开（更稳）
    heading_off = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.heading_command",
            "modify_fn": override_after,
            "modify_params": {"value": False, "num_steps": 0},
        },
    )
    heading_on = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.heading_command",
            "modify_fn": override_after,
            "modify_params": {"value": True, "num_steps": 800_000},
        },
    )
    rel_heading_envs_small = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_heading_envs",
            "modify_fn": override_after,
            "modify_params": {"value": 0.1, "num_steps": 800_000},  # 先只给10%环境上heading任务
        },
    )
    rel_heading_envs_full = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_heading_envs",
            "modify_fn": override_after,
            "modify_params": {"value": 1.0, "num_steps": 1_200_000},
        },
    )

    # (E) 逐步加大“腿连杆接触惩罚”的权重（刚开始别扣太狠）
    # 这是 modify_reward_weight 的官方用法：到指定步数后，把某 reward term 的 weight 改成指定值。:contentReference[oaicite:1]{index=1}
    leg_contact_soft = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "leg_contact_p", "weight": -0.2, "num_steps": 0},
    )
    leg_contact_medium = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "leg_contact_p", "weight": -0.5, "num_steps": 500_000},
    )
    leg_contact_hard = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "leg_contact_p", "weight": -1.0, "num_steps": 1_000_000},
    )

'''


@configclass
class WheelLegEnvCfg(ManagerBasedRLEnvCfg):

    scene: WheelLegSceneCfg = WheelLegSceneCfg(num_envs=4096, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events:EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    #curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
