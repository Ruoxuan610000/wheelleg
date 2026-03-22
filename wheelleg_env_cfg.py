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
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
import isaaclab_tasks.manager_based.user.wheelleg.mdp as mdp
from .wheelleg import WHEELLEG_CFG




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
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )

    height_command = mdp.HeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.HeightCommandCfg.Ranges(
            # The current wheel-leg asset stands around 0.30 m at reset.
            # Sampling much higher targets makes the policy learn to ignore the
            # height command and keep the rear legs folded.
            height=(0.30, 0.40),
        ),
    )

@configclass
class ActionsCfg:

    leg_pos = mdp.JointPositionActionWithOffsetCfg(
        asset_name="robot", 
        joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint",],
        scale=1.0,
        clip={".*": (-1.2, 1.2)},
        use_default_offset=True,
        preserve_order=True,
        )
    
    left_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint"],
        scale=8.0,
        clip={".*": (-8.0, 8.0)},
        use_default_offset=False,
        )

    right_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["right_wheel_joint"],
        scale=-8.0,
        clip={".*": (-8.0, 8.0)},
        use_default_offset=False,
        )

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) #, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel) #, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_hight = ObsTerm(func=mdp.base_pos_z) #, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity) #, noise=Unoise(n_min=-0.05, n_max=0.05))

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "height_command"})


        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg(
                                    "robot", 
                                    joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint","left_wheel_joint", "right_wheel_joint"],
                                    preserve_order=True)
                                    }
                            ) #, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
                            params={
                                "asset_cfg": SceneEntityCfg(
                                    "robot",
                                    joint_names=["left_forw_joint", "left_back_joint", "right_forw_joint", "right_back_joint","left_wheel_joint", "right_wheel_joint"],
                                    preserve_order=True)
                                    }
                            ) #, noise=Unoise(n_min=-1.5, n_max=1.5))

        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    robot_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-0.2, 0.5),
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
                "x": (-0.01, 0.01),
                "y": (-0.005, 0.005),
                "z": (-0.005, 0.005),
            },
        },
    )
    
    robot_inertia = EventTerm(
        func=mdp.randomize_inertia_independent,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_range": (0.95, 1.05),
        },
    )

    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.98, 1.02),
            "damping_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # 若你的 actuator / action term 支持 torque scale，也可挂这里
    default_joint_offset = EventTerm(
        func=mdp.randomize_default_joint_pos_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "offset_range": (-0.01, 0.01),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.01, 0.01),
                "pitch": (-0.02, 0.02),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(12.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.1, 0.1),
            },
        },
    )

@configclass
class RewardsCfg:
    is_alive = RewTerm(func=mdp.is_alive, weight=0.5)
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=3.0, 
        params={"command_name": "base_velocity", "std":0.35} 
    )

    track_lin_vel_xy_exp_enhanced = RewTerm(
        func=mdp.rew_track_lin_vel_xy_enhanced, 
        weight=0.0, 
        params={"command_name": "base_velocity", "std":0.35} 
    )

    track_yaw_rate_l2 = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.0, 
        params={"command_name": "base_velocity", "std":0.2}
    )

    balance_exp = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    base_height_reward = RewTerm(
        func=mdp.rew_base_height_exp,
        weight=3.0,
        params={"command_name": "height_command", "std": 0.005},
    )

    vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.5,)

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)

    niminal_state = RewTerm(func=mdp.symmetry_state, weight=-1.0)

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.01,    
    )

    leg_joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-5e-5, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_forw_joint", "left_back_joint","right_forw_joint", "right_back_joint",], preserve_order=True,)},
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    collison = RewTerm(
        func=mdp.undesired_contacts, weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), 
            "threshold": 1.0
        },
    )
    
    avoid_default_leg_pose = RewTerm(
        func=mdp.joint_pos_near_default_penalty,
        weight=-0.5,
        params={
            "threshold": 0.08,
            "power": 2.0,
            "normalize": True,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_forw_joint",
                    "left_back_joint",
                    "right_forw_joint",
                    "right_back_joint",
                ],
                preserve_order=True,
            ),
        },
    )


    joint_pos_l2 = RewTerm(func=mdp.joint_pos_limits, weight=-0.2)
    
    action_acc_l2 = RewTerm(func=mdp.rew_action_acc_l2, weight=-0.01)
    

@configclass
class TerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.85},
    )

    bad_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 10,}
    )

    #joint_pos_out_of_manual_limit = DoneTerm(
    #    func=mdp.joint_pos_out_of_manual_limit,
    #    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_forw_joint", "left_back_joint","right_forw_joint", "right_back_joint",], preserve_order=True), "bounds": (-1.57*0.67, 1.57*0.67)}
    #)


@configclass
class CurriculumsCfg:
    #terrain_levels = CurrTerm(func=mdp.terrain_levels_wheel_legged)

    command_lin_vel_x_stage1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": (0.0, 0.2), "num_steps": 0},
        },
    )

    command_lin_vel_x_stage2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": (0.0, 0.5), "num_steps": 200_000},
        },
    )

    command_lin_vel_x_stage3 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": (0.0, 0.8), "num_steps": 600_000},
        },
    )

    command_lin_vel_x_stage4 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": (-1.0, 1.0), "num_steps": 1_000_000},
        },
    )

    standing_envs_stage1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_standing_envs",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": 0.8, "num_steps": 0},
        },
    )

    standing_envs_stage2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_standing_envs",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": 0.5, "num_steps": 300_000},
        },
    )

    standing_envs_stage3 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_standing_envs",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": 0.2, "num_steps": 800_000},
        },
    )

    heading_off = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.heading_command",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": False, "num_steps": 0},
        },
    )

    heading_on = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.heading_command",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": True, "num_steps": 800_000},
        },
    )

    rel_heading_envs_stage1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_heading_envs",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": 0.0, "num_steps": 0},
        },
    )

    rel_heading_envs_stage2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_heading_envs",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": 0.2, "num_steps": 800_000},
        },
    )

    rel_heading_envs_stage3 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.rel_heading_envs",
            "modify_fn": mdp.override_after,
            "modify_params": {"value": 1.0, "num_steps": 1_200_000},
        },
    )

    yaw_reward_stage1 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "track_yaw_rate_l2", "weight": 0.0, "num_steps": 0},
    )

    yaw_reward_stage2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "track_yaw_rate_l2", "weight": 0.5, "num_steps": 800_000},
    )

    yaw_reward_stage3 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "track_yaw_rate_l2", "weight": 1.0, "num_steps": 1_200_000},
    )

@configclass
class WheelLegEnvCfg(ManagerBasedRLEnvCfg):

    scene: WheelLegSceneCfg = WheelLegSceneCfg(num_envs=4096, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events:EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        #if self.scene.height_scanner is not None:
        #    self.scene.height_scanner.update_period = self.decimation * self.sim.dt
