import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

WHEELLEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/root/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/WheelLeg/wheellegv3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.114),
        joint_pos={
            "left_forw_joint": 0.0,
            "left_back_joint": 0.0,
            "right_forw_joint": 0.0,
            "right_back_joint": 0.0,

            "left_forw_p_joint": 0.0,
            "left_back_p_joint": 0.0,
            "right_forw_p_joint": 0.0,
            "right_back_p_joint": 0.0,

            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        }
    ),

    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wheel_joint",
                "right_wheel_joint",
            ],
            effort_limit_sim=2.0,
            stiffness=0.0,
            damping=0.5,
        ),

        "left_leg_pair": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_forw_joint",
                "left_back_joint",
            ],
            effort_limit_sim=0.70,
            stiffness=6.0,
            damping=0.6,
        ),

        "right_leg_pair": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_forw_joint",
                "right_back_joint",
            ],
            effort_limit_sim=0.70,
            stiffness=6.0,
            damping=0.6,
        ),

    },
    soft_joint_pos_limit_factor=0.80,
)

WHEELLEG_MINIMAL_CFG = WHEELLEG_CFG.copy()
WHEELLEG_MINIMAL_CFG.spawn.usd_path = "/root/IsaacLab/source/isaaclab_assets/data/Robots/WheelLeg/wheelleg_minimal.usd"
