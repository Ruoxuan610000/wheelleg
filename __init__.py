import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-WheelLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelleg_env_cfg:WheelLegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegRSLRLCfg",
    },
)

gym.register(
    id="Isaac-WheelLeg-Tough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelleg_env_cfg_tough:WheelLegToughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegToughRSLRLCfg",
    },
)

gym.register(
    id="Isaac-WheelLeg-V2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheelleg_env_cfg_v2:WheelLegEnvV2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV2RSLRLCfg",
    },
)
