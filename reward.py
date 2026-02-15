from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def balance_exp(
        env, std:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
    ang_error = torch.square(roll)  + torch.square(pitch)

    return torch.exp(-ang_error / std**2)

def lin_vel_xyz_exp(
        env, std:float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset = env.scene[asset_cfg.name]
    vel = asset.data.root_lin_vel_w
    vx, vy, vz = vel[..., 0], vel[..., 1], vel[..., 2]
    lin_vel_error = torch.square(vx) + torch.square(vy) + torch.square(vz)

    return torch.exp(-lin_vel_error / std**2)

"""
def contact_based_reward(
        env, sensor_cfg: SceneEntityCfg, threshold: float
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact_time = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    return 
"""

def leg_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    return reward

