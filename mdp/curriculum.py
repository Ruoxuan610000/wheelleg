from __future__ import annotations

from collections.abc import Sequence
import torch
import isaaclab.envs.mdp as mdp

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

def terrain_levels_wheel_legged(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """复刻旧代码 _update_terrain_curriculum 的核心逻辑。"""
    robot: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    env_ids = torch.as_tensor(env_ids, device=robot.device, dtype=torch.long)
    if len(env_ids) == 0:
        return torch.mean(terrain.terrain_levels.float())

    # 旧代码: distance = ||root_xy - env_origin_xy||
    distance = torch.norm(
        robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    # 旧代码: move_up = distance > terrain_length / 2
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2.0

    # 这里复刻“tracking_lin_vel均值太差则降级”
    # 你需要保证 reward manager 中有 tracking_lin_vel 这一项的 episodic 统计
    tracking_mean = env.reward_manager._episode_sums["tracking_lin_vel"][env_ids] / env.max_episode_length_s
    move_down = tracking_mean < 0.4
    move_down = move_down & (~move_up)

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def update_command_range_wheel_legged(env, env_ids, old_value, *, step=0.05, max_abs=2.5, min_abs=1.0):
    """给 modify_term_cfg/modify_env_param 用：动态改 lin_vel_x range。"""
    # old_value 期望是 (min, max)
    lo, hi = old_value

    # 这里按你旧逻辑做“向外扩”
    new_lo = max(lo - step, -max_abs)
    new_hi = min(hi + step, max_abs)

    # 保持至少 [-1, 1] 这种初始安全范围边界
    new_lo = min(new_lo, -min_abs)
    new_hi = max(new_hi, min_abs)

    return (new_lo, new_hi)


def _get_reward_avg_per_second(env, env_ids: torch.Tensor, reward_term_name: str) -> torch.Tensor:
    """Return the episodic average reward per second for the given reward term."""
    if reward_term_name not in env.reward_manager._episode_sums:
        raise ValueError(
            f"Reward term '{reward_term_name}' was not found in reward_manager._episode_sums. "
            f"Available terms: {list(env.reward_manager._episode_sums.keys())}"
        )

    return env.reward_manager._episode_sums[reward_term_name][env_ids] / env.max_episode_length_s


def _get_reward_weight(env, reward_term_name: str) -> float:
    """Fetch the configured reward weight for a reward term."""
    return env.reward_manager.get_term_cfg(reward_term_name).weight


def update_command_curriculum(
    env,
    env_ids,
    old_value,
    *,
    lin_vel_reward_term: str = "track_lin_vel_xy_exp",
    ang_vel_reward_term: str | None = "track_yaw_rate_l2",
    curriculum_threshold: float = 0.7,
    ang_vel_threshold_scale: float = 0.8,
    expand_step: float = 0.1,
    terrain_expand_step: float = 0.05,
    basic_terrain_bonus_step: float = 0.45,
    basic_max_curriculum: float = 2.5,
    advanced_max_curriculum: float | None = None,
    use_terrain_curriculum: bool | None = None,
    success_env_attr: str = "success_ids",
    basic_env_attr: str = "basic_terrain_idx",
    advanced_env_attr: str = "advanced_terrain_idx",
):
    """Isaac Lab version of Wheel-Legged-Gym's ``update_command_curriculum``.

    This helper is designed for ``mdp.modify_term_cfg`` and returns an updated
    ``(min, max)`` tuple for ``commands.base_velocity.ranges.lin_vel_x``.

    Notes:
    - The original Wheel-Legged-Gym environment stores per-environment command
      ranges. Isaac Lab's command config is typically shared across all envs, so
      this function aggregates the old logic into one global range update.
    - The curriculum expands the command range only when the episodic command
      tracking reward is high enough.
    """
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return mdp.modify_term_cfg.NO_CHANGE

    lo, hi = float(old_value[0]), float(old_value[1])
    advanced_max_curriculum = basic_max_curriculum if advanced_max_curriculum is None else advanced_max_curriculum

    if use_terrain_curriculum is None:
        use_terrain_curriculum = bool(getattr(getattr(env.cfg, "terrain", None), "curriculum", False))

    lin_vel_weight = _get_reward_weight(env, lin_vel_reward_term)
    if lin_vel_weight <= 0.0:
        return mdp.modify_term_cfg.NO_CHANGE

    # Terrain-curriculum mode follows the old project more closely: it expands
    # commands when environments that just cleared a terrain level also tracked
    # the commanded linear velocity well enough.
    if use_terrain_curriculum and hasattr(env, success_env_attr):
        success_env_ids = getattr(env, success_env_attr)
        if success_env_ids is None:
            return mdp.modify_term_cfg.NO_CHANGE
        success_ids = torch.as_tensor(success_env_ids, device=env.device, dtype=torch.long)
        if success_ids.numel() == 0:
            return mdp.modify_term_cfg.NO_CHANGE

        success_mask = (
            _get_reward_avg_per_second(env, success_ids, lin_vel_reward_term)
            > curriculum_threshold * lin_vel_weight
        )
        success_ids = success_ids[success_mask]
        if success_ids.numel() == 0:
            return mdp.modify_term_cfg.NO_CHANGE

        range_step = terrain_expand_step
        range_limit = advanced_max_curriculum

        if hasattr(env, basic_env_attr):
            basic_env_ids = getattr(env, basic_env_attr)
            if basic_env_ids is not None:
                basic_env_ids = torch.as_tensor(basic_env_ids, device=env.device, dtype=torch.long)
                on_basic_terrain = (success_ids[:, None] == basic_env_ids[None, :]).any(dim=1)
                if torch.any(on_basic_terrain):
                    range_step += basic_terrain_bonus_step
                    range_limit = basic_max_curriculum

        return (
            max(lo - range_step, -range_limit),
            min(hi + range_step, range_limit),
        )

    # Flat-ground mode expands the shared command range only if both linear and
    # angular tracking stay above the target fraction of their maximum reward.
    lin_vel_avg = torch.mean(_get_reward_avg_per_second(env, env_ids, lin_vel_reward_term))
    lin_vel_ok = lin_vel_avg > curriculum_threshold * lin_vel_weight

    ang_vel_ok = True
    if ang_vel_reward_term is not None and ang_vel_reward_term in env.reward_manager._episode_sums:
        ang_vel_weight = _get_reward_weight(env, ang_vel_reward_term)
        if ang_vel_weight > 0.0:
            ang_vel_avg = torch.mean(_get_reward_avg_per_second(env, env_ids, ang_vel_reward_term))
            ang_vel_ok = ang_vel_avg > curriculum_threshold * ang_vel_weight * ang_vel_threshold_scale

    if not (lin_vel_ok and ang_vel_ok):
        return mdp.modify_term_cfg.NO_CHANGE

    return (
        torch.clamp(torch.tensor(lo - expand_step), min=-basic_max_curriculum, max=0.0).item(),
        torch.clamp(torch.tensor(hi + expand_step), min=0.0, max=basic_max_curriculum).item(),
    )

