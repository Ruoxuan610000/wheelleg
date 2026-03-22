from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class WheelLegRSLRLCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 4000
    save_interval = 50
    experiment_name = "wheelleg"
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=10,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )    

@configclass
class WheelLegToughRSLRLCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48

    # 既然要从 model3999.pt 继续训练，max_iterations 要大于 4000
    max_iterations = 4001
    save_interval = 50

    experiment_name = "wheelleg"
    run_name = "tough_terrain"
    clip_actions = 1.0

    # 从 checkpoint 继续训练
    resume = True
    load_run = "2026-03-11_15-10-32"         # 例如 "2026-03-10_12-00-00"
    load_checkpoint = "model_3999.pt"    # 指定继续训练的 checkpoint

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=10,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheelLegV2RSLRLCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 5000
    save_interval = 50
    experiment_name = "wheelleg_v2"
    empirical_normalization = False
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
