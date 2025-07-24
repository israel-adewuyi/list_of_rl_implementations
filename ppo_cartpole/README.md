# Proximal Policy Optimization (PPO)

This repo implements the [PPO paper](https://arxiv.org/pdf/1707.06347) on Cartpole task.

- Reference Implementation - [Arena's PPO tuturial](https://arena-chapter2-rl.streamlit.app/[2.3]_PPO)
- Library for env - [Gymnasium](https://gymnasium.farama.org/)
- Video of agent's performance can be viewed at `videos`

### Repository structure

- `ppo_cartpole.py` implements the code to run the PPO agent on Cartpole.

### Args (for reproducibility)

The `PPOArgs` class in `ppo_cartpole.py` contains all the hyperparams used in this implementation. For the sake of redundancy,

- **seed**: `1`
- **env_id**: `"CartPole-v1"`
- **mode**: `"classic-control"`

#### Wandb / Logging

- **use_wandb**: `False`
- **video_log_freq**: `None`
- **wandb_project_name**: `"PPOCartPole"`
- **wandb_entity**: `"self_research_"`

#### Duration Settings

- **total_timesteps**: `500_000`
- **num_envs**: `4`
- **num_steps_per_rollout**: `128`
- **num_minibatches**: `4`
- **batches_per_learning_phase**: `4`

#### Optimization Hyperparameters

- **lr**: `2.5e-4`
- **max_grad_norm**: `0.5`

#### RL Hyperparameters

- **gamma**: `0.99`

#### PPO-Specific Hyperparameters

- **gae_lambda**: `0.95`
- **clip_coef**: `0.2`
- **ent_coef**: `0.01`
- **vf_coef**: `0.25`

#### Training Device

- **device**: `"cpu"`

### Culmulative rewards

![rewards](assets/agent_rewards.png)

### To run

`python simple_policy_gradient/spg_cartpole.py`
