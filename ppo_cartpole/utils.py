import torch
import numpy as np
import gymnasium as gym


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    run_name: str,
    mode: str = "classic-control",
    video_log_freq: int = None,
    video_save_path: str = None,
    **kwargs,
):
    """
    Return a function that returns an environment after setting up boilerplate.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0 and video_log_freq:
            env = gym.wrappers.RecordVideo(
                env,
                f"{video_save_path}/{run_name}",
                episode_trigger=lambda episode_id: episode_id % video_log_freq == 0,
                disable_logger=True,
            )

        if mode == "atari":
            env = prepare_atari_env(env)
        elif mode == "mujoco":
            env = prepare_mujoco_env(env)

        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

def get_episode_data_from_infos(infos: dict) -> dict[str, int | float] | None:
    """
    Helper function: returns dict of data from the first terminated environment, if at least one terminated.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }