import torch
import numpy as np
from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float
import torch.nn as nn

class QNetwork(nn.Module):
    """Class representing the Q-Network. 
    The QNetwork maps states to distribution of qvalues over actions
    """
    def __init__(self, obs_shape: tuple[int], num_actions: int, hidden_size: list[int] = [120, 84]):
        """
        What does obs_shape actually do? We vectorize each observation??
            Look deeply into this, please. 
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_shape[0], hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], num_actions)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

@dataclass
class ReplayBufferSamples:
    """Should be an item to be 
    """
    obs: Float[Tensor, "sample_size, *obs_shape"]
    actions: Float[Tensor, "sample_size"] # Why action shape?? 
    rewards: Float[Tensor, "sample_size"]
    terminated: Float[Tensor, "sample_size"]
    next_obs: Float[Tensor, "sample_size, obs_shape"]

class ReplayBuffer:
    """ReplayBuffer to store experiences
    """
    def __init__(self, num_envs: int, obs_shape: tuple[int], action_shape: tuple[int], buffer_size: int, seed: int):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

        self.obs = np.empty((0, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, *self.action_shape), dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float32)
        self.terminated = np.empty(0, dtype=bool)
        self.next_obs = np.empty((0, *self.obs_shape), dtype=np.float32)

    def add(
        self,
        obs: Float[Tensor, "sample_size, *obs_shape"],
        actions: Float[Tensor, "sample_size, *action_shape"], # Why action shape?? 
        rewards: Float[Tensor, "sample_size"],
        terminated: Float[Tensor, "sample_size"],
        next_obs: Float[Tensor, "sample_size, obs_shape"]
    ):
        for data, expected_shape in zip(
            [obs, actions, rewards, terminated, next_obs], [self.obs_shape, 
            self.action_shape, (), (), self.obs_shape]
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == (self.num_envs, *expected_shape)

        # Add data to buffer, slicing off the old elements
        self.obs = np.concatenate((self.obs, obs))[-self.buffer_size :]
        self.actions = np.concatenate((self.actions, actions))[-self.buffer_size :]
        self.rewards = np.concatenate((self.rewards, rewards))[-self.buffer_size :]
        self.terminated = np.concatenate((self.terminated, terminated))[-self.buffer_size :]
        self.next_obs = np.concatenate((self.next_obs, next_obs))[-self.buffer_size :]

    def sample(self, sample_size: int, device: torch.device) -> ReplayBufferSamples:
        """
        Sample a batch of transitions from the buffer, with replacement.
        """
        indices = self.rng.integers(0, self.buffer_size, sample_size)

        return ReplayBufferSamples(
            obs=torch.tensor(self.obs[indices], dtype=torch.float32, device=device),
            actions=torch.tensor(self.actions[indices], device=device),
            rewards=torch.tensor(self.rewards[indices], dtype=torch.float32, device=device),
            terminated=torch.tensor(self.terminated[indices], device=device),
            next_obs=torch.tensor(self.next_obs[indices], dtype=torch.float32, device=device)
        )


if __name__ == "__main__":
    net= QNetwork(obs_shape=(4,), num_actions=2)
    print(f"Number of parameters in the current QNetwork is {sum(param.nelement() for param in net.parameters())}")

    