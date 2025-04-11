import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym

from torch import Tensor
from typing import Tuple, List
from jaxtyping import Float, Int
from torch.distributions import Categorical


def get_policy_network(hidden_state: int):
    return nn.Sequential(
        nn.Linear(4, hidden_state),
        nn.Tanh(),
        nn.Linear(hidden_state, 2)
    )

class SimplePolicyGradient:
    def __init__(self, hidden_size: int, epochs: int) -> None:
        self.env = gym.make("CartPole-v1")
        self.epochs = epochs
        self.policy = get_policy_network(hidden_state=hidden_size)

    def get_action(self, logits: Float[Tensor, "... 2"]) -> Int:
        """Function to sample action from the logits output of the policy network

        Args:
            logits (Float[Tensor, ... 2]): output of the policy network

        Returns:
            Int: sampled action
        """
        return Categorical(logits=logits).sample().item()

    def get_log_probs(self, logits: Float[Tensor, "2"], action: int) -> Float[Tensor, ""]:
        """Returns log probability of the chosen action."""
        return torch.log_softmax(logits, dim=-1)[action]

    def train_agent(self, ):
        for _ in range(self.epochs):
            obs, act, rewards, log_probs = self.train_one_epoch_step()
            print(sum(rewards))
            

    def train_one_epoch_step(self, ) -> Tuple[List[Tensor], List[Float], List[Int], List[Tensor]]:
        """Function to take rollout in the environment. 

        Returns:
            _type_: _description_
        """
        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_log_probs = []

        obs, _ = self.env.reset()
        obs = torch.Tensor(obs)

        done = False

        while not done:
            logits = self.policy(obs)
            act = self.get_action(logits)
            log_probs = self.get_log_probs(logits, act)

            next_obs, reward, terminated, truncated, _ = self.env.step(act)

            done = terminated or truncated

            batch_obs.append(obs.detach().clone())
            batch_act.append(act)
            batch_rew.append(reward)
            batch_log_probs.append(log_probs.detach().clone())
            
            # print(type(obs), type(reward), type(act), type(log_probs))

            obs = torch.Tensor(next_obs)
        
        return batch_obs, batch_act, batch_rew, batch_log_probs

  
if __name__ == "__main__":
    policy_net = get_policy_network(32)
    print(policy_net)

    spg = SimplePolicyGradient(32, 2)
    spg.train_agent()
