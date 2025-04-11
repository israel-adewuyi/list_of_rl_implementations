import time
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym

from torch import Tensor
from typing import Tuple, List
from jaxtyping import Float, Int, Bool
from dataclasses import dataclass
from torch.distributions import Categorical


@dataclass
class SPGArgs:
    """Dataclass for the necessary parameters"""

    env_name: str = "CartPole-v1"
    hidden_size: Int = 32
    num_epochs: Int = 2
    project_name: str = "policy_gradients"
    apply_discount: Bool = False
    gamma: Float = 0.99
    lr: Float = 0.004

    def __post_init__(self):
        self.run_name = f"spg_{self.env_name}_lr={self.lr}_num_epochs={self.num_epochs}_time={time.strftime('%Y-%m-%d %H:%M:%S')}"

def get_policy_network(hidden_state: int):
    return nn.Sequential(
        nn.Linear(4, hidden_state),
        nn.Tanh(),
        nn.Linear(hidden_state, 2)
    )

class SimplePolicyGradient:
    def __init__(self, config: SPGArgs) -> None:
        self.env = gym.make(config.env_name)
        self.epochs = config.num_epochs
        self.policy = get_policy_network(hidden_state=config.hidden_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        
        print(config.run_name)

    def get_action(self, logits: Float[Tensor, "2"]) -> Int:
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
        """Function to train the agent"""
        for _ in range(self.epochs):
            _, _, rewards, log_probs = self.train_one_epoch_step()
            loss = self.compute_loss(rewards, log_probs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(sum(rewards))


    def compute_loss(self, rewards: List[float], log_probs: List[Tensor]) -> Float:
        """Computes the loss term for a trajectory.
        
        See: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient

        Args:
            rewards (List[float]): all the rewards received on the particular trajectory
            log_probs (List[Tensor]): the log_prob of the correct action taken at the t-th timestep

        Returns:
            Float: loss
        """
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + R
            returns.insert(0, R) 

        returns = torch.Tensor(returns)
        
        loss = -(torch.stack(log_probs) * returns).mean()

        return loss

    def train_one_epoch_step(self, ) -> Tuple[List[Tensor], List[Float], List[Int], List[Tensor]]:
        """Function to take rollout in the environment. 
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

            batch_obs.append(obs)
            batch_act.append(act)
            batch_rew.append(reward)
            batch_log_probs.append(log_probs)

            obs = torch.Tensor(next_obs)
        
        return batch_obs, batch_act, batch_rew, batch_log_probs

  
if __name__ == "__main__":
    policy_net = get_policy_network(32)
    print(policy_net)

    args = SPGArgs()

    spg = SimplePolicyGradient(args)
    spg.train_agent()
