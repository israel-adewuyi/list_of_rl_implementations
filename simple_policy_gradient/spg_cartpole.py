import time
import wandb
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym

from tqdm import tqdm
from torch import Tensor
from typing import Tuple, List
from jaxtyping import Float, Int, Bool
from dataclasses import dataclass
from dotenv import load_dotenv
from torch.distributions import Categorical

load_dotenv()

@dataclass
class SPGArgs:
    """Dataclass for the necessary parameters"""

    seed: Int = 0
    env_name: str = "CartPole-v1"
    hidden_size: Int = 64
    num_epochs: Int = 50
    project_name: str = "policy_gradients"
    apply_discount: Bool = True
    gamma: Float = 0.99
    lr: Float = 0.002
    batch_size: Int = 5000
    device: str = "cuda:3"

    def __post_init__(self):
        self.run_name = f"spg_{self.env_name}_batch_size={self.batch_size}_num_epochs={self.num_epochs}_seed={self.seed}_time={time.strftime('%Y-%m-%d %H:%M:%S')}"

def get_policy_network(hidden_state: int):
    return nn.Sequential(
        nn.Linear(4, hidden_state),
        nn.Tanh(),
        nn.Linear(hidden_state, 2)
    )

class SimplePolicyGradient:
    def __init__(self, config: SPGArgs) -> None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.env = gym.make(config.env_name)
        self.env.reset(seed=config.seed)
        self.config = config
        self.gamma = config.gamma if config.apply_discount else 1.0
        self.epochs = config.num_epochs
        self.policy = get_policy_network(hidden_state=config.hidden_size).to(config.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.lr)
        wandb.init(project=config.project_name, name=config.run_name)

    def get_action(self, logits: Float[Tensor, "2"]) -> Int:
        """Function to sample action from the logits output of the policy network

        Args:
            logits (Float[Tensor, 2]): output of the policy network

        Returns:
            Int: sampled action
        """
        return Categorical(logits=logits).sample().item()

    def get_log_probs(self, logits: Float[Tensor, "2"], action: int) -> Float[Tensor, ""]:
        """Returns log probability of the chosen action."""
        return torch.log_softmax(logits, dim=-1)[action]

    def train_agent(self, ):
        """Function to train the agent + log necessary metrics"""
        for step in tqdm(range(self.epochs)):
            batch_len, rewards, undiscounted_batch_ret, log_probs = self.train_one_epoch_step()

            loss = self.compute_loss(rewards, log_probs)
            grad_norm = sum(p.grad.norm().item() for p in self.policy.parameters() if p.grad is not None)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            wandb.log(
                {
                    "loss": loss.item(),
                    # "rewards_sum": sum(rewards),
                    "rewards_avg_per_episode": np.array(undiscounted_batch_ret, dtype=np.float32).mean(),
                    "episode_length": np.mean(batch_len), 
                    "grad_norm": grad_norm
                }, step
            )


    def compute_loss(self, returns: List[float], log_probs: List[Tensor]) -> Float:
        """Computes the loss term for a trajectory.
        
        See: https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient

        Args:
            rewards (List[float]): all the rewards received on the particular trajectory
            log_probs (List[Tensor]): the log_prob of the correct action taken at the t-th timestep

        Returns:
            Float: loss
        """
        returns = torch.Tensor(returns).to(self.config.device)
        returns = ((returns - returns.mean()) / (returns.std() + 1e-8))
        loss = -(torch.stack(log_probs).to(self.config.device) * returns).mean()

        return loss
    
    def compute_batch_returns(self, rewards: List[Int]) -> List[Float]:
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        return returns

    def train_one_epoch_step(self, ) -> Tuple[List[Tensor], List[Int], List[Float], List[Tensor]]:
        """Function to take rollout in the environment. 
        """
        batch_len, batch_rew, batch_ret, undiscounted_batch_ret, batch_log_probs = [], [], [], [], []
        obs, _ = self.env.reset(seed=self.config.seed)

        while True:
            obs = torch.Tensor(obs).to(self.config.device)
            logits = self.policy(obs)
            act = self.get_action(logits)
            log_probs = self.get_log_probs(logits, act)
            obs, reward, terminated, truncated, _ = self.env.step(act)
            done = terminated or truncated
            
            batch_rew.append(reward)
            batch_log_probs.append(log_probs)
            
            if done:
                undiscounted_batch_ret.append(sum(batch_rew))
                temp_batch_ret = self.compute_batch_returns(batch_rew)
                batch_len.append(len(batch_rew))
                batch_ret += temp_batch_ret
                batch_rew = []
                obs, _ = self.env.reset(seed=self.config.seed)
                if len(batch_log_probs) >= self.config.batch_size:
                    break
        
        return batch_len, batch_ret, undiscounted_batch_ret, batch_log_probs

  
if __name__ == "__main__":
    args = SPGArgs()
    spg = SimplePolicyGradient(args)
    spg.train_agent()