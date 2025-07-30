import time
import torch
import wandb
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float, Int
import torch.nn as nn
from tqdm import tqdm
from utils import (
    linear_schedule,
    get_episode_data_from_infos,
    set_global_seeds,
    make_env
    )

@dataclass
class DQNArgs:
    # Basic / global
    seed: int = 1
    env_id: str = "CartPole-v1"
    num_envs: int = 1

    # Wandb / logging
    use_wandb: bool = False
    wandb_project_name: str = "DQNCartPole"
    wandb_entity: str | None = None
    video_log_freq: int | None = 50

    # Duration of different phases / buffer memory settings
    total_timesteps: int = 500_000
    steps_per_train: int = 10
    trains_per_target_update: int = 100
    buffer_size: int = 10_000

    # Optimization hparams
    batch_size: int = 32
    learning_rate: float = 2.5e-4

    # RL-specific
    gamma: float = 0.99
    exploration_fraction: float = 0.2
    start_e: float = 1.0
    end_e: float = 0.1

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        assert self.total_timesteps - self.buffer_size >= self.steps_per_train
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // self.steps_per_train
        self.video_save_path = "videos"
    
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
    obs: Float[Tensor, "sample_size *obs_shape"]
    actions: Float[Tensor, "sample_size"] # Why action shape?? 
    rewards: Float[Tensor, "sample_size"]
    terminated: Float[Tensor, "sample_size"]
    next_obs: Float[Tensor, "sample_size obs_shape"]

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
        obs: Float[Tensor, "sample_size *obs_shape"],
        actions: Float[Tensor, "sample_size *action_shape"], # Why action shape?? 
        rewards: Float[Tensor, "sample_size"],
        terminated: Float[Tensor, "sample_size"],
        next_obs: Float[Tensor, "sample_size obs_shape"]
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

def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv,
    q_network: QNetwork,
    rng: np.random.Generator,
    obs: Float[np.ndarray, "num_envs *obs_shape"],
    epsilon: float,
    device: torch.device
) -> Int[np.ndarray, "num_envs *action_shape"]: 
    """
    Take random actions with probability epsilon and greedy actions
    the rest of the time
    """
    obs = torch.from_numpy(obs).to(device)

    num_actions = envs.single_action_space.n
    if rng.random() < epsilon:
        return rng.integers(0, num_actions, size=(envs.num_envs, ))
    else:
        q_scores = q_network(obs)
        return q_scores.argmax(-1).detach().cpu().numpy() 


class DQNAgent:
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        q_network: QNetwork,
        start_eps: float,
        end_eps: float,
        exploration_fraction: float,
        total_timesteps: int,
        buffer: ReplayBuffer,
        rng: np.random.Generator,
        args: DQNArgs,
    ):
        self.envs = envs
        self.q_network = q_network
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.buffer = buffer
        self.rng = rng
        self.args = args

        self.step = 0
        self.obs, _ = self.envs.reset()
        self.epsilon = start_eps

    def play_step(self) -> dict:
        actions = self.get_actions(self.obs)
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        true_next_obs = next_obs.copy()
        for n in range(self.envs.num_envs):
            if (terminated | truncated)[n]:
                true_next_obs[n] = infos["final_observation"][n]

        self.buffer.add(self.obs, actions, rewards, terminated, true_next_obs)
        self.step += self.envs.num_envs
        return infos

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        self.epsilon = linear_schedule(self.step, self.start_eps, self.end_eps, self.exploration_fraction,
                                       self.total_timesteps)
        return epsilon_greedy_policy(self.envs, self.q_network, self.rng, obs, self.epsilon, self.args.device)

class DQNTrainer:
    def __init__(self, args: DQNArgs):
        set_global_seeds(args.seed)
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(idx=idx, run_name=self.run_name, **args.__dict__) for idx in range(args.num_envs)]
        )

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        num_envs = self.envs.num_envs
        action_shape = self.envs.single_action_space.shape
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        assert action_shape == ()

        # Create our replay buffer
        self.buffer = ReplayBuffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed)

        # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
        self.q_network = QNetwork(obs_shape, num_actions).to(args.device)
        self.target_network = QNetwork(obs_shape, num_actions).to(args.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=args.learning_rate)

        # Create our agent
        self.agent = DQNAgent(
            envs=self.envs,
            buffer=self.buffer,
            q_network=self.q_network,
            start_eps=args.start_e,
            end_eps=args.end_e,
            exploration_fraction=args.exploration_fraction,
            total_timesteps=args.total_timesteps,
            rng=self.rng,
            args=args
        )

    def add_to_replay_buffer(self, n: int, verbose: bool = False):
        """
        Takes n steps with the agent, adding to the replay buffer (and logging any results). 
        Should return a dict of data from the last terminated episode, if any.

        Optional argument `verbose`: if True, we can use a progress bar (useful to check how long 
        the initial buffer filling is taking).
        """
        data = None

        for _ in tqdm(range(n), disable=not verbose, desc="Filling replay buffer"):
            infos = self.agent.play_step()
            data = data or get_episode_data_from_infos(infos)

        return data

    def prepopulate_replay_buffer(self):
        """
        Called to fill the replay buffer before training starts.
        """
        n_steps_to_fill_buffer = self.args.buffer_size // self.args.num_envs
        self.add_to_replay_buffer(n_steps_to_fill_buffer, verbose=True)

    def training_step(self, step: int) -> Float[Tensor, ""]:
        """
        Samples once from the replay buffer, and takes a single training step. The `step` argument 
        is used to track the number of training steps taken.
        """
        batch = self.buffer.sample(self.args.batch_size, self.args.device)

        with torch.inference_mode():
            target_max = self.target_network(batch.next_obs).max(-1).values

        predicted_q_values = self.q_network(batch.obs)[range(len(batch.actions)), batch.actions]

        td_error = batch.rewards + self.args.gamma * (1 - batch.terminated.float()) * target_max - predicted_q_values

        loss = td_error.pow(2).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if step % self.args.trains_per_target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.args.use_wandb:
            wandb.log({
                "td_loss": loss,
                "q_values": predicted_q_values.mean().item(),
                "epsilon": self.agent.epsilon
                },
                step=self.agent.step,
            )


    def train(self) -> None:
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            wandb.watch(self.q_network, log="all", log_freq=50)

        self.prepopulate_replay_buffer()

        pbar = tqdm(range(self.args.total_training_steps))
        last_logged_time = time.time()  # so we don't update the progress bar too much

        for step in pbar:
            data = self.add_to_replay_buffer(self.args.steps_per_train)
            if data is not None and time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(**data)

            self.training_step(step)

        self.envs.close()
        if self.args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    net= QNetwork(obs_shape=(4,), num_actions=2)
    print(f"Number of parameters in the current QNetwork is {sum(param.nelement() for param in net.parameters())}")
    args = DQNArgs(total_timesteps=400_000)  # changing total_timesteps will also change ???
    trainer = DQNTrainer(args)
    trainer.train()
    