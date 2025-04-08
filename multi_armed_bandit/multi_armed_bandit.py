import numpy as np
import gymnasium as gym

from typing import Optional

ActType: int
ObsType: int

class MultiArmedBandit(gym.Env):
    """MultiArmedBandit environment"""
    
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    rewards: np.ndarray
    
    def __init__(self, seed: int, num_arms: int):
        self.num_arms = num_arms
        self.action_space = gym.spaces.Discrete(num_arms)
        self.observation_space = gym.spaces.Discrete(1)
        self.reset(seed)

    def step(self, action: ActType):
        if not self.action_space.contain(action):
            return
            
        reward = self.np_random.normal(loc=self.action_space[action], scale=1.0)
        obs = 0
        terminated = False
        truncated = False
        info = {"best_arm":self.best_arm}

        return (obs, reward, terminated, truncated, info)
        
    def render(self, ):
        pass
        
    def reset(self, seed: int):
        super().reset(seed)
        self.rewards = self.np_random.normal(loc=0.0, scale=1.0, size=self.num_arms)
        self.best_arm = int(np.argmax(self.rewards))

        obs = 0
        info = {"best_arm":self.best_arm}

        return obs, info