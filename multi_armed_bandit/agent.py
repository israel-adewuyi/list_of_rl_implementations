import numpy as np

ActType = int

class Agent:
    """Base class for multi-armed bandit agents"""
    
    rng: np.random.Generator
    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)
        
    def get_actions(self, ) -> ActType:
        return NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass
        
    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

class RandomAgent(Agent):
    """Agent that chooses an action (an arm) randomly"""
    
    def get_actions(self, ) -> ActType:
        return np.random.choice(self.num_arms)



if __name__ == "__main__":
    agent = RandomAgent(10, 42)
    print(agent.get_actions())