import numpy as np

from typing import Optional

ActType = int

class Agent:
    """Base class for multi-armed bandit agents"""
    
    rng: np.random.Generator
    def __init__(self, num_arms: int, seed: int) -> None:
        self.num_arms = num_arms
        self.reset(seed)
        
    def get_action(self, ) -> ActType:
        return NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass
        
    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

class RandomAgent(Agent):
    """Agent that chooses an action (an arm) randomly"""
    
    def get_action(self, ) -> ActType:
        return np.random.choice(self.num_arms)


class EpsilonGreedyAgent(Agent):
    """Agent that chooses action according to the epsilon-greedy action selection method
        \(Q_{n + 1} = Q_n + \frac{1}{n} \cdot [R_n + Q_n] \)
    """
    def __init__(self, num_arms: int, seed: int, epsilon: float, initial_value: int) -> None:
        self.epsilon = epsilon
        self.initial_value = initial_value

        super().__init__(num_arms, seed)

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        
        
    def get_action(self, ) -> ActType:
        prob = self.rng.random()
        
        if prob < self.epsilon:
            action = self.rng.integers(0, self.num_arms)
        else:
            action = int(np.argmax(self.Q))

        return action
            

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.Q = np.full((self.num_arms, ), self.initial_value, dtype=float)
        self.N = np.zeros((self.num_arms))



if __name__ == "__main__":
    agent = RandomAgent(10, 42)
    agent2 = EpsilonGreedyAgent(10, 0, 0.1, 4)
    print(agent.get_action())