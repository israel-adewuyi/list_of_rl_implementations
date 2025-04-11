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

    def __repr__(self, ):
        return f"EpsilonGreedyAgent_eps={self.epsilon}_initial_value={self.initial_value})"


class UCBAgent(Agent):
    """Agent that chooses actions according to the Upper Confidence Bound action selection method
    \(A_t = \text{argmax}_a[Q(a) + c \sqrt{\frac{ln t}{N(a)}}] \)
    """
    def __init__(self, num_arms: int, seed: int, c: float) -> None:
        self.c = c
        super().__init__(num_arms, seed)
        
    def get_action(self, ) -> ActType:
        action = int(np.argmax(self.Q + self.c * np.sqrt(np.log2(self.t) / (self.N + 1e-8))))
        return action
    
    def observe(self, action: ActType, reward: float, info: dict) -> None:
        self.t += 1
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        
    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        self.t = 1
        self.Q = np.zeros((self.num_arms))
        self.N = np.zeros((self.num_arms))

    def __repr__(self, ) -> None:
        return f"UCBAgent_c={self.c}"
    
if __name__ == "__main__":
    agent = RandomAgent(10, 42)
    agent2 = EpsilonGreedyAgent(10, 0, 0.1, 4)
    print(agent.get_action())