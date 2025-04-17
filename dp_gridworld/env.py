"""Module containing environment classes for grid world implementations."""
import numpy as np

from typing import Tuple


class Environment:
    """Implementation of the base environment class
    """
    def __init__(self,
                num_states: int,
                num_actions: int,
                start: int = 0,
                terminal: np.ndarray = None
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = terminal

        self.T, self.R = self.build()

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build environment dynamics.
        
        Constructs transition probabilities and rewards for all state-action pairs.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transition probs and rewards for every state, action pair
        """
        T = np.zeros((self.num_states, self.num_actions, self.num_states))
        R = np.zeros((self.num_states, self.num_actions, self.num_states))

        for state in range(self.num_states):
            for action in range(self.num_actions):
                states, rewards, probs = self.dynamics(state, action)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[state, action, all_s] = all_p
                R[state, action, all_s] = all_r

        return T, R

    def dynamics(self, state: int, action: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns next states, rewards, and transition probabilities for a state-action pair."""
        return NotImplementedError()

    def render(self, pi: np.ndarray) -> None:
        """Visualizes the policy in the environment."""
        return NotImplementedError()

    def out_pad(self, states: np.ndarray, rewards: np.ndarray, probs: np.ndarray):
        """Pads dynamics arrays to full state space size."""
        all_s = np.arange((self.num_states))
        all_p = np.zeros((self.num_states))
        all_r = np.zeros((self.num_states))
        for idx, state in enumerate(states):
            all_p[state] += probs[idx]
            all_r[state] += rewards[idx]
        return all_s, all_r, all_p


class Norvig(Environment):
    """Implementation of Norvig's grid world environment with stochastic transitions."""
    def render(self, pi: np.ndarray) -> None:
        assert len(pi) == self.num_states
        emoji = ["⬆️", "➡️", "⬇️", "⬅️"]
        grid = [emoji[act] for act in pi]
        grid[self.terminal[0]] = "🟩"
        grid[self.terminal[1]] = "🟥"
        grid[5] = "⬛"

        grid_str = ""
        for i in range(self.height):
            grid_str += "  ".join(grid[(i * self.width) : ((i + 1) * self.width)])
            grid_str += "\n"
        print(grid_str)

    def dynamics(self, state: int, action: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))
        out_probs = np.zeros(self.num_actions) + 0.1
        out_probs[action] = 0.7
        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]
        for i, s_new in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue
            new_state = state_index(s_new)
            if new_state in self.walls:
                out_states[i] = state
            else:
                out_states[i] = new_state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]
        return (out_states, out_rewards, out_probs)

    def __init__(
        self,
        height: int,
        width: int,
        penalty: float,
    ) -> None:
        self.height = height
        self.width = width
        self.penalty = penalty
        self.num_states = self.height * self.width
        self.num_actions = 4

        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        start = self.width * (self.height - 1)
        terminal = np.array([self.width - 1, start - 1], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1])
        super().__init__(self.num_states, self.num_actions, start, terminal)

if __name__ == "__main__":
    norvig = Norvig(3, 4, -0.04)
    pi_random = np.random.randint(0, 4, (12,))
    # norvig.render(pi_random)
    # print(norvig.T[2])
    print(norvig.R[7])
