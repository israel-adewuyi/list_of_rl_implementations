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
    ) -> None: #TODO: verify that type of terminal is a np array. Could be just list
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = terminal

        self.T, self.S = self.build()

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
        return NotImplementedError()

    def render(self, ):
        return NotImplementedError()

    def out_pad(self, ):
        return NotImplementedError()



class Norvig(Environment):
    def dynamics(self, state: int, action: int):
        pass

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
        
        start #TODO: Calculate start, based on height and width

        super().__init__(self.num_states, self.num_actions)
        self.T, self.S = self.build()