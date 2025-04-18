import einops
import argparse
import numpy as np

from env import Environment, Norvig


def value_iteration_loop(
    env: Environment,
    eps: float,
    gamma: int
) -> np.ndarray:
    """Performs value iteration to find optimal policy in a gridworld environment.

    Args:
        env: Environment object containing state/action dynamics
        eps: Convergence threshold for value iteration
        gamma: Discount factor for future rewards

    Returns:
        Optimal policy as array of actions for each state
    """
    num_states = env.num_states
    V, policy = np.zeros(num_states), np.zeros(num_states)
    transitions = env.T
    rewards = env.T

    while True:
        v = V.copy()
        for state in range(num_states):
            action_values = einops.einsum(
                transitions[state, :, :], (rewards[state, :, :] + gamma * v),
                "n_actions n_states, n_actions n_states -> n_actions"
            )
            V[state] = np.max(action_values)

        if np.max(abs(v - V)) < eps:
            break

    for state in range(num_states):
        action_value = einops.einsum(
            transitions[state, :, :], (rewards[state, :, :] + gamma * V),
            "num_actions num_states, num_actions num_states -> num_actions"
        )
        policy[state] = np.argmax(action_value)
    return policy.astype(int)


def main(height: int, width: int, penalty: int) -> None:
    """Main function - Run value iteration on a grid world of specified dimensions."""
    norvig = Norvig(height, width, penalty)
    pi = value_iteration_loop(norvig, 0.00001, 0.99)
    norvig.render(pi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run value iteration on grid world env")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--penalty", type=float, default=-0.01)

    args = parser.parse_args()
    main(args.height, args.width, args.penalty)
        