import einops
import numpy as np
from env import Environment, Norvig

def policy_evaluation(
    env: Environment,
    pi: np.ndarray,
    gamma: float = 0.99,
    eps =1e-8,
    max_iterations: int=10_000
) -> np.ndarray:
    """
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Args:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
        max_iterations: int - Maximum number of iterations to run
    Outputs:
        value : float (num_states,) - The value function for policy pi
    """
    num_states = env.num_states
    V = np.zeros((num_states))

    for _ in range(max_iterations):
        cur_v = V.copy()
        for state in range(num_states):
            action = pi[state]
            trans_probs = env.T[state, action, :]
            rewards = env.R[state, action, :]
            cur_v[state] = np.dot(trans_probs, rewards + (gamma * V))
            assert isinstance(cur_v[state], float)

        if max(abs(cur_v - V)) < eps:
            print(f"Policy eval steps - {_}")
            break

        V = cur_v.copy()
    return V

def policy_improvement(
    env: Environment,
    V: np.ndarray,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Args:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    """
    num_states = env.num_states
    new_pi = np.zeros((num_states), dtype=int)
    for state in range(num_states):
        trans_probs = env.T[state, :, :] # num_actions, num_states
        rewards = env.R[state, :, :] # num_actions, num_states
        action_value = einops.einsum(
            trans_probs, (rewards + (gamma * V)),
            "num_actions num_states, num_actions num_states -> num_actions"
        )
        new_pi[state] = np.argmax(action_value)
    return new_pi

def policy_iteration_loop(env: Environment, gamma=0.99, max_iterations=10_000):
    """Run the policy iteration loop for the 
    Args:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions representing an optimal policy
    """
    pi = np.zeros(shape=env.num_states, dtype=int)
    for _ in range(max_iterations):
        V = policy_evaluation(env, pi, gamma=gamma, eps=1e-8, max_iterations=max_iterations)
        pi_better = policy_improvement(env, V, gamma)
        if np.array_equal(pi_better, pi):
            print(_)
            return pi_better
        else:
            pi = pi_better
    print(f"Failed to converge after {max_iterations} steps.")
    return pi



if __name__ == "__main__":
    norvig = Norvig(3, 4, -0.04)
    # pi_random = np.random.randint(0, 4, (12,))
    pi_zeros = np.zeros((12, ), dtype=int)
    norvig.render(pi_zeros)
    # V = policy_eval_numerical(norvig, pi_random)

    pi = policy_iteration_loop(norvig)
    norvig.render(pi)

    # print("Got here")
    # pi_refined = policy_improvement(norvig, V, 0.99)
    # pi_reference = policy_improvement_ref(norvig, V, 0.99)

    # print(f"Got to the second checkpoint")

    # if np.allclose(pi_refined, pi_reference):
    #     print("Both policies are the same")
    # else:
    #     print("They are so f-ing different")
    #     print(pi_refined)
    #     print(pi_reference)

    # print(f"Got to the third checkpoint")

