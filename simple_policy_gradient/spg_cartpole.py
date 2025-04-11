import torch
import torch.nn as nn
# import gymnasium as gym


def get_policy_network(hidden_state: int):
    return nn.Sequential(
        nn.Linear(4, hidden_state),
        nn.Tanh(),
        nn.Linear(hidden_state, 2)
    )
    
if __name__ == "__main__":
    policy_net = get_policy_network(32)
    print(policy_net)

