from torch import Tensor
from jaxtyping import Float
import torch.nn as nn

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
    obs: Float[Tensor, "sample_size, *obs_shape"]
    actions: Float[Tensor, "sample_size"] # Why action shape?? 
    rewards: Float[Tensor, "sample_size"]
    terminated: Float[Tensor, "sample_size"]
    next_obs: Float[Tensor, "sample_size, obs_shape"]

if __name__ == "__main__":
    net= QNetwork(obs_shape=(4,), num_actions=2)
    print(f"Number of parameters in the current QNetwork is {sum(param.nelement() for param in net.parameters())}")

    