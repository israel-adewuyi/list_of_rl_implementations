import random
from typing import List

def generate_random_board(include_space: bool = False, seed: int =42) -> List[str]:
    # random.seed(seed)
    chars = ["O", "X", " "]
    probabilities = [0.3, 0.3, 0.4]
    if include_space:
        chars.append(" ")
    board = random.choices(chars, weights=probabilities, k=9)

    return board