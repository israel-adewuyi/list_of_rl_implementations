import random
from typing import List

def generate_random_board(include_space: bool = False, seed: int =42) -> List[str]:
    # random.seed(seed)
    chars = ["O", "X"]
    if include_space:
        chars.append(" ")
    board = random.choices(chars, k=9)

    return board