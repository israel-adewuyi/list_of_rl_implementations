import random
from typing import List

def generate_random_board(seed=42) -> List[str]:
    # random.seed(seed)
    chars = ['O', 'X']
    board = random.choices(chars, k=9)

    return board