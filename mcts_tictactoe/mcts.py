from typing import List

class Board:
    """This class manages game states, check for wins / draws and generates legal moves"""
    def __init__(self, board: List[int]) -> None:
        self.board = board
        

class Node:
    """Represents each state in the MCTS tree, tracking visits, rewards, and untried moves."""
    pass

class MCTS:
    """Handles the MCTS logic (Selection, Expansion, Simulation, Backpropagation)"""
    pass





if __name__ == "__main__":
    init_board = Board()