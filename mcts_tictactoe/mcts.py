import math
import random

from typing import List, Optional
from utils import generate_random_board


class Board:
    """This class manages game states, check for wins / draws and generates legal moves"""

    def __init__(self, board: List[Optional[str]], current_player: str = "X") -> None:
        self.board = board if board is not None else [" " for _ in range(9)] 
        self.current_player = current_player
        self.next_player = "X" if current_player == "O" else "O"

    def is_winning_state(self, player: str = None) -> bool:
        player = player if player else self.current_player

        flag = False
        for idx in range(0, 9, 3):
            flag |= self._check_row_win(idx, player)
        for idx in range(3):
            flag |= self._check_col_win(idx, player)
        flag |= self._check_diag_win(player)
        # print(flag)
        return flag

    def is_draw_state(self) -> bool:
        countx = sum(player == "X" for player in self.board)
        counto = sum(player == "O" for player in self.board)
        countempty = sum(player == " " for player in self.board)
        return (countempty == 0)

    def _check_row_win(self, idx: int, player: str) -> bool:
        assert idx % 3 == 0, f"The first index in a row should be a multiple of 3, but getting {idx}"
        return self.board[idx] == self.board[idx + 1] == self.board[idx + 2] == player
    def _check_col_win(self, idx: int, player: str) -> bool:
        assert idx < 3, f"The first index in a col should be less than of 3, but getting {idx}"
        return self.board[idx] == self.board[idx + 3] == self.board[idx + 6] == player
    def _check_diag_win(self, player: str) -> bool:
        return (self.board[0] == self.board[4] == self.board[8] == player) or \
                (self.board[2] == self.board[4] == self.board[6] == player)

    def is_game_over(self, ) -> bool:
        """Checks if the game is over, with the current board state

        Returns:
            bool: _description_
        """
        return (self.is_winning_state(self.current_player) or
                self.is_winning_state(self.next_player) or
                len(self.get_legal_moves()) == 0)

    def make_move(self, player: str, position: int) -> None:
        """Makes a move by placing the player's str. 

        Args:
            player (_type_): _description_
        """
        assert self.board[position] == " ", "Entry to be moved to should be empty"
        self.board[position] = player
        self.current_player = "O" if player is "X" else "X"
        self.next_player = player

    def get_legal_moves(self, ) -> List[int]:
        """Gets a list of position on the current board that is unoccupied i.e value at this position is None

        Returns:
            legal_moves: list of legal moves
        """
        legal_moves = []
        for idx, position in enumerate(self.board):
            if position == " ":
                legal_moves.append(idx)
        return legal_moves

    def __repr__(self) -> str:
        return f"{self.board[0 : 3]} \n{self.board[3 : 6]} \n{self.board[6 : 9]}"    

class Node:
    """Represents each state in the MCTS tree, tracking visits, rewards, and untried moves."""
    
    def __init__(self, board_state: Board=None, parents: Board=None) -> None:
        self.state = board_state
        self.parents = parents
        self.children = []
        self.untried_moves = self.state.get_legal_moves()
        self.visits = 0
        self.wins = 0
        
    def make_move(self, position: int) -> "Node":
        new_state = Board(self.state.board.copy(), self.state.current_player)
        new_state.make_move(self.state.current_player, position)
        new_node = Node(new_state, parents=self)
        self.children.append(new_node)
        self.untried_moves.remove(position)
        return new_node # TODO: Why am I returning this? 

class MCTS:
    """Handles the MCTS logic (Selection, Expansion, Simulation, Backpropagation)"""

    def __init__(self, root: Node, c: float = 0.99):
        self.root = root
        self.c = c

    def selection(self, node: Node) -> Node:
        while node.children and not node.untried_moves:
            node = max(node.children, key=self._UCB)
        return node

    def _UCB(self, node: Node) -> float:
        if node.visits == 0:
            return float('inf')
        parent_visits = node.parents.visits if node.parents else self.root.visits
        return (node.wins / node.visits) + self.c * math.sqrt(math.log(parent_visits) / node.visits)


    def expansion(self, node: Node) -> Node:
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            return node.make_move(move)
        return node

    def simulation(self, node: Node) -> int:
        if node.state.is_game_over():
            if node.state.is_winning_state(self.root.state.current_player):
                return 1
            elif node.state.is_winning_state(self.root.state.next_player):
                return -1
            return 0

        sim_board = Board(node.state.board.copy(), node.state.current_player)
        # Simulate until game over
        while not sim_board.is_game_over():
            legal_moves = sim_board.get_legal_moves()
            move = random.choice(legal_moves)
            sim_board.make_move(sim_board.current_player, move)

        # Evaluate outcome relative to root player
        if sim_board.is_winning_state(self.root.state.current_player):
            return 1
        elif sim_board.is_winning_state(self.root.state.next_player):
            return -1
        return 0

    def backpropagation(self, ):
        pass

    def run(self, num_iterations: int):
        for _ in range(num_iterations):
            selected_node = self.selection(self.root)
            new_node = self.expansion(selected_node)
            # select node to explore and expand it
            reward = self.simulation(new_node)
            # simulate game play till game over
            # do backpropagation``


if __name__ == "__main__":
    board = generate_random_board(include_space=False, seed=2)
    # board = ['X', 'O', 'X', 'O', 'O', 'X', 'X', 'X', 'O']
    print("Random board generated: ", board)
    init_board = Board(board=board, current_player="O")
    print(init_board.__repr__())
    # print(init_board.is_winning_state("O"))
    # print(init_board.is_winning_state("X"))
    # print(init_board.is_draw_state())
    
    root = Node(init_board)
    mcts = MCTS(root)
    mcts.run(1)

    # temp_node = Node(board_state=init_board)
    # print(temp_node.make_move())