from typing import List, Optional
from utils import generate_random_board


class Board:
    """This class manages game states, check for wins / draws and generates legal moves"""

    def __init__(self, board: List[Optional[str]], current_player: str = "X") -> None:
        self.board = board if board is not None else [" " for _ in range(9)] 
        self.current_player = current_player

    def is_winning_state(self, player: str = None) -> bool:
        player = player if player else self.current_player

        flag = False
        for idx in range(0, 9, 3):
            flag |= self._check_row_win(idx, player)
        for idx in range(3):
            flag |= self._check_col_win(idx, player)
        flag |= self._check_diag_win(player)
        print(flag)
        return flag

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
        if self.is_winning_state() or self.get_legal_moves() is not None:
            return True
        return False

    def make_move(self, player):
        """Makes a move by placing the player's str. 

        Args:
            player (_type_): _description_
        """
        pass

    def get_legal_moves(self, ) -> List[str]:
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
    
    def __init__(self, board_state: Board, _=None, parents: Board=None) -> None:
        self.state = board_state
        self.parents = parents
        self.children = []
        self.untried_moves = self.state.get_legal_moves()
        self.visits = 0
        self.wins = 0
        
    def make_move(self, ):
        return self.untried_moves

class MCTS:
    """Handles the MCTS logic (Selection, Expansion, Simulation, Backpropagation)"""
    
    def __init__(self, root: Node):
        # this should be of type node
        pass

    def selection(self, node: Node):
        # Should do ucb eval for the node passed to it.
        pass

    def expansion(self, ):
        # add a random node to the current node
        pass

    def simulation(self, ):
        # while(True):
            # simulate game till end
        pass

    def backpropagation(self, ):
        pass

    def run(self, num_iterations: int):
        pass
        # for _ in range(num_iterations):
            # select node to explore and expand it
            # simulate game play till game over
            # do backpropagation``


if __name__ == "__main__":
    board = generate_random_board(include_space=False, seed=2)
    # board = ['X', '', 'O', 'X', 'O', 'O', 'O', '', 'O']
    print("Random board generated: ", board)
    init_board = Board(board=board, current_player="O")
    print(init_board.__repr__())
    print(init_board.is_winning_state())
    
    
    temp_node = Node(board_state=init_board)
    print(temp_node.make_move())