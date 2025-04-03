from typing import List, Optional

class Board:
    """This class manages game states, check for wins / draws and generates legal moves"""

    def __init__(self, board: List[Optional[int]], current_player: str = "X") -> None:
        self.board = board if board is not None else [0 for _ in range(9)] # TODO: Might revisit this.
        self.current_player = current_player

    def is_winning_state(self, player: str = None) -> bool:
        player = player if player else self.current_player

        flag = True
        for idx in range(0, 9, 3):
            flag |= self._check_row_win(idx, player)
        for idx in range(3):
            flag |= self._check_col_win(idx, player)
        flag |= self._check_diag_win(player)

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

    def is_game_over(self, ):
        pass

    def make_move(self, player):
        pass

    def get_legal_moves(self, ):
        pass
        

class Node:
    """Represents each state in the MCTS tree, tracking visits, rewards, and untried moves."""


class MCTS:
    """Handles the MCTS logic (Selection, Expansion, Simulation, Backpropagation)"""
    pass





if __name__ == "__main__":
    init_board = Board()