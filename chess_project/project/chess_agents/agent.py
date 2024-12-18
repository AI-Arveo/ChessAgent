import math
import chess
from functools import wraps

from ChessAgent.chess_project.project.Logger import BoardRepr
from ChessAgent.chess_project.project.Logger import Logger

from ChessAgent.chess_project.project.chess_utilities.utility import Utility

class Agent:
    def __init__(self, utility: Utility, time_limit_move: float) -> None:
        self.utility = utility
        self.time_limit_move = time_limit_move
        self.logger = Logger()  # Use instance-level logger

    def calculate_move(self, board: chess.Board, depth: int):
        # Wrapper for minimax to get the best move
        data = {"best_move": None}
        self.logger.clear()  # Clear previous logs
        self.minimax(board, depth, -math.inf, math.inf, True, True, data)
        self.logger.write()  # Write log to file
        return data["best_move"]

    def log_tree(func):
        @wraps(func)
        def wrapper(agent, board: chess.Board, *args, **kwargs):
            depth = args[0]  # First argument is depth
            evaluation = agent.utility.board_value(board)
            board_repr = BoardRepr(board.unicode(invert_color=True), depth, evaluation)
            agent.logger.append(board_repr)  # Log current board state
            return func(agent, board, *args, **kwargs)

        return wrapper

    @log_tree
    def minimax(self, board: chess.Board, depth: int, alpha, beta, maximizing_player: bool, save_move: bool, data: dict):
        if depth == 0 or board.is_game_over():
            evaluation = self.utility.board_value(board)
            return evaluation

        if maximizing_player:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta, False, False, data)
                board.pop()
                if evaluation > max_eval:
                    max_eval = evaluation
                    if save_move:
                        data["best_move"] = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta, True, False, data)
                board.pop()
                if evaluation < min_eval:
                    min_eval = evaluation
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval
