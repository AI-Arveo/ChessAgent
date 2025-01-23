import chess
from chess_project.project.chess_agents.agent import Agent
import time
from chess_project.project.chess_utilities.utility import Utility
from chess_project.project.chess_openings.ReadBin import Read_Book
from chess.polyglot import zobrist_hash


class TranspositionEntry:
    def __init__(self, flag: str, value: float, depth: int, sorted_moves: list[chess.Move]) -> None:
        """
        Represents an entry in the transposition table.

        :param flag: Type of entry (e.g., exact, lower bound, upper bound).
        :param value: Evaluation score of the position.
        :param depth: Search depth of the entry.
        :param sorted_moves: List of sorted moves for this position.
        """
        self.flag = flag
        self.value = value
        self.depth = depth
        self.sorted_moves = sorted_moves


class MiniMaxAgent(Agent):
    def __init__(self, utility: Utility, time_limit_move: float) -> None:
        super().__init__(utility, time_limit_move)
        self.name = "ChessAgent"
        self.author = "MatteoArvo"
        self.beginning = True
        self.Read_Book = Read_Book()
        self.transpositionTable: dict[int, TranspositionEntry] = dict()

    def calculate_move(self, board: chess.Board) -> chess.Move:
        """
        Calculates the best move for the current position.

        :param board: Current chess board state.
        :return: The best move found.
        """
        start_time = time.time()

        # 1. Use opening book for opening moves
        opening_move = self.get_opening_move(board)
        if opening_move:
            return opening_move

        # 2. Use endgame tablebases for positions with 5 or fewer pieces
        endgame_move = self.get_endgame_move(board)
        if endgame_move:
            return endgame_move

        # 3. Use iterative deepening and minimax for general play
        return self.perform_search(board, start_time)

    def get_opening_move(self, board: chess.Board) -> chess.Move | None:
        """
        Get a move from the opening book.

        :param board: Current chess board state.
        :return: An opening move if available, otherwise None.
        """
        opening_move = self.Read_Book.Opening(board)
        if opening_move:
            print(f"Opening move: {opening_move}")
        return opening_move

    def get_endgame_move(self, board: chess.Board) -> chess.Move | None:
        """
        Get a move from the endgame tablebase if applicable.

        :param board: Current chess board state.
        :return: An endgame move if available, otherwise None.
        """
        if len(board.piece_map()) <= 5:
            endgame_move = self.use_tablebase(board)
            if endgame_move:
                print(f"Endgame move: {endgame_move}")
                return endgame_move
        return None

    def perform_search(self, board: chess.Board, start_time: float) -> chess.Move:
        """
        Search for the best move using iterative deepening and minimax.

        :param board: Current chess board state.
        :param start_time: Start time of the search.
        :return: The best move found within the time limit.
        """
        possible_moves = list(board.legal_moves)
        print(f"{len(possible_moves)} legal moves found.")

        best_move = None
        best_value = float("-inf")
        depth = 1
        alpha, beta = float("-inf"), float("inf")
        flip_value = True if board.turn == chess.WHITE else False  # Positive for white, negative for black

        while time.time() - start_time < self.time_limit_move:
            for move in self.sortMoves(board):
                if time.time() - start_time > self.time_limit_move:
                    break

                board.push(move)
                move_value = self.minimax(board, depth - 1, flip_value, alpha, beta)
                board.pop()

                if move_value > best_value:
                    best_value = move_value
                    best_move = move

                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break

            depth += 1

        print(f"Best move: {best_move}, value: {best_value}")
        return best_move

    def sortMoves(self, board: chess.Board) -> list[chess.Move]:
        """
        Sort legal moves by their evaluation values to prioritize better moves.

        :param board: Current chess board state.
        :return: A sorted list of moves.
        """
        moves_with_eval = []

        for move in board.legal_moves:
            board.push(move)
            move_value = self.utility.board_value(board)
            moves_with_eval.append((move, move_value))
            board.pop()

        moves_with_eval.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        return [move for move, _ in moves_with_eval]

    def minimax(self, board: chess.Board, depth: int, is_maximizing: bool, alpha: float = float("-inf"),
                beta: float = float("inf")) -> float:
        """
        MiniMax algorithm with alpha-beta pruning.

        :param board: Current chess board state.
        :param depth: Depth to search.
        :param is_maximizing: True if the current player is maximizing, False otherwise.
        :param alpha: Best score the maximizing player can guarantee.
        :param beta: Best score the minimizing player can guarantee.
        :return: The evaluation value of the board.
        """
        if depth == 0 or board.is_game_over():
            return self.utility.board_value(board)

        if is_maximizing:
            max_value = float("-inf")
            for move in self.sortMoves(board):
                board.push(move)
                value = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return max_value
        else:
            min_value = float("inf")
            for move in self.sortMoves(board):
                board.push(move)
                value = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_value
