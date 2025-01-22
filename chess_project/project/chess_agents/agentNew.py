import chess
from chess_project.project.chess_agents.agent import Agent
import time
from chess_project.project.chess_utilities.utility import Utility
from chess_project.project.chess_openings.ReadBin import Read_Book
#from chess_project.project.utils.chess_utils import hash
from chess.polyglot import zobrist_hash


class TranspositionEntry:
    def __init__(self, flag: str, value: float, depth: int, sorted_moves: list[chess.Move]) -> None:
        """
        Represents an entry in the transposition table.

        :param flag: The flag indicating the type of entry (e.g., exact, lower bound, upper bound).
        :param value: The evaluation value of the position.
        :param depth: The search depth at which the entry was recorded.
        :param sorted_moves: A list of sorted moves for this position.
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
        Calculate the best move for the current position using opening books, tablebases, or search.
        :param board: Current chess board.
        :return: Best move found.
        """
        start_time = time.time()

        #1: Use opening book if in the opening phase
        opening_move = self.get_opening_move(board)
        if opening_move:
            return opening_move

        #2: Use syzygy tablebase if in the endgame
        endgame_move = self.get_endgame_move(board)
        if endgame_move:
            return endgame_move

        #3: If not in previous, Perform search using iterative deepening and negamax
        return self.perform_search(board, start_time)

    def get_opening_move(self, board: chess.Board) -> chess.Move | None:
        """
        Attempt to retrieve a move from the opening book.
        :param board: Current chess board.
        :return: Move from opening book or None if not available.
        """
        opening_move = self.Read_Book.Opening(board)
        if opening_move:
            print(f"Opening move: {opening_move}")
        return opening_move

    def get_endgame_move(self, board: chess.Board) -> chess.Move | None:
        """
        Attempt to retrieve a move from the endgame tablebase.
        :param board: Current chess board.
        :return: Move from tablebase or None if not applicable.
        """
        if len(board.piece_map()) <= 5:
            endgame_move = self.use_tablebase(board)
            if endgame_move:
                print(f"Endgame move (Syzygy): {endgame_move}")
                return endgame_move
        return None

    def perform_search(self, board: chess.Board, start_time: float) -> chess.Move:
        """
        Perform a search using iterative deepening and negamax.
        :param board: Current chess board.
        :param start_time: Time when the search started.
        :return: Best move found within the time limit.
        """
        possible_moves = list(board.legal_moves)  # Get list of all legal moves from the current board position
        print(f"{len(possible_moves)} legal moves were found")

        best_move = None
        flip_value = 1 if board.turn == chess.WHITE else -1 # Positive for white, negative for black
        best_value = float("-inf")
        depth = 1
        alpha, beta = float("-inf"), float("inf")

        while time.time() - start_time < self.time_limit_move:
            for move in self.sortMoves(board):
                if time.time() - start_time > self.time_limit_move:
                    break

                board.push(move)
                move_value = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()

                # Update the best move if a better value is found
                if move_value > best_value:
                    best_value = move_value
                    best_move = move

                # Update alpha for alpha-beta pruning
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break

            depth += 1

        print(f"Best move: {best_move} with value: {best_value}")
        return best_move

    def sortMoves(self, board: chess.Board) -> list[chess.Move]:
        """
        Sort legal moves based on their evaluation values.

        This function evaluates each legal move by temporarily making the move on the board,
        calculating the board's evaluation value using the utility function, and then sorting the moves
        based on these evaluations. This helps in prioritizing the most promising moves for search
        algorithms like MiniMax or alpha-beta pruning.

        :param board: The current chess board.
        :return: A list of legal moves sorted by their evaluation values.
        """
        movesWithEval = []  # List to store moves paired with their evaluation values

        # Evaluate each legal move
        for move in board.legal_moves:
            board.push(move) # Make the move on the board temporarily
            move_value = self.utility.board_value(board) # Evaluate the board position after the move
            movesWithEval.append((move, move_value)) # Store the move along with its evaluation value
            board.pop() # Undo the move to restore the board state

        # Sort the moves based on their evaluation values
        # If it's white's turn, sort in descending order (higher values are better for white)
        # If it's black's turn, sort in ascending order (lower values are better for black)
        movesWithEval.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)

        # Return only the moves, discard the evaluation values
        return [move for move, _ in movesWithEval]

    def minimax(self, board: chess.Board, depth: int, is_maximizing: bool, alpha: float = float("-inf"),
                beta: float = float("inf")) -> float:
        """
        Perform the MiniMax algorithm with alpha-beta pruning.

        :param board: Current chess board.
        :param depth: Depth to search.
        :param is_maximizing: True if the current player is maximizing, False otherwise.
        :param alpha: Best score the maximizing player can guarantee.
        :param beta: Best score the minimizing player can guarantee.
        :return: Evaluation value of the board.
        """
        # Check if we have reached the maximum depth or a terminal state
        if depth == 0 or board.is_game_over():
            return self.utility.board_value(board)

        # Maximizing player's turn (White)
        if is_maximizing:
            max_value = float("-inf")
            for move in self.sortMoves(board):
                board.push(move)
                value = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                # Alpha-beta pruning: Stop searching if we found a better move
                if beta <= alpha:
                    break
            return max_value

        # Minimizing player's turn (Black)
        else:
            min_value = float("inf")
            for move in self.sortMoves(board):
                board.push(move)
                value = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                min_value = min(min_value, value)
                beta = min(beta, value)
                # Alpha-beta pruning: Stop searching if the maximizing player already found a better move
                if beta <= alpha:
                    break
            return min_value