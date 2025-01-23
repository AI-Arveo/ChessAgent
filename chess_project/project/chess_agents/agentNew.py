import chess
from chess_project.project.chess_agents.agent import Agent
import time
from chess_project.project.chess_utilities.utility import Utility
from chess_project.project.chess_openings.ReadBin import Read_Book
from chess.polyglot import zobrist_hash


class TranspositionEntry:
    def __init__(self, flag: str, value: float, depth: int, sorted_moves: list[chess.Move]) -> None:
        """
        Stores information about a position for reuse in the search.
        """
        self.flag = flag  # Type of entry (e.g., exact, lower bound, upper bound)
        self.value = value  # Evaluation score of the position
        self.depth = depth  # Depth of the recorded search
        self.sorted_moves = sorted_moves  # Moves considered best from this position


class MiniMaxAgent(Agent):
    def __init__(self, utility: Utility, time_limit_move: float) -> None:
        super().__init__(utility, time_limit_move)
        self.name = "ChessAgent"  # Agent name
        self.author = "MatteoArvo"  # Author name
        self.beginning = True  # Tracks if the game is in the opening phase
        self.Read_Book = Read_Book()  # Reads moves from an opening book
        self.transpositionTable: dict[int, TranspositionEntry] = dict()  # Cache for previously seen positions

    def calculate_move(self, board: chess.Board) -> chess.Move:
        """
        Determines the best move for the current position.
        """
        start_time = time.time()  # Record when the calculation starts

        # Check if an opening move is available
        opening_move = self.get_opening_move(board)
        if opening_move:
            return opening_move

        # Check if an endgame move is available
        endgame_move = self.get_endgame_move(board)
        if endgame_move:
            return endgame_move

        # If no book or tablebase move, perform a search
        return self.perform_search(board, start_time)

    def get_opening_move(self, board: chess.Board) -> chess.Move | None:
        """
        Retrieves a move from the opening book if available.
        """
        opening_move = self.Read_Book.Opening(board)  # Fetch move from the book
        if opening_move:
            print(f"Opening move: {opening_move}")  # Print for debugging
        return opening_move

    def get_endgame_move(self, board: chess.Board) -> chess.Move | None:
        """
        Retrieves a move from the endgame tablebase if applicable.
        """
        if len(board.piece_map()) <= 5:  # Check if 5 or fewer pieces are left
            endgame_move = self.use_tablebase(board)  # Fetch move from the tablebase
            if endgame_move:
                print(f"Endgame move: {endgame_move}")
                return endgame_move
        return None

    def perform_search(self, board: chess.Board, start_time: float) -> chess.Move:
        """
        Searches for the best move using iterative deepening and minimax with alpha-beta pruning.

        :param board: The current chess board state.
        :param start_time: The time when the search started.
        :return: The best move found within the time limit.
        """
        best_move = None  # Store the best move found
        best_value = float("-inf") if board.turn == chess.WHITE else float("inf")  # Initialize based on the player
        depth = 1  # Start searching at depth 1
        alpha, beta = float("-inf"), float("inf")  # Alpha-beta bounds

        print(f"{len(list(board.legal_moves))} legal moves found.")  # Debugging: Legal move count

        # Iterative deepening search
        while time.time() - start_time < self.time_limit_move:
            current_best = None  # Best move for the current depth

            for move in self.sortMoves(board):  # Sort moves to prioritize better options
                if time.time() - start_time > self.time_limit_move:
                    break  # Exit if time limit exceeded

                board.push(move)  # Make the move
                move_value = self.minimax(board, depth - 1, board.turn == chess.BLACK, alpha, beta)  # Evaluate
                board.pop()  # Undo the move

                # Maximizing for White
                if board.turn == chess.WHITE:
                    if move_value > best_value:
                        best_value = move_value
                        current_best = move
                        alpha = max(alpha, best_value)  # Update alpha
                # Minimizing for Black
                else:
                    if move_value < best_value:
                        best_value = move_value
                        current_best = move
                        beta = min(beta, best_value)  # Update beta

                # Pruning: Stop searching if the condition is met
                if beta <= alpha:
                    break

            if current_best:
                best_move = current_best  # Update the best move after completing the depth
            depth += 1  # Increase depth for the next iteration

        print(f"Best move: {best_move}, value: {best_value}, depth: {depth - 1}")  # Debugging: Print results
        return best_move

    def sortMoves(self, board: chess.Board) -> list[chess.Move]:
        """
        Sorts moves based on their evaluation values to prioritize stronger moves.
        """
        moves_with_eval = []

        for move in board.legal_moves:
            board.push(move)  # Try the move
            move_value = self.utility.board_value(board)  # Evaluate the board
            moves_with_eval.append((move, move_value))  # Save the move and its value
            board.pop()  # Undo the move

        # Sort moves by value (descending for white, ascending for black)
        moves_with_eval.sort(key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        return [move for move, _ in moves_with_eval]  # Return only the moves

    def minimax(self, board: chess.Board, depth: int, is_maximizing: bool, alpha: float = float("-inf"),
                beta: float = float("inf")) -> float:
        """
        MiniMax algorithm with alpha-beta pruning to evaluate the board position.
        """
        # Base case: Return evaluation if max depth is reached or the game is over
        if depth == 0 or board.is_game_over():
            value = self.utility.board_value(board)
            if value is None:
                value = 0.0  # Default to 0.0 if board_value returns None
            return float(value)

        if is_maximizing:  # Maximizing player's turn (White)
            max_value = float("-inf")
            for move in self.sortMoves(board):
                board.push(move)  # Try the move
                value = self.minimax(board, depth - 1, False, alpha, beta)  # Recursive search
                board.pop()  # Undo the move
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # Prune the search
            return max_value

        else:  # Minimizing player's turn (Black)
            min_value = float("inf")
            for move in self.sortMoves(board):
                board.push(move)  # Try the move
                value = self.minimax(board, depth - 1, True, alpha, beta)  # Recursive search
                board.pop()  # Undo the move
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Prune the search
            return min_value

