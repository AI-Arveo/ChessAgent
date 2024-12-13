from abc import ABC
import chess
import chess.engine

class Utility(ABC):
    """
    A utility class for evaluating chess board positions using Stockfish and material evaluation.
    """
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.5,  # Bishops are slightly more versatile than knights
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100  # Extremely high value for king safety
    }

    def __init__(self):
        """
        Initialize the Utility class with a Stockfish engine.
        :param stockfish_path: Path to the Stockfish executable.
        """
        self.stockfish_path = "../../../stockfish/stockfish-windows-x86-64-avx2.exe"
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Stockfish engine: {e}")

    def board_value(self, board: chess.Board, depth: int = 15) -> float:
        """
        Evaluate the current board position using Stockfish.
        :param board: The chess board to evaluate.
        :param depth: Depth of Stockfish analysis.
        :return: A numerical score (positive favors white, negative favors black).
        """
        try:
            result = self.engine.analyse(board, chess.engine.Limit(depth=depth))
            score = result["score"].white()

            # Handle mate scenarios
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000

            # Convert centipawns to a scaled score
            return score.score() / 100.0
        except Exception as e:
            print(f"Error analyzing board with Stockfish: {e}")
            return 0  # Return 0 in case of error

    def material_value(self, board: chess.Board) -> float:
        """
        Evaluate the board based on material value.
        :param board: The chess board to evaluate.
        :return: A material evaluation score (positive favors white, negative favors black).
        """
        score = 0
        for square, piece in board.piece_map().items():
            value = self.PIECE_VALUES.get(piece.piece_type, 0)
            score += value if piece.color == chess.WHITE else -value
        return score

    def close(self):
        """
        Safely close the Stockfish engine.
        """
        try:
            self.engine.quit()
        except Exception as e:
            print(f"Failed to close Stockfish engine: {e}")
