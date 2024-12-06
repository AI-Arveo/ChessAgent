import chess
from project.chess_utilities.utility import Utility

class ExampleUtility(Utility):

    def __init__(self) -> None:
        pass
    
    # Calculate the amount of white pieces minus the amount of black pieces    
    def board_value(self, board: chess.Board):
        n_white = 0
        n_white += len(board.pieces(piece_type=chess.PAWN, color=chess.WHITE))
        n_white += len(board.pieces(piece_type=chess.BISHOP, color=chess.WHITE))
        n_white += len(board.pieces(piece_type=chess.KNIGHT, color=chess.WHITE))
        n_white += len(board.pieces(piece_type=chess.ROOK, color=chess.WHITE))
        n_white += len(board.pieces(piece_type=chess.QUEEN, color=chess.WHITE))

        n_black = 0
        n_black += len(board.pieces(piece_type=chess.PAWN, color=chess.BLACK))
        n_black += len(board.pieces(piece_type=chess.BISHOP, color=chess.BLACK))
        n_black += len(board.pieces(piece_type=chess.KNIGHT, color=chess.BLACK))
        n_black += len(board.pieces(piece_type=chess.ROOK, color=chess.BLACK))
        n_black += len(board.pieces(piece_type=chess.QUEEN, color=chess.BLACK))
        return n_white - n_black


