import numpy as np
from chess import Board
import torch

def board_to_matrix(board: Board):
    # 8x8 is the size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix

def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())  # Convert to UCI format here
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)

def encode_move_to_index(move, board):
    """
    Encode a move into a single integer index corresponding to the flattened neural network output.

    :param move: The chess move in UCI format.
    :param board: The current chess board.
    :return: An integer index for the move.
    """
    source_square = move.from_square
    target_square = move.to_square

    # Determine the piece type and color at the source square
    piece = board.piece_at(source_square)
    if piece is None:
        raise ValueError("No piece at the source square")

    piece_type = piece.piece_type - 1  # Piece types range from 1-6
    piece_color = 0 if piece.color else 6  # White: 0-5, Black: 6-11

    channel = piece_type + piece_color

    # Flatten the index for the target square
    index = (channel * 64) + target_square
    return index

def encode_moves(moves):
    """
    Create a mapping for moves and encode them as unique indices.

    :param moves: List of moves (in UCI format or convertible to UCI).
    :return: Encoded move indices and a move-to-index mapping.
    """
    # Convert moves to UCI strings if they are tensors or other formats
    moves_uci = [move.uci() if isinstance(move, torch.Tensor) else move for move in moves]

    # Create a unique mapping of moves to indices
    move_to_int = {move: idx for idx, move in enumerate(set(moves_uci))}

    # Encode moves using the mapping
    return np.array([move_to_int[move] for move in moves_uci], dtype=np.int64), move_to_int

