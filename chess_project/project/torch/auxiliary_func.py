import numpy as np
import torch
import chess


def board_to_tensor(board: chess.Board):
    """
    Convert a chess board into a tensor representation.
    :param board: chess.Board object representing the current game state.
    :return: A PyTorch tensor of shape (13, 8, 8).
    """
    # Initialize a tensor with 13 channels (12 for pieces, 1 for legal moves)
    tensor = torch.zeros((13, 8, 8), dtype=torch.float32)

    # Map pieces to channels (White: 0-5, Black: 6-11)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        channel = piece.piece_type - 1
        if not piece.color:  # Black pieces are in channels 6-11
            channel += 6
        row, col = divmod(square, 8)
        tensor[channel, row, col] = 1.0

    # Add legal moves to the 13th channel
    for move in board.legal_moves:
        to_square = move.to_square
        row, col = divmod(to_square, 8)
        tensor[12, row, col] = 1.0

    return tensor


def encode_moves(moves):
    """
    Encode a list of moves into numerical indices.
    :param moves: List of moves in UCI string format.
    :return: A numpy array of encoded moves and a move-to-index mapping dictionary.
    """
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.int64), move_to_int


def decode_move_with_probabilities(output, board):
    """
    Decode the neural network's output into a move based on probabilities.
    :param output: Neural network's raw output tensor (logits or probabilities).
    :param board: chess.Board object representing the current game state.
    :return: Best move in UCI format and its associated probability.
    """
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1).squeeze(0)

    # Sort moves by probability in descending order
    move_indices = probabilities.argsort(descending=True)

    for move_index in move_indices:
        try:
            # Map the move index to UCI format (you need a mapping in your training)
            uci_move = board.legal_moves[move_index.item()]  # Replace with your int-to-uci mapping if needed
            move = chess.Move.from_uci(uci_move)
            if move in board.legal_moves:
                return move.uci(), probabilities[move_index].item()
        except Exception:
            continue

    # Fallback: Return the first legal move if no valid move is found
    return list(board.legal_moves)[0].uci(), 0.0


def create_input_for_nn(board: chess.Board) -> torch.Tensor:
    """
    Converts a single chess board into a feature representation for the neural network.
    """
    # Example implementation: Flatten the board's piece positions
    feature_vector = torch.zeros(64, dtype=torch.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            feature_vector[square] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
    return feature_vector



def decode_move(move_index, move_to_int):
    """
    Decode a move index back to its UCI string format.
    :param move_index: Numerical index of the move.
    :param move_to_int: Dictionary mapping UCI moves to indices.
    :return: Decoded move in UCI format.
    """
    int_to_move = {v: k for k, v in move_to_int.items()}
    return int_to_move[move_index]
