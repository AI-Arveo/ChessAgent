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


def encode_moves(move: chess.Move, board: chess.Board) -> int:
    """
    Converteer een schaakzet naar een unieke numerieke index op basis van de legale zetten van het bord.

    :param move: De zet die moet worden gecodeerd.
    :param board: Het huidige bord waarvan de legale zetten worden gebruikt.
    :return: Een numerieke index die overeenkomt met de zet.
    """
    # Haal de lijst van legale zetten
    legal_moves = list(board.legal_moves)

    # Maak een mapping van zetten naar indices
    move_to_index = {legal_move: idx for idx, legal_move in enumerate(legal_moves)}

    # Controleer of de zet legaal is en retourneer de index
    if move in move_to_index:
        return move_to_index[move]
    else:
        raise ValueError(f"Zet {move} is niet legaal op het huidige bord.")


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


def create_input_for_nn(board: chess.Board) -> list[float]:
    """
    Converts a chess board state into a flat list of features suitable for input to a neural network.
    Each square of the board is represented numerically, along with game-specific metadata.

    Features:
    - 64 board squares: Encoded with piece type and color.
      - 0: Empty square
      - 1: White pawn, 2: White knight, ..., 6: White king
      - -1: Black pawn, -2: Black knight, ..., -6: Black king
    - 1 turn indicator: 1 for white, -1 for black.
    - 4 castling rights: [white kingside, white queenside, black kingside, black queenside]
    - 1 en passant target: Encoded as a square index (0-63) or -1 if no target.
    - 1 half-move clock: Number of half-moves since the last pawn move or capture.
    - 1 full-move number: Current move number.

    Total feature size: 64 + 1 + 4 + 1 + 1 + 1 = 72.
    """
    features = []

    # Encode board squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_value = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
        else:
            piece_value = 0
        features.append(piece_value)

    # Encode turn indicator
    features.append(1 if board.turn == chess.WHITE else -1)

    # Encode castling rights
    features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)

    # Encode en passant square
    en_passant_square = board.ep_square
    features.append(en_passant_square if en_passant_square is not None else -1)

    # Encode half-move clock
    features.append(board.halfmove_clock)

    # Encode full-move number
    features.append(board.fullmove_number)

    return features

def decode_move(move_index, move_to_int):
    """
    Decode a move index back to its UCI string format.
    :param move_index: Numerical index of the move.
    :param move_to_int: Dictionary mapping UCI moves to indices.
    :return: Decoded move in UCI format.
    """
    int_to_move = {v: k for k, v in move_to_int.items()}
    return int_to_move[move_index]
