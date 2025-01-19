from abc import ABC
import chess
import torch
from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork


class Utility(ABC):
    """
    A utility class for evaluating chess board positions using the custom neural network.
    """
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.5,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100  # High value for king safety
    }

    def __init__(self, model_path, device):
        """
        Initialize the Utility class with the neural network model.
        :param model_path: Path to the trained neural network model (.pth file).
        :param device: The device to use (e.g., "cpu" or "cuda").
        """
        self.device = device
        try:
            # Load the neural network model
            input_channels = 13  # 12 for pieces, 1 for legal moves
            board_size = 8  # Chessboard size
            num_classes = 512  # Match to output classes
            self.model = NeuralNetwork(input_channels, board_size, num_classes).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()  # Set the model to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the neural network model: {e}")

    def board_value(self, board: chess.Board) -> float:
        """
        Evaluate the current board position using the neural network.
        :param board: The chess board to evaluate.
        :return: A numerical score (positive favors white, negative favors black).
        """
        try:
            # Convert the board to a tensor representation
            from chess_project.project.torch.auxiliary_func import board_to_tensor
            board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(board_tensor)

            # For now, return the sum of probabilities as a placeholder score
            probabilities = torch.softmax(output, dim=1)
            score = probabilities.sum().item()
            return score
        except Exception as e:
            print(f"Neural network evaluation failed, falling back to material evaluation: {e}")
            return self.material_value(board)

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
        Safely close resources, if any.
        """
        pass
