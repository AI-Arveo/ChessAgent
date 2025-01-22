import chess
from torch import nn
import torch
from chess_project.project.chess_utilities.utility import isDraw
from enum import Enum

class PIECES(Enum):
    B_ROOK = chess.Piece.from_symbol('r')
    B_KNIGHT = chess.Piece.from_symbol('n')
    B_BISHOP = chess.Piece.from_symbol('b')
    B_QUEEN = chess.Piece.from_symbol('q')
    B_KING = chess.Piece.from_symbol('k')
    B_PAWN = chess.Piece.from_symbol('p')
    W_ROOK = chess.Piece.from_symbol('R')
    W_KNIGHT = chess.Piece.from_symbol('N')
    W_BISHOP = chess.Piece.from_symbol('B')
    W_QUEEN = chess.Piece.from_symbol('Q')
    W_KING = chess.Piece.from_symbol('K')
    W_PAWN = chess.Piece.from_symbol('P')

class Heuristic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def execute(self, board: chess.Board) -> float:
        pass

    def featureExtraction(board: chess.Board) -> torch.Tensor:
        pass

    def RelativeScore(self) -> bool:
        """
        returns the relative score or absolute score from whites perspective
        """
        return False

class NeuralNetwork(Heuristic):
    # num_moves zijn het aantal legale moves die op die beurt beschikbaar zijn
    def __init__(self, input_channels, board_size, num_moves):
        super().__init__()
        # we kiezen voor convolution layers, omdat het neuraal netwerk dan beter de spatial relationships/patterns
        # tussen pieces gaat leren

        # gebruik sequential om meerdere layers na elkaar te gebruiken
        self.layers: nn.Sequential = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1), # 1ste convolutionl layer
                                    # ReLU introduceert niet-lineariteit, waardoor het netwerk complexe functies kan leren
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # 2de convolutional layer
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                                    # 3de convolutional layer
                                    nn.ReLU(),
                                    # maxPool2d om dimensies te verkleinen
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten_size = (board_size // 2) * (board_size // 2) * 32
        self.layers: nn.Sequential = nn.Sequential(nn.Flatten(), # 64x832
                                    nn.Linear(in_features=self.flatten_size, out_features=board_size * board_size * 64),
                                    nn.Linear(in_features=board_size * board_size * 64,out_features= 128), # 1ste fully connected layer (4096x128)
                                    # er is geen activation function nodig bij lineare/fully connected lagen
                                    nn.Linear(in_features=128, out_features=512), # 2e fully connected layer
                                    nn.Linear(in_features=512, out_features=num_moves)) # 3e fully connected layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers.forward(x)
        return out  # Return raw logits

    def execute(self, board: chess.Board) -> float:
        features: torch.Tensor = self.featureOutOfBoard(board)
        out: float = self.forward(features)[0]
        return out

    def featureOutOfBoard(self, board: chess.Board) -> torch.Tensor:
        """
        This method is used to extract the features out of the given board state
        Black<0, White>0 and 0 = no piece
        We will assign values to each piece:
        - King = 10
        - Queen = 5
        - Rook = 4
        - Bishop = 3 -> We make the Bishop's value be slightly higher than the knight (which normally
        - Knight = 2 -> would also be valued at 3), because the bishop is slightly stronger
        - Pawns = 1
        """

        boardString: str = board.fen() # gives the Forsyth-Edwards Notation string
        # creates a one-dimensional tensor containing 65 values of datatype 32-bit floating point
        features: torch.Tensor = torch.Tensor([65],dtype=torch.float32)

        # color is needed to find the color of the piece
        color: int = 1
        piece: int = 0
        position: int = 0

        # manipulating the input string so it can be used for processing
        positions: str = boardString.split()[0] # take the first item in the split
        positions = positions.replace("/","")
        turnString: str = boardString.split()[1]
        turn: int

        # find who's turn it is out of the turnString
        if turnString == "w":
            turn = 10
        else:
            turn = -10

        # if in the position you find a number, it indicates that there are x empty spaces.
        # if it isn't a number -> is a piece
        for elem in positions:
            if elem.isnumeric():
                for i in range(int(elem)):
                    features[position] = 0
                    position += 1

            else:
                if elem.isupper():
                    color = 1
                else:
                    color = -1

                match elem.lower():
                    case 'k':
                        piece = 10
                    case 'q':
                        piece = 5
                    case 'r':
                        piece = 4
                    case 'b':
                        piece = 3
                    case 'n':
                        piece = 2
                    case 'p':
                        piece = 1
                    case _:
                        piece = 0

                features[position] = color * piece
                position += 1
        features[position] = turn
        return features


class FullPerspectiveHeuristic(Heuristic):
    def __init__(self) -> None:
        super().__init__()
        layer1Size = 64 * 64 * 10 * 2
        self.layerWV = nn.Sequential(
            nn.Linear(layer1Size, 512),
            nn.Linear(512, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 1)
        )

    def RelativeScore(self) -> bool:
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)

    def execute(self, board: chess.Board) -> float:
        turn = 1 if board.turn == chess.WHITE else -1
        if board.is_checkmate():
            return turn * (20000.0)
        elif isDraw(board):
            return 0.0
        features = FullPerspectiveHeuristic.featureExtraction(board)
        features = features.to('cuda' if torch.cuda.is_available() else 'cpu')
        score = self.forward(features).item()
        return score * turn

    def featureExtraction(board: chess.Board) -> torch.Tensor:
        """
        Creates a tensor of every position for every piece. Then look up the
        position of the white & black king. Converts the tensor and matrix multiplies it with
        the white king tensor. Then invert all positions to the black perspective. And do
        the same for the black side. Then calculate the white&black worldView
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        piece_map = board.piece_map()
        white_tensor_map = {piece.value: torch.zeros(64, device=device) for piece in PIECES}

        # Indicate for every piece it's position using on hot encoding
        for square, piece in piece_map.items():
            white_tensor_map[piece][square] = 1

        whiteKingTensor = white_tensor_map.pop(chess.Piece.from_symbol('K'))
        blackKingTensor = white_tensor_map.pop(chess.Piece.from_symbol('k')).flip([0])

        whiteTensor = torch.concat(list(white_tensor_map.values()))
        whiteWorldView: torch.Tensor = whiteKingTensor.outer(whiteTensor)
        blackTensor = torch.concat([tensor.flip([0]) for tensor in white_tensor_map.values()])
        blackWorldView = blackKingTensor.outer(blackTensor)

        whiteWorldView = torch.flatten(whiteWorldView)
        blackWorldView = torch.flatten(blackWorldView)

        # Concatenate the 2 values based on the turn
        positionTensor = torch.concat([whiteWorldView, blackWorldView]).to(
            "cpu") if board.turn == chess.WHITE else torch.concat([blackWorldView, whiteWorldView]).to('cpu')
        return positionTensor





