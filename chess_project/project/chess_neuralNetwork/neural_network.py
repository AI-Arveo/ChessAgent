import chess
from torch import nn
import torch

class NeuralNetwork(nn.Module):
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






