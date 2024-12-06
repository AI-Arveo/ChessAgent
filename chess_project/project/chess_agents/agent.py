from abc import ABC
import chess
from project.chess_utilities.utility import Utility

"""A generic agent class"""


class Agent(ABC):

    def __init__(self, utility: Utility, time_limit_move: float) -> None:
        """Setup the Search Agent"""
        self.utility = utility
        self.time_limit_move = time_limit_move

    def calculate_move(self, board: chess.Board):
        pass
