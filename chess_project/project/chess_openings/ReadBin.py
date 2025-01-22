import chess
import chess.polyglot
import random
import os

class Read_Book:
    def __init__(self):
        self.Path = os.path.join(os.path.dirname(__file__), '../chess_openings/baron30.bin')
        self.reader = chess.polyglot.open_reader(self.Path)
    def Opening(self, board) -> chess.Move | None:
        """
        Finds a specific position in the opening book. If it exists it will yield a random next move
        Otherwise return None.
        """
        moves = []
        try:
            for entry in self.reader.find_all(board):
                moves.append(entry.move)
        except Exception as e:
            print(f"Error opening file: {e}")
        if len(moves) != 0:
            return random.choice(moves)
        return None