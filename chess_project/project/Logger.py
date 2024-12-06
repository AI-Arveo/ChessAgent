from itertools import groupby
import chess

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BoardRepr:
    def __init__(self, board: chess.Board, depth: int, evaluation: float):
        self.board_str = board.unicode(invert_color=True)  # Board representation in text
        self.depth = depth
        self.evaluation = evaluation

    def __str__(self):
        return f"Depth: {self.depth}, Eval: {self.evaluation}\n{self.board_str}"

    def __repr__(self):
        return f"<BoardRepr depth={self.depth}, eval={self.evaluation}>"


class Logger(metaclass=Singleton):
    log_file = "minimax_tree.txt"

    def __init__(self):
        self.entries = []

    def append(self, board_repr: BoardRepr):
        """Add a BoardRepr object to the log."""
        for idx, entry in enumerate(self.entries):
            if entry.depth == board_repr.depth and idx < len(self.entries) - 1:
                if self.entries[idx + 1].depth < board_repr.depth:
                    self.entries = self.entries[:idx] + [board_repr] + self.entries[idx:]
                    return
        self.entries.append(board_repr)

    def clear(self):
        """Clear the log file and in-memory entries."""
        open(self.log_file, "w").close()
        self.entries.clear()

    def write(self):
        """Write the logged entries to the file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            for depth, group in groupby(self.entries, key=lambda x: x.depth):
                boards_at_depth = list(group)
                f.write(f"=== Depth {depth} ===\n")
                for board_repr in boards_at_depth:
                    f.write(f"{board_repr}\n")
                f.write("\n")
