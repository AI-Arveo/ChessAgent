import chess
import random
import csv
import os


class ReadOpening:
    def __init__(self):
        """
        Initialize the opening book by reading all .tsv files from a fixed directory.
        """
        # Fixed directory for .tsv files
        self.tsv_directory = r"D:\PythonProjects\ChessAgent\chess_project\project\chess_openings"
        self.opening_book = self.load_opening_moves()

    def load_opening_moves(self):
        """
        Load opening moves from all .tsv files in the fixed directory.
        :return: Dictionary mapping board positions (PGN or FEN) to lists of moves.
        """
        opening_book = {}
        for filename in os.listdir(self.tsv_directory):
            if filename.endswith(".tsv"):
                file_path = os.path.join(self.tsv_directory, filename)
                print(f"Parsing file: {filename}")
                try:
                    with open(file_path, 'r') as tsv_file:
                        reader = csv.DictReader(tsv_file, delimiter='\t')
                        for row in reader:
                            position = row['pgn']  # Adjust key if necessary
                            move = row['uci']      # Adjust key if necessary
                            if position not in opening_book:
                                opening_book[position] = []
                            opening_book[position].append(move)
                except Exception as e:
                    print(f"Error parsing file {filename}: {e}")
        return opening_book

    def Opening(self, board) -> chess.Move | None:
        """
        Finds a specific position in the opening book. If it exists, yield a random next move.
        Otherwise, return None.
        :param board: Current chess board.
        :return: Random legal move from the opening book or None.
        """
        pgn = board.variation_san(board.move_stack)
        if pgn in self.opening_book:
            uci_moves = self.opening_book[pgn]
            uci_move = random.choice(uci_moves)
            return chess.Move.from_uci(uci_move)
        return None


# Example Usage
if __name__ == "__main__":
    read_book = ReadOpening()

    board = chess.Board()
    move = read_book.Opening(board)

    if move:
        print(f"Recommended opening move: {move}")
    else:
        print("No opening move found for this position.")
