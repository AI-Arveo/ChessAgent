import chess
import chess.syzygy

# where to download the tableset for endings?
# https://chess.massimilianogoi.com/download/tablebases/
# download the 3-4-5 tableset (is only 1GB)
# Then unzip the tableset in the chess_project/chess_endings/3-4-5/{data}
# Make sure you have the right path


# Path to the Syzygy tablebases (update this path to the folder containing your .rtbw/.rtbz files)
tablebase_path = "D:\\User\\Mateo\\Unif\\S5\\Artificiële Intelligentie\\Chess_AI_Project\\ChessAgent\\chess_project\\project\\chess_endings\\3-4-5"

def initialize_tablebase(path):
    """
    Initializes the Syzygy tablebase.
    :param path: Path to the directory containing tablebase files.
    :return: Syzygy tablebase object.
    """
    try:
        tablebase = chess.syzygy.open_tablebase(path)
        print("Tablebase successfully loaded!")
        return tablebase
    except Exception as e:
        print(f"Error initializing tablebase: {e}")
        return None

def probe_position(board, tablebase):
    """
    Probes the Syzygy tablebase for the given board position.
    :param board: chess.Board object representing the current position.
    :param tablebase: Syzygy tablebase object.
    """
    print("\nCurrent Position:")
    print(board)

    # Probe win-draw-loss (WDL)
    try:
        wdl = tablebase.probe_wdl(board)
        result = "Win" if wdl > 0 else "Draw" if wdl == 0 else "Loss"
        print(f"Win-Draw-Loss (WDL): {result}")
    except Exception as e:
        print(f"WDL probe error: {e}")

    # Probe distance to zero (DTZ)
    try:
        dtz = tablebase.probe_dtz(board)
        print(f"Distance to Zero (DTZ): {dtz} ply")
    except Exception as e:
        print(f"DTZ probe error: {e}")

    # Find best move using the tablebase
    try:
        legal_moves = list(board.legal_moves)
        best_move = None
        best_wdl = -2  # WDL ranges from -2 (loss) to +2 (win)

        for move in legal_moves:
            board.push(move)
            try:
                wdl = tablebase.probe_wdl(board)
                if wdl > best_wdl:
                    best_wdl = wdl
                    best_move = move
            except Exception as e:
                pass  # Skip moves we can't evaluate
            board.pop()
        print(f"Best Move: {best_move} (WDL: {best_wdl})")
    except Exception as e:
        print(f"Best move probe error: {e}")

if __name__ == "__main__":
    # Initialize the Syzygy tablebase
    with initialize_tablebase(tablebase_path) as tablebase:
        if tablebase:
            # Example positions (replace with your own FENs)

            # Example 1: King and rook vs king
            fen1 = "8/8/8/8/8/8/k7/K6R w - - 0 1"
            board1 = chess.Board(fen1)
            probe_position(board1, tablebase)

            # Example 2: King and pawn vs king
            fen2 = "8/8/8/8/8/4P3/8/4K1k1 w - - 0 1"
            board2 = chess.Board(fen2)
            probe_position(board2, tablebase)

            # Example 3: King and queen vs king and rook
            fen3 = "8/8/8/8/8/8/k7/KQ6 b - - 0 1"
            board3 = chess.Board(fen3)
            probe_position(board3, tablebase)

    print("Program finished.")
