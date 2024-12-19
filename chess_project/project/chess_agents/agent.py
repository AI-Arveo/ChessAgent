import math
import chess
import chess.engine
import csv
from functools import wraps
from chess_project.project.chess_endings.syzygy_endings import initialize_tablebase
from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork
from chess_project.project.chess_utilities.utility import Utility
from chess_project.project.Logger import Logger
import torch


class Agent:
    def __init__(self, utility: Utility, time_limit_move: float, opening_files: list, tablebase_path: str):
        """
        Initialize the chess agent.

        :param utility: Utility class for board evaluation.
        :param time_limit_move: Maximum time to calculate a move.
        :param opening_files: List of file paths to opening book datasets.
        :param tablebase_path: Path to Syzygy tablebases for endgame evaluations.
        """
        self.utility = utility
        self.time_limit_move = time_limit_move
        self.opening_book = self.load_opening_book(opening_files)
        self.tablebase = initialize_tablebase(tablebase_path)
        self.nn_model = NeuralNetwork()  # Neural network for mid-game evaluation
        self.logger = Logger()

    def load_opening_book(self, file_paths):
        """
        Load the opening book from provided .tsv files.

        :param file_paths: List of .tsv files containing opening moves.
        :return: Dictionary mapping PGN positions to UCI moves.
        """
        opening_book = {}
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    opening_book[row['pgn']] = row['uci']
        return opening_book

    def calculate_move(self, board: chess.Board, depth: int):
        """
        Calculate the best move for the current board position.

        :param board: Current chess board.
        :param depth: Depth for Minimax search.
        :return: Best move in UCI format.
        """
        #First we check the opening book for the current position
        opening_move = self.select_opening_move(board)
        if opening_move:
            return opening_move

        # Step 2: Use Syzygy tablebase if in endgame
        if len(board.piece_map()) <= 5:
            best_move = self.use_tablebase(board)
            if best_move:
                return best_move

        # Step 3: Use Minimax with neural network evaluation for mid-game
        data = {"best_move": None}
        self.logger.clear()
        self.minimax(board, depth, -math.inf, math.inf, True, True, data)
        self.logger.write()
        return data["best_move"]

    def select_opening_move(self, board):
        """
        Check if the current position has a move in the opening book.

        :param board: Current chess board.
        :return: Best move in UCI format or None if not found.
        """
        pgn = board.variation_san(board.move_stack)
        return self.opening_book.get(pgn)

    def use_tablebase(self, board):
        """
        Use the Syzygy tablebase to find the best move in endgame positions.

        :param board: Current chess board.
        :return: Best move in UCI format or None if not found.
        """
        try:
            legal_moves = list(board.legal_moves)
            best_move = None
            best_wdl = -2  # Worst case WDL

            for move in legal_moves:
                board.push(move)
                try:
                    wdl = self.tablebase.probe_wdl(board)
                    if wdl > best_wdl:
                        best_wdl = wdl
                        best_move = move
                except Exception:
                    pass
                board.pop()
            return best_move
        except Exception:
            return None

    @wraps
    def minimax(self, board: chess.Board, depth: int, alpha, beta, maximizing_player: bool, save_move: bool, data: dict):
        """
        Perform a Minimax search with alpha-beta pruning.

        :param board: Current chess board.
        :param depth: Search depth.
        :param alpha: Alpha value for pruning.
        :param beta: Beta value for pruning.
        :param maximizing_player: True if maximizing player, False otherwise.
        :param save_move: Save the best move if True.
        :param data: Dictionary to store the best move.
        :return: Evaluation score of the board.
        """
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        if maximizing_player:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta, False, False, data)
                board.pop()
                if evaluation > max_eval:
                    max_eval = evaluation
                    if save_move:
                        data["best_move"] = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta, True, False, data)
                board.pop()
                if evaluation < min_eval:
                    min_eval = evaluation
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, board: chess.Board):
        """
        Evaluate the board using the neural network.

        :param board: Current chess board.
        :return: Evaluation score.
        """
        # Convert the board to a tensor representation
        board_tensor = self.board_to_tensor(board)
        with torch.no_grad():
            evaluation = self.nn_model(board_tensor.unsqueeze(0))  # Add batch dimension
        return evaluation.item()

    def board_to_tensor(self, board: chess.Board):
        """
        Convert the chess board into a tensor for the neural network.

        :param board: Current chess board.
        :return: PyTorch tensor representation of the board.
        """
        # 12 -> 1 channel for each piece type (I.E. pawn, knight, rook, ...)
        # 8x8 = size of the chessboard
        tensor = torch.zeros((12, 8, 8))
        piece_map = board.piece_map() #Returns position of every piece as a dict
        for square, piece in piece_map.items():
            channel = piece.piece_type - 1 #pawn = 1, knight = 2, ...
            if not piece.color:
                channel += 6  # Black pieces in separate channels (white = 0-5 (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING), black = 6-11 (Ofcourse same types))
            row, col = divmod(square, 8) #mark position of each piece on the board using the for loop
            tensor[channel, row, col] = 1 #Voorbeeld: White pawn on e2 = tensor[0][6][4]
        return tensor
