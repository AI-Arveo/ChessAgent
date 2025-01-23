from PIL.features import features
from statistics import linear_regression
from typing import Generator
import chess.pgn
import chess.engine
import chess.polyglot
import io
import chess
import os

import torch
from numpy.ma.core import mvoid
from openpyxl.styles.builtins import total
from scipy.constants import electron_volt
from torch.utils.data import Dataset
from tornado.gen import sleep

from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork
from chess_project.project.chess_utilities.utility import isDraw

board_tensor = torch.Tensor
eval_tensor = torch.Tensor
chess_data = tuple[chess.Board, float]
chess_data_tensor = tuple[board_tensor, torch.Tensor]
chess_data_batch = list[chess_data_tensor]

class ChessDataSet(Dataset):
    def __init__(self, data_gen: Generator[chess_data,None,None]):
        super().__init__()
        self.labels: list[float] = []
        self.data: list[chess.Board] = []
        for board, value in data_gen:
            self.data.append(board)
            self.labels.append(value)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> chess_data_tensor:
        return NeuralNetwork.featureOutOfBoard(self.data[index]), torch.tensor(self.labels[index])

class Loader():
    def __init__(self, data_parser:list["DataParser"], batch_size: int = 32, heuristic = NeuralNetwork):
        self.data_parser = data_parser
        self.heuristic = heuristic
        self.data: list[chess_data_batch] = None
        self.data_size = sum([parser.size for parser in data_parser])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size

    def __iter__(self) -> Generator[chess_data_tensor, None, None]:
        if self.data:
            yield from self.data
            return
        features_device, evaluations_device = [], []
        current_batch: list[tuple[chess.Board, float]] = []
        for parser in self.data_parser:
            for chess_data in parser.values():
                current_batch.append(chess_data)
                if len(current_batch) is self.batch_size:
                    features, evaluations = self.data_to_tensor(current_batch)
                    del features_device
                    features_device = features.to(self.device)
                    del evaluations_device
                    evaluations_device = evaluations.to(self.device)
                    yield features_device, evaluations_device
                    current_batch = []
            if len(current_batch) != 0:
                features, evaluations = self.data_to_tensor(current_batch)
                del features_device
                features_device = features.to(self.device)
                del evaluations_device
                evaluations_device = evaluations.to(self.device)
                yield features_device, evaluations_device

    def __len__(self):
        return self.data_size

    def data_to_tensor(self, chess_data: list[chess_data]):
        """
        Converts the data (list) to a tensor for further use. When evaluating
        you need to adjust for when black is playing. The evaluation is relative to white
        => invert evaluation
        """
        board_features: list[board_tensor] = []
        evaluations: list[torch.Tensor] = []
        for board, evaluation in chess_data:
            board_feature = self.heuristic.featureOutOfBoard(board)
            board_features.append(board_feature)
            if board.turn == chess.BLACK:
                evaluations.append(torch.tensor(-evaluation))
                continue
            evaluations.append(torch.tensor(evaluation))
        evaluations = torch.stack(evaluations).unsqueeze(1)
        features = torch.stack(board_features)
        return features, evaluations

class DataParser():
    def __init__(self, file_path:str):
        self.size = 0
        self.data_set: ChessDataSet = None
        self.file_path = file_path
        self.cached_file = file_path +".cache"

    def parse(self, overwriteCache=False):
        if not overwriteCache and os.path.exists(self.cached_file):
            print(f'Values are available in previous data')
            with open(self.cached_file, "rb") as file:
                self.size = sum(1 for _ in file)
            return

        print(f"Reading file: {self.file_path}")
        i = 0
        reader = chess.polyglot.MemoryMappedReader("../../../chess_project/project/chess_openings/baron30.bin")
        engine = chess.engine.SimpleEngine.popen_uci("../../../stockfish/stockfish-windows-x86-64-avx2.exe")
        cache_file = open(self.cached_file, "w")
        boards: list[chess.Board] = []
        opening_drops = 0
        mate_drops = 0
        draw_drops = 0
        for game in self.read_game():
            board: chess.Board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if reader.get(board):
                    opening_drops += 1
                    continue
                elif board.is_checkmate():
                    mate_drops += 1
                elif isDraw(board):
                    draw_drops += 1
                    continue
                value = self.evalStockFish(board, engine=engine)  # Corrected method call
                if value is None:
                    continue
                fen_position = board.fen()
                cache_file.write(f"{fen_position},{value}\n")
            if (i + 1) % 100 == 0:
                print(f"Read {i + 1} games for extracting positions", end="\r")
                cache_file.flush()
            i += 1
        reader.close()
        cache_file.close()
        engine.close()
        position_count = len(boards) - opening_drops - mate_drops - draw_drops
        print(f"""
        Read in {position_count} different positions
        Omitted positions:
        - {opening_drops} were opening moves
        - {mate_drops} were mates
        - {draw_drops} were draws
              """)

    def get_data_loader(self,batch_size:int) -> Loader:
        return Loader(self,batch_size=batch_size)


    def read_game(self):
        """
        This method will read the given file-path
        """
        with open(self.file_path, encoding='utf-8') as file:
            game = chess.pgn.read_game(file)
            while game is not None:
                game = chess.pgn.read_game(file)
                if game is not None:
                    yield game

    def values(self) -> Generator[tuple[chess.Board, float],None,None]:
        with open(self.cached_file) as data:
            lines = data.readlines()
            total_lines = len(lines)
            lines_read = 0
            for line in lines:
                try:
                    lines_read += 1
                    percentage = lines_read/total_lines
                    if percentage%5 == 0:
                        print("lines read: ",percentage)
                    position_str = line.split(',')[0]
                    value = float(line.split(",")[1])
                    board = chess.Board()
                    board.set_board_fen(position_str.split(""[0]))
                    yield board, value
                except IndexError:
                    continue

    def evalStockFish(self,board: chess.Board, engine: chess.engine.SimpleEngine | None = None) -> float:
        if engine is None:
            with chess.engine.SimpleEngine.popen_uci("../../../stockfish/stockfish-windows-x86-64-avx2.exe") as fish_engine:
                info = fish_engine.analyse(board,limit=chess.engine.Limit(time=0.3,depth=5))
                score1 = info["score"].white().score(mate_score=100000)/100.0
                return score1
        info = engine.analyse(board, limit=chess.engine.Limit(time=0.1,depth=3))
        score2 = info["score"].white()
        if score2.is_mate():
            return None
        score1 = score2.score()/100.0
        return score1
