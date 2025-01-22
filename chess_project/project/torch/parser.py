import chess.pgn
import io
import chess
import os
from chess_project.project.chess_utilities.utility import isDraw
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from chess import pgn
from tqdm import tqdm

from chess_project.project.chess_engines.uci_engineOld import utility
from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork
from auxiliary_func import create_input_for_nn, encode_moves
from chess_project.project.chess_utilities.utility import Utility



class Parser():
    def open_file(self):
        with open(self.file_path, "r") as pgn_file:
            pgn = chess.pgn.read_game(pgn_file)
            print("pgn: " + str(pgn[0]))

    def parse(self,file_path , overwriteCache=False):
        self.file_path = file_path
        self.cached_file = file_path + ".cache"
        self.size = 0
        if not overwriteCache and os.path.exists(self.cached_file):
            print(f"Found previous data, values are already available")
            with open(self.cached_file, 'rb') as file:
                self.size = sum(1 for _ in file)
            return

        print(f"Reading file: {self.file_path}")
        boards: list[chess.Board] = []
        i: int = 0
        reader = chess.polyglot.MemoryMappedReader("project/Opening_Book/baron30.bin")
        engine = chess.engine.SimpleEngine.popen_uci(
            "project/chess_engines/stockfish/stockfish-windows-x86-64-avx2.exe")
        cacheFile = open(self.cached_file, 'w')
        openingDrops = 0
        mateDrops = 0
        drawDrops = 0
        for game in self.open_file():
            board: chess.Board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if reader.get(board) is not None:
                    openingDrops += 1
                    continue
                elif board.is_checkmate():
                    mateDrops += 1
                    continue
                elif isDraw(board):
                    drawDrops += 1
                    continue
                value = Parser.evaluateUsingStockFish(board, engine=engine)
                if value is None:
                    continue
                fenString = board.fen()
                cacheFile.write(f"{fenString},{value}\n")
            if (i + 1) % 100 == 0:
                print(f"Read {i + 1} games for extracting positions", end='\r')
                cacheFile.flush()
            i += 1
        reader.close()
        cacheFile.close()
        engine.close()
        positionCount = len(boards) - openingDrops - mateDrops - drawDrops
        print(f"""
            Read in {positionCount} different positions
            Omitted positions:
            - {openingDrops} were opening moves
            - {mateDrops} were mates
            - {drawDrops} were draws
                      """)


parser = Parser()
parser.parse("D:\\User\Mateo\\Unif\\S5\\Artificiële Intelligentie\\Chess_AI_Project\\ChessAgent\\LichessEliteDatabase\\lichess_elite_2014-01.pgn")