import chess.pgn
import io
import chess
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from chess import pgn
from tqdm import tqdm
from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork
from auxiliary_func import create_input_for_nn, encode_moves

pgn_directory = r"D:\PythonProjects\ChessAgent\LichessEliteDatabase"
def process_pgn_files(directory):
    processed_files = 0  #number of processed files

    for filename in os.listdir(directory):
        if filename.startswith("lichess_elite_") and filename.endswith(".pgn"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")

            with open(file_path, "r") as pgn_file:
                #for loop to process each game
                for game_count, game in enumerate(iter(lambda: chess.pgn.read_game(pgn_file), None), start=1):
                    moves_list = []
                    node = game

                    # Collect moves
                    while node.next():  # Traverse the game tree
                        move = node.next().move  # Extract the Move object
                        moves_list.append(move)
                        node = node.next()

                    print(f"  Game {game_count}: {moves_list}")

            print(f"Finished processing file: {filename}.\n")

            processed_files += 1
            if processed_files == 10:  # Stop after processing 10 files
                break

process_pgn_files(pgn_directory)
