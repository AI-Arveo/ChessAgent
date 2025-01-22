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

file_path = "D:\\User\Mateo\\Unif\\S5\\Artificiële Intelligentie\\Chess_AI_Project\\ChessAgent\\LichessEliteDatabase\\lichess_elite_2014-01.pgn"
with open(file_path,"r") as pgn_file:
    pgn = chess.pgn.read_game(pgn_file)
    print("pgn: "+str(pgn[0]))

    moves_list = []
    while pgn and not pgn.is_end():
        print('pgn2: '+str(pgn))
        move = pgn.next()
        if move is None:
            continue
        else:
            moves_list.append(move)
    board = chess.Board()
    for move in moves_list:
        print('move: '+str(move))
        board.push(move)



print(moves_list)