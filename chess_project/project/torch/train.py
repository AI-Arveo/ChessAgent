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

# Configuration
PGN_DIR = "../../../LichessEliteDatabase"  # Path to the PGN files
MODEL_SAVE_PATH = "./chess_model.pth"  # Path to save the trained model
BATCH_SIZE = 64  # Batch size for training
EPOCHS = 10  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate
LIMIT_OF_FILES = 1  # Limit the number of PGN files to process

# Function to load PGN files
def load_pgn(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

# Load games from PGN files
print("Loading games from PGN files...")
files = [file for file in os.listdir(PGN_DIR) if file.endswith(".pgn")]
games = []
for i, file in enumerate(tqdm(files[:LIMIT_OF_FILES], desc="Processing PGN files")):
    games.extend(load_pgn(os.path.join(PGN_DIR, file)))
print(f"GAMES PARSED: {len(games)}")

# Preprocess data
print("Preprocessing data...")
X, y = create_input_for_nn(games)
X, y = X[:2500000], y[:2500000]  # Limit the number of samples for training
y, move_to_int = encode_moves(y)
num_classes = len(move_to_int)  # Number of unique moves (classes)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Create DataLoader for training
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the neural network
print("Initializing neural network...")
# Update input channels to match X
input_channels = X.shape[1]  # Automatically use the first dimension of X for channels
board_size = 8  # Chessboard is 8x8
model = NeuralNetwork(input_channels, board_size, num_classes)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
        # Forward pass
        outputs = model(inputs)
        #print("output: "+str(outputs))
        print("targets: "+str(targets))
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print epoch summary
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
print("Saving model...")
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
