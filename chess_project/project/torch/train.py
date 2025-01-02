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
X, y = create_input_for_nn(games)  # X contains board representations; y contains UCI moves
print("y before encode_moves:", y[:10])  # Debug: Check the format of y
y, move_to_int = encode_moves(y)  # Encode UCI moves to indices
num_classes = len(move_to_int)  # Number of unique moves (classes)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Create DataLoader for training
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the neural network and move it to the device
input_channels = 13  # 12 for pieces, 1 for legal moves
board_size = 8  # Chessboard is 8x8
model = NeuralNetwork(input_channels, board_size, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
        for param in model.parameters():
            param.requires_grad = True
        # Move inputs and targets to the device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)  # Should have requires_grad=True
        outputs.requires_grad = True
        print(f"outputs.requires_grad: {outputs.requires_grad}")

        # Flatten outputs to match targets
        batch_size = outputs.shape[0]
        outputs = outputs.view(batch_size, -1)

        # Compute the loss
        loss = criterion(outputs, targets)  # Should have requires_grad=True
        print('loss: '+str(loss))
        print(f"loss.requires_grad: {loss.requires_grad}")

        # Backward pass and optimization
        optimizer.zero_grad()

        # computes the gradients of the loss with respect to the model parameters
        # which are then used by the optimizer to update the parameters
        loss.backward()  # Ensure this works without error
        #loss.backward
        optimizer.step()

        running_loss += loss.item()
    #Epoch Summary
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
print("Saving model...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")