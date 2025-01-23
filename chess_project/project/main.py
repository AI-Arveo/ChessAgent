import os
import torch
from torch import nn
import torch.cuda

from chess_project.project.chess_agents.agentNew import MiniMaxAgent
from chess_project.project.chess_engines.uci_engine import UciEngine
from chess_project.project.chess_utilities.machineLearnUtility import MachineLearningUtility
from chess_project.project.chess_utilities.utility import Utility
from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork

if __name__ == "__main__":
    # Determine the device to use for PyTorch (GPU or CPU)
    device = torch.device('cpu')

    # Resolve the absolute path to the model file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "FullPerspectiveHeuristic_1_0,0.pth")  # Ensure correct file name

    # Debugging: Print the model path and verify existence
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load the model and transfer it to the device
    network = torch.load(model_path, map_location=device)
    network = network.to(device)

    # Initialize the utility and agent
    utility = MachineLearningUtility(network)
    agent = MiniMaxAgent(utility, 5.0)

    # Initialize and start the UCI engine loop
    engine = UciEngine("ChessAgent", "MatteoArvo", agent)
    engine.uci_loop()
