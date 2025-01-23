import torch
from torch import nn
import torch.cuda

from chess_project.project.chess_agents.agentNew import MiniMaxAgent
from chess_project.project.chess_engines.uci_engine import UciEngine
from chess_project.project.chess_utilities.machineLearnUtility import MachineLearningUtility
from chess_project.project.chess_utilities.utility import Utility
from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    network: NeuralNetwork =torch.load("../FullPerspectiveHeuristic_1_0,0.pth") #IMPORTANT, which model it needs to use
    network = network.to(device)
    utility: Utility = MachineLearningUtility(network)
    agent: MiniMaxAgent = MiniMaxAgent(utility, 5.0)
    engine: UciEngine = UciEngine("ChessAgent", "MatteoArvo", agent)
    engine.uci_loop()