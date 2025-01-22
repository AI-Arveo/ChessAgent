#!/usr/bin/python3
import chess
from torch.cuda import utilization

from chess_project.project.chess_agents.example_agent import Agent
from chess_project.project.chess_engines.uci_engineOld import uci_loop
from chess_project.project.chess_utilities.example_utility import Utility


if __name__ == "__main__":
    # Create your utility
    utility = Utility()
    # Create your agent
    agent = Agent(utility, 5.0)
    # Create the engine
    engine = uci_loop("Example engine", "Arne", agent)
    # Run the engine (will loop until the game is done or exited)
    engine.engine_operation()
