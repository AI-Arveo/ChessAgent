#!/usr/bin/python3
import chess

from ChessAgent.chess_project.project.chess_agents.example_agent import ExampleAgent
from ChessAgent.chess_project.project.chess_engines.uci_engine import UciEngine
from ChessAgent.chess_project.project.chess_utilities.example_utility import ExampleUtility


if __name__ == "__main__":
    # Create your utility
    utility = ExampleUtility()
    # Create your agent
    agent = ExampleAgent(utility, 5.0)
    # Create the engine
    engine = UciEngine("Example engine", "Arne", agent)
    # Run the engine (will loop until the game is done or exited)
    engine.engine_operation()
