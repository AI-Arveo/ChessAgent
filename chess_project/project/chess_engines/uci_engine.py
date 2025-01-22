import chess
import chess.pgn
from chess_project.project.chess_agents.agent import Agent


class UciEngine():

    def __init__(self, name: str, author: str, agent: Agent) -> None:
        self.name = name
        self.author = author
        self.agent = agent
        self.board = chess.Board()  # Initialize a clean chess board
        self.options = {
            "Threads": 1,
            "Hash": 16,
            "Skill Level": 20,
        }  # Default options for UCI compatibility

    def uci_loop(self):
        """
        Main loop to handle UCI commands from Arena GUI.
        """
        while True:
            command = input().strip()  # Read input from the GUI
            if not command:
                continue

            if command == "uci":
                # UCI initialization command
                print(f"id name {self.name}")
                print(f"id author {self.author}")
                print("uciok")
            elif command == "isready":
                print("readyok") # Readiness check
            elif command == "ucinewgame":
                self.board = chess.Board()
            elif command.startswith("position"):
                self.__set_position(command)
            elif command.startswith("go"):
                self.__go()
            elif command == "quit":
                break

    def __set_option(self, input_val: str):
        """
        Set engine options from the 'setoption' command.
        """
        parts = input_val.split(" ")
        option_name = " ".join(parts[2:parts.index("value")])
        value = parts[parts.index("value") + 1]
        if option_name in self.options:
            self.options[option_name] = int(value) if value.isdigit() else value
            print(f"info string Set {option_name} to {value}")
        else:
            print(f"info string Unknown option {option_name}")

    def __set_position(self, input_val: str):
        """
        Set the board position from the 'position' command.
        """
        parts = input_val.split(" ")
        if "startpos" in parts:
            self.board = chess.Board()  # Start from the standard initial position
            if "moves" in parts:
                moves_index = parts.index("moves") + 1  # Find where moves start in the command
                for move in parts[moves_index:]:
                    self.board.push_uci(move) # Apply each move to the board
        elif "pgn" in parts:
            # Extract the PGN string
            pgn_index = parts.index("pgn") + 1
            pgn_data = " ".join(parts[pgn_index:])
            self.__load_pgn(pgn_data) # Set the board to the FEN position

    def __go(self):
        """
        Process the 'go' command and calculate the best move.

        This function starts the engine's calculation for the best move
        based on the current board position.

        :param input_val: The 'go' command received from the GUI.
        """
        # Use the agent to calculate the best move for the current board position
        best_move = self.agent.calculate_move(self.board)
        self.board.push(best_move) # Push the best move

        # Output the best move in UCI format for Arena GUI to process
        print(f"bestmove {best_move.uci()}")

