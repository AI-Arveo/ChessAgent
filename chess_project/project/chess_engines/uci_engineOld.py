import sys
import chess
import torch
from chess_project.project.chess_utilities.utility import Utility
from chess_project.project.torch.auxiliary_func import decode_move_with_probabilities, board_to_tensor
from chess_project.project.chess_neuralNetwork.neural_network import NeuralNetwork

# Load the trained model
MODEL_PATH = "D:\\User\\Mateo\\Unif\\S5\\Artificiële Intelligentie\\Chess_AI_Project\\ChessAgent\\chess_project\\chess_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Utility with the custom neural network model
try:
    utility = Utility(model_path=MODEL_PATH, device=DEVICE)
except Exception as e:
    print(f"Error initializing Utility: {e}")
    sys.exit(1)

board = chess.Board()  # Initialize a new chess board


def uci_loop():
    """
    Main loop to handle UCI commands from Arena GUI.
    """
    while True:
        try:
            command = input().strip()  # Read a command from Arena GUI
        except EOFError:
            break

        if command == "uci":
            # UCI initialization command
            print("id name MyChessAgent")
            print("id author Your Name")
            print("uciok")
        elif command == "isready":
            # Readiness check
            print("readyok")
        elif command.startswith("position"):
            # Update board position
            try:
                if "moves" in command:
                    parts = command.split(" moves")
                    fen = parts[0].replace("position fen ", "").replace("position startpos", chess.STARTING_FEN).strip()
                    moves = parts[1].strip().split()
                    board.set_fen(fen)
                    for move in moves:
                        board.push_uci(move)
                else:
                    board.set_fen(chess.STARTING_FEN)
                print(f"Board updated to FEN: {board.fen()}")  # Debugging log
            except Exception as e:
                print(f"Error updating board position: {e}")
        elif command.startswith("go"):
            # Calculate the best move
            print("Calculating best move...")
            try:
                # Convert board to tensor and evaluate with the neural network
                board_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = utility.model(board_tensor)  # Pass through the neural network
                best_move, probability = decode_move_with_probabilities(output, board)
                print(f"bestmove {best_move}")  # Respond with the best move in UCI format
            except Exception as e:
                # Fallback: Play a random legal move if the neural network fails
                print(f"Error during move calculation: {e}")
                try:
                    fallback_move = list(board.legal_moves)[0]
                    print(f"fallback_move {fallback_move}")
                except IndexError:
                    print("No legal moves available. Stalemate or checkmate.")
        elif command == "quit":
            # Quit the engine
            utility.close()
            break
        else:
            # Ignore unknown commands for compatibility
            print(f"info string Unknown command: {command}")


if __name__ == "__main__":
    try:
        uci_loop()
    except Exception as main_error:
        print(f"Unexpected error in UCI loop: {main_error}")
