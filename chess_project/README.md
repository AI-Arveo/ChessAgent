# Chess Engine Project

Welcome to the **Chess Engine Project**! This project implements a chess engine powered by a combination of advanced algorithms, including a MiniMax-based search with alpha-beta pruning, and machine learning models to evaluate board positions. The engine is designed to play chess efficiently and effectively.

---

## Features

- **MiniMax Algorithm**: Evaluates possible moves and finds the optimal strategy for both players.
- **Alpha-Beta Pruning**: Optimizes the MiniMax search by eliminating unnecessary branches, allowing deeper searches within the same time frame.
- **Iterative Deepening**: Gradually increases search depth for improved move quality.
- **Move Sorting**: Prioritizes promising moves to make pruning more effective.
- **Opening Book**: Uses precomputed opening moves for faster and stronger play in the opening phase.
- **Endgame Tablebases**: Fetches optimal moves for positions with 5 or fewer pieces.
- **Machine Learning Utility**: Integrates a neural network model for accurate board evaluation.

---

## Requirements

Before running the chess engine, make sure you have the following installed:

- **Python 3.8+**
- **Dependencies**: Install the required libraries using:
  ```bash
  pip install -r requirements.txt

- **Torch Model**  
   Place the neural network model file (`FullPerspectiveHeuristic_1_0,0.pth`) in the root directory of the project. This file is essential for the engine's evaluation capabilities, as it leverages a trained neural network to assess board positions accurately.

---

## Running the Engine

The chess engine is designed to work with UCI (Universal Chess Interface)-compatible GUIs like [Arena](http://www.playwitharena.de/). Follow these steps to run the engine:

1. **Locate the Engine**  
   The main executable file for this project is `run_engine.bat`. This script sets up and launches the chess engine in a UCI-compatible format.

2. **Add the Engine to Arena**  
   - Open Arena.
   - Navigate to **Engines** → **Manage** → **Add**.
   - Select `run_engine.bat` as the engine executable.

3. **Configure the Engine**  
   - Ensure Arena is set to use the UCI protocol.
   - Adjust engine options, such as time controls or threads, as needed.

4. **Start Playing**  
   Once the engine is added, you can either play against it or let it analyze games automatically.

---

## Project Structure

- **`chess_project/`**  
  Contains the core files for the engine, including agents, utilities, and the neural network logic.

- **`run_engine.bat`**  
  A batch file that initializes and runs the chess engine for UCI compatibility.

- **`FullPerspectiveHeuristic_1_0,0.pth`**  
  The pre-trained neural network model for evaluating chess board positions.

- **`requirements.txt`**  
  A list of Python dependencies required for the project. Install them using:
  ```bash
  pip install -r requirements.txt

---

## How It Works

1. **Input from GUI**  
   The engine communicates with UCI-compatible GUIs (like Arena) by receiving commands such as:
   - `position`: Sets up the board position, either from the starting position or with specific moves.
   - `go`: Instructs the engine to calculate the best move.
   - `isready`: Checks if the engine is ready to start.
   - `quit`: Terminates the engine.

2. **Decision-Making**  
   Depending on the phase of the game, the engine uses different strategies to find the best move:
   - **Opening Phase**: The engine consults its opening book (`Read_Book`) to play precomputed, strong moves.
   - **Endgame Phase**: If 5 or fewer pieces are left, it uses endgame tablebases to find perfect moves.
   - **Middle Game or Other Scenarios**: The engine performs iterative deepening with the MiniMax algorithm, optimized by alpha-beta pruning, to evaluate moves and search deeper for better options.

3. **Move Evaluation**  
   The engine integrates a neural network model to evaluate board positions. This allows for more sophisticated decision-making beyond traditional evaluation heuristics.

4. **Output to GUI**  
   After calculating the best move, the engine sends it back to the GUI in UCI format, such as: