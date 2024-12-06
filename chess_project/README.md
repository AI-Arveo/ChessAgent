# 5AI Chess Project
A basic framework built around python-chess. 

The framework contains a basic class structure for utilities, agents and engines (can be used in combination with a GUI).

## Agents
Agents contain the logic to search trough the game tree (this is where you would place a **minimax**). 
Agents use a Utility to give values to positions, these values are then used to select the best action according to your search tree.

## Utilities
Utilities contain the logic to give a value to a state of the game (heuristics). This value determines how advantageous a position is for either side.

## Engines
The uci_engine can be used to interact with Graphical programs, this way you can play against your own agent.
Remember that these programs usually require a .bat or .exe file to work. An example of such a .bat file is provided.

## Examples
This folder contains three example scripts to show you how you can use (or train) your agent.
The scripts show how to play against yourself, how to play against stockfish and how to read from pgn files.
To play against stockfish, remember to install it from https://stockfishchess.org/ and reference the path where you installed it correctly.
