#!/usr/bin/python3
import chess.pgn
import os

""" A game from a pgn file is played out """
def pgn_example():
    # Open the file containing the game information
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../data/kasparov-deep-blue-1997.pgn')
    pgn = open(filename)
    running = True

    while running:
        # Read in the game 
        pgn_game = chess.pgn.read_game(pgn)
        
        # Check if the game file is correct
        if pgn_game:
            board = pgn_game.board()

            # Play the moves specified in the file
            for move in pgn_game.mainline_moves():
                print(board)
                print("----------------------------------------")
                board.push(move)
        else:
            running = False


if __name__ == "__main__":
    pgn_example()
