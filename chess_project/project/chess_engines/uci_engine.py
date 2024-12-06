import chess
from project.chess_agents.agent import Agent

class UciEngine():
    
    def __init__(self, name: str, author: str, agent: Agent) -> None:
        self.name = name
        self.author = author
        self.agent = agent
        
        
    def engine_operation(self):
        # Create a clean chess board
        board = chess.Board()

        # Continuously receive commands and process them
        while True:
            input_val = input().split(' ')

            if len(input_val) > 2:
                if input_val[0] == "position" and \
                        input_val[1] == "startpos" and \
                        input_val[2] == "moves":
                    board = chess.Board()
                    for move in input_val[3::]:
                        board.push_uci(move)
                elif input_val[0] == "go":
                    print("bestmove {}".format(self.agent.calculate_move(board)))

            elif len(input_val) > 1:
                if input_val[0] == "position" and \
                        input_val[1] == "startpos":
                    board = chess.Board()
                    for move in input_val[3::]:
                        board.push_uci(move)

            elif len(input_val) > 0:
                if input_val[0] == "uci":
                    self.__uci()

                elif input_val[0] == "quit":
                    break

                elif input_val[0] == "ucinewgame":
                    board = chess.Board()

                elif input_val[0] == "isready":
                    print("readyok")
                    
                    
    def __uci(self):
        print("""id name {}
id author {}
option name Debug Log File type string default
option name Contempt type spin default 0 min -100 max 100
option name Threads type spin default 1 min 1 max 128
option name Hash type spin default 16 min 1 max 1048576
option name Clear Hash type button
option name Ponder type check default false
option name MultiPV type spin default 1 min 1 max 500
option name Skill Level type spin default 20 min 0 max 20
option name Move Overhead type spin default 30 min 0 max 5000
option name Minimum Thinking Time type spin default 20 min 0 max 5000
option name Slow Mover type spin default 89 min 10 max 1000
option name nodestime type spin default 0 min 0 max 10000
option name UCI_Chess960 type check default false
option name SyzygyPath type string default <empty>
option name SyzygyProbeDepth type spin default 1 min 1 max 100
option name Syzygy50MoveRule type check default true
option name SyzygyProbeLimit type spin default 6 min 0 max 6
uciok""".format(self.name, self.author))