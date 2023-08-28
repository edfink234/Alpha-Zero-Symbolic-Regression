from __future__ import print_function
import logging
import sys
sys.path.append('..')
from Game import Game
from .SymRegLogic import Board
import numpy as np

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""

log = logging.getLogger(__name__)

class SymRegGame(Game):
    def __init__(self, n=3):
        self.n = n
        if n < 3:
            log.warning(f"n = {n} is not allowed since it is less than 3, setting n to 3 now.")
            self.n = 3
        self.b = Board(self.n) #initial board

    def getInitBoard(self):
        # return initial board
        return self.b.pieces

    def getBoardSize(self):
        # a number
        return self.n

    def getActionSize(self):
        # return number of operators in action space
        return self.b.init_legal

    def getNextState(self, board, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.getActionSize(): 
            return board
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = b[action]
        b.execute_move(move)
        return (b.pieces)

    def getValidMoves(self, board):
        # return a fixed size binary vector
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves()
        return np.array(legalMoves)

    def getGameEnded(self, board):
        # return -1 if not ended, 0 <= result
        b = Board(self.n)
        b.pieces = np.copy(board)
        if 0 in b.pieces:
            return -1
        result = b.is_win()
        return result


    def stringRepresentation(self, board):
        # bytes representation of numpy array (canonical board)
        return board.tostring()

    #TODO: Modify this so that it works for symbolic regression instead of tictactoe?
    @staticmethod
    def display(board):
        n = board.shape[0]

        print("   ", end="")
        for y in range(n):
            print (y,"", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == -1: print("X ",end="")
                elif piece == 1: print("O ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print ("-", end="-")
        print("--")
