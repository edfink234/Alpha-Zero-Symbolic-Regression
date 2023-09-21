import logging
import sys
sys.path.append('..')
from Game import Game
from .SymRegLogic import Board
import numpy as np

"""
Game class implementation for the game of Symbolic Regression
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""

log = logging.getLogger(__name__)

class SymRegGame(Game):
    def __init__(self, n=3):
        self.n = n
        if n < 1:
            log.warning(f"n = {n} is not allowed since it is less than 1, setting n to 1 now.")
            self.n = 1
        self.b = Board(self.n) #initial board

    def getInitBoard(self):
        # return initial board
        self.b.pieces.clear()
        return self.b.pieces
        
    def getBoardSize(self):
        # the maximum number of elements in a tree of depth N
        return 2*(2**self.n)-1

    def getActionSize(self):
        # return number of operators in action space
        return self.b.action_size

    def getNextState(self, board, action):
        # if player takes action on board, return next (board,player)
        # action is an index i from 0 to getActionSize-1
#        print("hi")
        b = Board(self.n)
        b.pieces = board
        if self.getGameEnded(board) != -1: #Then it ended, don't make the action
            return (b.pieces)
        move = b[action] #move is the i'th action in the set of all possible actions (self.__tokens)
        b.execute_move(move)
        return (b.pieces)

    def getValidMoves(self, board):
        # return a fixed size binary vector
        b = Board(self.n)
        b.pieces = board
        legalMoves =  b.get_legal_moves()
        return legalMoves

    def getGameEnded(self, board):
        # return -1 if not ended, 0 <= result <= 1 (i.e., the score) if ended
        b = Board(self.n)
        b.pieces = board
        result = b.complete_status()
        return result


    def stringRepresentation(self, board):
        # bytes representation of numpy array (canonical board)
        return bytes(board)
