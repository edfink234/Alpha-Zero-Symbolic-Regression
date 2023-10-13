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
    def __init__(self, n=3, expression_type="prefix", visualize_exploration = True):
        self.n = n
        self.expression_type = expression_type
        if n < 1:
            log.warning(f"n = {n} is not allowed since it is less than 1, setting n to 1 now.")
            self.n = 1
        self.visualize_exploration = visualize_exploration
        self.b = Board(self.n, self.expression_type, self.visualize_exploration) #initial board

    def getInitBoard(self):
        # return initial board
#        self.b.pieces.clear()
        self.b.pieces = [self.b._Board__tokens_inv_dict[i] for i in Board.init_expression] #Ground truth Prefix: + * const cos x3 - * x0 x0 const. closest ansatz prefix = ['+', '*', 'const', 'cos']
            #Ground truth postfix: const x3 cos * x0 x0 * const - +. closest ansatz postfix = ['const', 'x3', 'cos', '*', 'x0']
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
        b = Board(self.n, self.expression_type, self.visualize_exploration)
        b.pieces = board
        if self.getGameEnded(board) != -1: #Then it ended, don't make the action
            return (b.pieces)
        move = b[action] #move is the i'th action in the set of all possible actions (self.__tokens)
        b.execute_move(move)
        return (b.pieces)

    def getValidMoves(self, board):
        # return a fixed size binary vector
        b = Board(self.n, self.expression_type, self.visualize_exploration)
        b.pieces = board
        legalMoves =  b.get_legal_moves()
        return legalMoves

    def getGameEnded(self, board):
        # return -1 if not ended, 0 <= result <= 1 (i.e., the score) if ended
        b = Board(self.n, self.expression_type, self.visualize_exploration)
        b.pieces = board
        result = b.complete_status()
        return result

    def stringRepresentation(self, board):
        # bytes representation of list (canonical board) -> Less memory :)
        return bytes(board)
