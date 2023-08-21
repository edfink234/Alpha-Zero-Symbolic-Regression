'''
Board class for the game of TicTacToe.
Default board size is 3x3.
Board data:
  1=white(O), -1=black(X), 0=empty
  first dim is column , 2nd is row:
     pieces[0][0] is the top left square,
     pieces[2][0] is the bottom left square,
Squares are stored and manipulated as (x,y) tuples.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the board for the game of Othello by Eric P. Nichols.

'''

from sympy import symbols, Eq
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
import numpy as np

# from bkcharts.attributes import color
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.__operators = ['+', '-', '*', 'cos', 'x']
        self.__operators_float = [sum(ord(i) for i in j) for j in self.__operators]
        self.__operators_dict = {operator:name for (operator, name) in zip(self.__operators_float, self.__operators)}
        assert(len(set(self.__operators_float)) == len(self.__operators_float))
        self.n = n
        # Create the empty expression list.
        self.pieces = np.array([0]*self.n)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.__operators_float[index]

    def get_legal_moves(self):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.
        curr_move = self.n if 0 not in self.pieces else np.where(self.pieces==0)[0][0]
        
        #TODO: Have a way of checking which operators are legal based on current expression list (self.pieces)
        return self.__operators_float

    def has_legal_moves(self):
        return 0 in self.pieces
    
    def is_win(self):
        """Check whether the given player has created a complete (length self.n) and parseable expression
        """
        if 0 in self.pieces: #Expression list not complete
            return False
        else:
            
            expression_str = ' '.join([self.__operators_dict[i] for i in self.pieces])
            print(expression_str)
            x = symbols('x')
            transformations = (standard_transformations + (implicit_multiplication_application,))
            try:
                parsed_expr = parse_expr(expression_str, transformations=transformations)
                print("parsed_expr =",parsed_expr)
                return True
            except ValueError:
#                print("ValueError")
                return False
            except SyntaxError:
#                print("SyntaxError")
                return False
            except TypeError:
#                print("TypeError")
                return False
                

    def execute_move(self, move):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """
        self.pieces[np.where(self.pieces==0)[0][0]] = move

