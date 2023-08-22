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

from sympy import symbols, Eq, lambdify
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
import numpy as np

# from bkcharts.attributes import color
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    best_expression = None
    data = None
    best_loss = np.inf

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.__operators = ['+', '-', '*', 'cos', 'x']
        self.init_legal = len(self.__operators)
        self.__operators_float = [sum(ord(i) for i in j) for j in self.__operators] #[43, 45, 42, 325, 120]
        self.legal_moves_dict = {'+' : ['cos', 'x'], '-' : ['cos', 'x'], '*' : ['cos', 'x'], 'cos' : ['cos', 'x'], 'x' : ['+', '-', '*', 'cos']}
        self.__operators_dict = {operator:name for (operator, name) in zip(self.__operators_float, self.__operators)}
        self.__operators_inv_dict = {name:operator for (operator, name) in zip(self.__operators_float, self.__operators)}
        self.__legal_moves_dict = {self.__operators_inv_dict[key]: [self.__operators_inv_dict[i] for i in value] for (key, value) in self.legal_moves_dict.items()}
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
        if 0 not in self.pieces:
            return self.__operators_float
        if not np.any(self.pieces): #All pieces are 0
            return [1, 1, 0, 1, 1]
        
        curr_move = np.where(self.pieces==0)[0][0]
        
        legal_moves = [1 if i in self.__legal_moves_dict.get(self.pieces[curr_move-1], self.__operators_float) else 0 for i in self.__operators_float]

        return legal_moves

    def has_legal_moves(self):
        return 0 in self.pieces
    
    def is_win(self):
        """Check whether the given player has created a complete (length self.n) and parseable expression
        """
        if 0 in self.pieces: #Expression list not complete
            return False
        else:
            
            expression_str = ' '.join([self.__operators_dict[i] for i in self.pieces])
            x = symbols('x')
            transformations = (standard_transformations + (implicit_multiplication_application,))
            try:
                parsed_expr = parse_expr(expression_str, transformations=transformations)
                model_selection = lambdify(x, parsed_expr)
                X, y = Board.data[:, 0], Board.data[:, 1]
                loss = np.sum((model_selection(X)-y)**2)
                if loss < Board.best_loss:
                    Board.best_expression = parsed_expr
                    Board.best_loss = loss
                    print(f"New best expression: {Board.best_expression}")
                    print(f"New best loss: {Board.best_loss:.3f}")
                    return True
                return False
            except (ValueError, SyntaxError, TypeError) as e:
#                print(e.__class__)
                expression_str += " x"
                #If adding 'x' doesn't work, we strip off tokens until an expression is valid
                while expression_str:
                    try:
                        parsed_expr = parse_expr(expression_str, transformations=transformations)
                        model_selection = lambdify(x, parsed_expr)
                        X, y = Board.data[:, 0], Board.data[:, 1]
                        loss = np.sum((model_selection(X)-y)**2)
                        if loss < Board.best_loss:
                            Board.best_expression = parsed_expr
                            Board.best_loss = loss
                            print(f"New best expression: {Board.best_expression}")
                            print(f"New best loss: {Board.best_loss:.3f}")
                            return True
                        return False
                    except:
                        expression_str = ' '.join(expression_str.split()[:-1])
                return False
          
                

    def execute_move(self, move):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """
        self.pieces[np.where(self.pieces==0)[0][0]] = move

