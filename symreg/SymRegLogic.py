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

from sympy import symbols, Eq, lambdify, latex
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application)
from sympy.utilities.lambdify import implemented_function
import numpy as np
from scipy.optimize import curve_fit
import math

# from bkcharts.attributes import color
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    best_expression = None
    data = None
    best_loss = np.inf

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.__operators = ['+', '-', '*', 'cos', 'x', 'const', 'grad']
        self.init_legal = len(self.__operators)
        self.__operators_float = [sum(ord(i) for i in j) for j in self.__operators] #[43, 45, 42, 325, 120]
        self.legal_moves_dict = {'+' : ['cos', 'x', 'const'], '-' : ['cos', 'x', 'const'], '*' : ['cos', 'x', 'const'], 'cos' : ['x', 'const'], 'x' : ['+', '-', '*', 'cos', 'const'], 'const': ['+', '-', '*', 'cos', 'x'], 'grad': ['x']}
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
        """Returns a list of 1's and 0's representing if the i'th operator in self.__operators is legal given the current state s (represented by the numpy array self.pieces)
        """
        if 0 not in self.pieces: #Then the state expression is complete and this function is kind of meaningless, so just return the supported operators I guess.
            print("ok, that's wierd.")
            return self.__operators_float
        if not np.any(self.pieces): #If all pieces are 0. At the beginning, all pieces are 0 (i.e. false), so the legal moves are '+', '-', 'cos', 'x', 'const', and 'grad', but not '*'
            return [1, 1, 0, 1, 1, 1, 1]
        
        curr_move = np.where(self.pieces==0)[0][0]
        
        #legal_moves stores a list of 1's and 0's representing which of the moves in self.__operators_float are legal
        #based on the last move. The last move is self.pieces[curr_move-1]
        legal_moves = [1 if i in self.__legal_moves_dict.get(self.pieces[curr_move-1]) else 0 for i in self.__operators_float]

        return legal_moves

    def is_win(self):
        """Check whether the given player has created a complete (length self.n) expression (again), and
        checks if it can be made parseable. Returns the score of the expression, where 0 <= score <= 1
        """
        if 0 in self.pieces: #Expression list not complete
            return -1
        else:
            grad = implemented_function('grad', lambda x: np.gradient(x))
            
            """
            Feel free to add more custom operators like the one above. Just make sure that you also:
               1. add the operators to the self.__operators list
               2. add the entry to the self.legal_moves_dict, which should contain the new operator name as key
                  and the list of operators that can come after it as value
            """
            
            expression_str = ' '.join([self.__operators_dict[i] for i in self.pieces])
            num_consts = expression_str.count("const")
            x = symbols('x') #TODO: Add arbitrary amount of input terms -> multivariate! (v.s. univariate)
            transformations = (standard_transformations + (implicit_multiplication_application,))
            X, Y = Board.data[:, 0], Board.data[:, 1]
            
            if num_consts:
                y = symbols(f'y(:{num_consts})')
                consts = [f"y{i}" for i in range(num_consts)]
                temp_dict = {key:value for (key,value) in zip(consts, y)}
                temp_dict.update({"grad": grad})
                for i in range(num_consts):
                    expression_str = expression_str.replace("const", f"y{i}", 1)
                
                first = False
                while expression_str:
                    try:
                        parsed_expr = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)
                        break
                    except:
                        if not first:
                            expression_str += " x"
                            first = True
                        else:
                            expression_str = ' '.join(expression_str.split()[:-1])
                
                if not expression_str:
                    return 0
                
                model_selection = lambdify((x, *y), parsed_expr)
                #try to optimize parameters y0, y1, ..., yn
                try:
                    parameters, covariance = curve_fit(model_selection, X, Y, p0 = np.ones(num_consts))
                #if it didn't work, set the parameters y0, y1, ..., yn to random values
                except RuntimeError:
                    parameters = np.random.random(num_consts)
                y_pred = model_selection(X, *parameters)
                if isinstance(y_pred, np.float64):
                    y_pred = np.full_like(Y, fill_value = y_pred)
                loss = np.sum((y_pred-Y)**2)
                assert y_pred.shape == Y.shape, f"{y_pred.shape}, {Y.shape}"
                                
                for i in range(num_consts):
                    expression_str = expression_str.replace(f"y{i}", f"{parameters[i]:0.3f}")
                
                
            else:
                first = False
                temp_dict = {"grad": grad}
                while expression_str:
                    try:
                        parsed_expr = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)
                        break
                    except:
                        if not first:
                            expression_str += " x"
                            first = True
                        else:
                            expression_str = ' '.join(expression_str.split()[:-1])
                
                if not expression_str:
                    return 0
                
                model_selection = lambdify(x, parsed_expr)
                
                y_pred = model_selection(X)
                if isinstance(y_pred, np.float64) or isinstance(y_pred, int):
                    y_pred = np.full_like(Y, fill_value = y_pred)
                loss = np.sum((y_pred-Y)**2)
                assert y_pred.shape == Y.shape, f"{y_pred.shape}, {Y.shape}"
            

            if loss < Board.best_loss:
                try:
                    Board.best_expression = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)
                except:
                    print("faulty expression_str =",expression_str)
                    try:
                        import re
                        float_pattern = r'-?\d+\.\d+'
                        floats_found = re.findall(float_pattern, expression_str)
                        new_expression = re.sub(float_pattern, r'(\g<0>)', expression_str)
                        new_expression = new_expression.replace('x','(x)')
                        Board.best_expression = parse_expr(new_expression, transformations=transformations, local_dict = temp_dict)
                        print("fixed expression_str =", Board.best_expression)
                    except:
                        print("faulty new_expression_str =", new_expression)
                        return 0

                Board.best_loss = loss
                print(f"New best expression: {Board.best_expression}")
                print(f"New best expression latex: {latex(Board.best_expression)}")
                print(f"New best loss: {Board.best_loss:.3f}")
            return math.exp(-0.005*loss)
      
                

    def execute_move(self, move):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """
        self.pieces[np.where(self.pieces==0)[0][0]] = move

