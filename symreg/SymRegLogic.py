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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import curve_fit
import math
from visualize_tree import *

def loss_func(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    return score if not np.isnan(score) and not np.isinf(score) and 0 <= score <= 1 else 0

# from bkcharts.attributes import color
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    best_expression = None
    data = None
    best_loss = np.inf

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.__num_features = len(Board.data[0])-1 #This assumes that the labels are just 1 column (hence the -1)
        self.__input_vars = [f'x{i}' for i in range(self.__num_features)] # (x0, x1, ..., xN)
        
        self.__unary_operators = ['cos']
        self.__binary_operators = ['+', '-', '*']
        self.__operators = self.__unary_operators + self.__binary_operators
        
        self.__other_tokens = ["const"]
        
        self.__tokens = self.__operators + self.__input_vars + self.__other_tokens
        self.__tokens_float = list(range(1,1+len(self.__tokens)))
        
        num_operators = len(self.__operators)
        num_unary_operators = len(self.__unary_operators)
        
        self.__operators_float = self.__tokens_float[0:num_operators]
        self.__unary_operators_float = self.__operators_float[0:num_unary_operators]
        self.__binary_operators_float = self.__operators_float[num_unary_operators:]
        self.__input_vars_float = self.__tokens_float[num_operators:num_operators+self.__num_features]
        self.__other_tokens_float = self.__tokens_float[num_operators+self.__num_features:]
        
        self.action_size = len(self.__tokens)
                
        self.__tokens_dict = {operator:name for (operator, name) in zip(self.__tokens_float, self.__tokens)} #Converts number to string
        self.__tokens_inv_dict = {name:operator for (operator, name) in zip(self.__tokens_float, self.__tokens)}

        assert(len(set(self.__tokens_dict)) == len(self.__tokens_dict))
        assert((self.__unary_operators_float+self.__binary_operators_float+self.__input_vars_float+self.__other_tokens_float) == self.__tokens_float)
        self.n = n #depth of RPN tree
        # Create the empty expression list.
        self.pieces = []

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.__tokens_float[index]
    
    def __num_binary_ops(self):
        count = 0
        for i in self.pieces:
            if i in self.__binary_operators_float:
                count += 1
        return count
    
    def __num_unary_ops(self):
        count = 0
        for i in self.pieces:
            if i in self.__unary_operators_float:
                count += 1
        return count
    
    def __num_leaves(self):
        count = 0
        leaves = self.__input_vars_float + self.__other_tokens_float
        for i in self.pieces:
            if i in leaves:
                count += 1
        return count
    
    def get_legal_moves(self):
        """Returns a list of 1's and 0's representing if the i'th operator in self.__operators is legal given the current state s (represented by the list self.pieces)
        """
        if not self.pieces: #At the beginning, self.pieces is empty, so the only legal moves are the features and const
            return [0]*len(self.__operators) + [1]*(self.__num_features) + [1]
        
        num_binary, num_leaves = self.__num_binary_ops(), self.__num_leaves()
        
        binary_allowed = 0 if num_binary == num_leaves - 1 else 1 #The number of binary operators can never exceed number of leaves - 1 in any RPN expression
        
        unary_allowed = 1 if num_leaves >= 1 and getRPNdepth([self.__tokens_dict[i] for i in self.pieces + [self.__unary_operators_float[-1]] ])[0] <= self.n else 0
        
        leaves_allowed = 1 if getRPNdepth([self.__tokens_dict[i] for i in self.pieces + [self.__input_vars_float[-1]] ])[0] <= self.n else 0
        
        return ([unary_allowed]*len(self.__unary_operators) + [binary_allowed]*len(self.__binary_operators) + [leaves_allowed]*(self.__num_features) + [leaves_allowed])
        
    def rpn_to_infix(self, rpn_expression):
        stack = []
        for token in rpn_expression.split():
            if token not in self.__operators: #other
                stack.append(token)
            elif token in self.__unary_operators: #unary operator
                operand = stack.pop()
                result = f'{token}({operand})'
                stack.append(result)
            else: #binary operator
                right_operand = stack.pop()
                left_operand = stack.pop()
                result = f'({left_operand} {token} {right_operand})'
                stack.append(result)
        
        return stack[-1]

    def complete_status(self):
        """Check whether the given player has created a complete (depth self.n) expression (again), and
        checks if it can be made parseable. Returns the score of the expression, where 0 <= score <= 1
        """
        depth, complete = getRPNdepth(expression := [self.__tokens_dict[i] for i in self.pieces])
        if not complete or depth < self.n: #Expression not complete
            return -1
        else:
            grad = implemented_function('grad', lambda x: np.gradient(x))
            
            expression_str = self.rpn_to_infix(expression := ' '.join(expression))
            print(expression_str)
            plot_rpn_expression_tree(expression, block=False)
            
            num_consts = expression_str.count("const")
            x = symbols(f'x(:{self.__num_features})')
            input_vars = [f"x{i}" for i in range(self.__num_features)]
            temp_dict = {key:value for (key,value) in zip(input_vars, x)} #So sympy knows that x0, x1, ..., xN are input variables
            transformations = standard_transformations
            X, Y = Board.data[:, :self.__num_features], Board.data[:, -1]
            
            if num_consts: #If there are "consts" in the expression that need to be determined/optimized
                y = symbols(f'y(:{num_consts})')
                consts = [f"y{i}" for i in range(num_consts)]
                temp_dict.update({key:value for (key,value) in zip(consts, y)})
                temp_dict.update({"grad": grad})
                for i in range(num_consts):
                    expression_str = expression_str.replace("const", f"y{i}", 1)
                
                parsed_expr = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)
                model_selection_str = str(parsed_expr)
                for i in range(self.__num_features):
                    model_selection_str = model_selection_str.replace(f"x{i}", f"x[{i}]")
                func_str = f"""
def model_selection(x, {', '.join(consts)}):
    from numpy import cos, gradient as grad
    return {model_selection_str}
                """
                
                exec(func_str, globals(), lcls:=locals())
                model_selection = lcls["model_selection"]
                
                #try to optimize parameters y0, y1, ..., yn
                try:
                    parameters, covariance = curve_fit(model_selection, X.T, Y, p0 = np.random.random(num_consts))
                #if it didn't work, set the parameters y0, y1, ..., yn to random values
                except RuntimeError:
                    parameters = np.random.random(num_consts)
                y_pred = model_selection(X.T, *parameters)
                
                if isinstance(y_pred, np.float64) or isinstance(y_pred, int):
                    y_pred = np.full_like(Y, fill_value = y_pred)
                loss = mean_squared_error(y_pred, Y)
                assert y_pred.shape == Y.shape, f"{y_pred.shape}, {Y.shape}"
                
                for i in range(num_consts):
                    expression_str = expression_str.replace(f"y{i}", f"{parameters[i]:0.3f}")
                
                
            else:
                first = False
                temp_dict.update({"grad": grad})
                parsed_expr = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)
                model_selection = lambdify(x, parsed_expr)
                
                y_pred = model_selection(*[X[:,i] for i in range(self.__num_features)])
                if isinstance(y_pred, np.float64) or isinstance(y_pred, int):
                    y_pred = np.full_like(Y, fill_value = y_pred)
                loss = mean_squared_error(y_pred, Y)

            for i in range(self.__num_features):
                expression_str = expression_str.replace(f"x[{i}]", f"x{i}")
            if loss < Board.best_loss:
                Board.best_expression = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)

                Board.best_loss = loss
                print(f"New best expression: {Board.best_expression}")
                print(f"New best expression latex: {latex(Board.best_expression)}")
                print(f"New best loss: {Board.best_loss:.3f}")
            
            self.pieces.clear()
            return loss_func(Y, y_pred) #1/(1+np.sqrt(loss)) #math.exp(-0.005*loss)
      
                

    def execute_move(self, move):
        """Perform the given move
        """
        #make the move
        self.pieces.append(move)
       
        
        
        
