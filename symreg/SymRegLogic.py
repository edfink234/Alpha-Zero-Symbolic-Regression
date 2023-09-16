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

def loss_func(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    return score if not np.isnan(score) and not np.isinf(score) and 0 <= score <= 1 else 0

# from bkcharts.attributes import color
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    best_expression = None
    data = None
    best_loss = np.inf
    stack = []

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.__num_features = len(Board.data[0])-1 #This assumes that the labels are just 1 column (hence the -1)
        self.__input_vars = [f'x{i}' for i in range(self.__num_features)] # (x0, x1, ..., xN)
        
        self.__unary_operators = ['cos', 'grad']
        self.__binary_operators = ['+', '-', '*']
        self.__operators = self.__unary_operators + self.__binary_operators
        
        self.__other_tokens = ["const", "stack"]
        
        self.__tokens = self.__operators + self.__input_vars + self.__other_tokens
        self.__tokens_float = list(range(1,1+len(self.__tokens)))
        
        num_operators = len(self.__operators)
        num_unary_operators = len(self.__unary_operators)
        
        self.__operators_float = self.__tokens_float[0:num_operators]
        self.__unary_operators_float = self.__operators_float[0:num_unary_operators]
        self.__binary_operators_float = self.__operators_float[num_unary_operators:]
        self.__input_vars_float = self.__tokens_float[num_operators:num_operators+self.__num_features]
        self.__other_tokens_float = self.__tokens_float[num_operators+self.__num_features:]
        
        self.action_size = len(self.__tokens)  #TODO: stacks for different types?
                
        self.__tokens_dict = {operator:name for (operator, name) in zip(self.__tokens_float, self.__tokens)} #Converts number to string
        self.__tokens_inv_dict = {name:operator for (operator, name) in zip(self.__tokens_float, self.__tokens)}

        assert(len(set(self.__tokens_dict)) == len(self.__tokens_dict))
        assert((self.__unary_operators_float+self.__binary_operators_float+self.__input_vars_float+self.__other_tokens_float) == self.__tokens_float)
        self.n = n
        # Create the empty expression list.
        self.pieces = []

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.__tokens_float[index]

    def get_legal_moves(self):
        """Returns a list of 1's and 0's representing if the i'th operator in self.__operators is legal given the current state s (represented by the list self.pieces)
        """
        
        if not self.pieces: #At the beginning, self.pieces is empty, so the only legal moves are the operators
            return [1]*len(self.__operators) + [0]*(self.__num_features) + [0, 0]
        
        elif self.__tokens_dict[self.pieces[-1]] == "grad":
            return [0]*len(self.__operators) + [1]*(self.__num_features) + [0, 0]
        
        elif self.pieces[-1] in self.__operators_float: #If the last move was an operator
            if Board.stack: #if stack's not empty you can choose it
                #return [0]*len(self.__operators) + [0]*(self.__num_features) + [0, 1]
                return [0]*len(self.__operators) + [1]*(self.__num_features) + [1, 1]
            else: #if stack's empty you can't choose it
                return [0]*len(self.__operators) + [1]*(self.__num_features) + [1, 0]
        
        elif self.pieces[-1] in self.__input_vars_float + self.__other_tokens_float:
            if self.pieces[-2] in self.__unary_operators_float: #Then the only legal moves are the operators
                return [1]*len(self.__operators) + [0]*(self.__num_features) + [0, 0]
            elif self.pieces[-2] in self.__binary_operators_float:
                if Board.stack: #if stack's not empty you can choose it
                    #return [0]*len(self.__operators) + [1]*(self.__num_features) + [1, 0]
                    return [0]*len(self.__operators) + [1]*(self.__num_features) + [1, 1]
                else: #if stack's empty you can't choose it
                    return [0]*len(self.__operators) + [1]*(self.__num_features) + [1, 0]
            else:
                return [1]*len(self.__operators) + [0]*(self.__num_features) + [0, 0]
        
        else: #TODO: Delete if never happens
            raise ValueError(f"get_legal_moves Exception! self.pieces = {self.pieces}")

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
        if len(Board.stack) < self.n: #Expression not complete
            return -1
        else:
            grad = implemented_function('grad', lambda x: np.gradient(x))
            
            expression_str = self.rpn_to_infix(Board.stack[-1])
            
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
                
                first = False
                while expression_str:
                    try:
                        parsed_expr = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)
                        break
                    except:
                        print(f"Error, {expression_str} didn't parse!")
                        exit()
                        if not first:
                            expression_str += " x0"
                            first = True
                        else:
                            expression_str = ' '.join(expression_str.split()[:-1])
                
                if not expression_str:
                    return 0
                
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
                try:
                    loss = mean_squared_error(y_pred, Y)
                    if np.isclose(loss, 0):
                        raise TypeError
                except:
                    print("Zero loss!!")
                    print("Expression =",model_selection_str)
                assert y_pred.shape == Y.shape, f"{y_pred.shape}, {Y.shape}"
                
                for i in range(num_consts):
                    expression_str = expression_str.replace(f"y{i}", f"{parameters[i]:0.3f}")
                
                
            else:
                first = False
                temp_dict.update({"grad": grad})
                while expression_str:
                    try:
                        parsed_expr = parse_expr(expression_str, transformations=transformations, local_dict = temp_dict)
                        break
                    except:
                        if not first:
                            expression_str += " x0"
                            first = True
                        else:
                            expression_str = ' '.join(expression_str.split()[:-1])
                
                if not expression_str:
                    return 0
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
            return loss_func(Y, y_pred) #1/(1+np.sqrt(loss)) #math.exp(-0.005*loss)
      
                

    def execute_move(self, move):
        """Perform the given move
        """
        #First make the move
        self.pieces.append(move)
        
        #Then figure out if we need to update the stack
        #Case 1: Unary
        if len(self.pieces) >= 2 and self.pieces[-2] in self.__unary_operators_float: #Then the move made was unary operand
            operand = self.__tokens_dict[self.pieces[-1]]
            if operand == "stack":
                operand = Board.stack[-1]
            operator = self.__tokens_dict[self.pieces[-2]]

            
            Board.stack.append(f"{operand} {operator}")
        
        #Case 2: Binary
        elif len(self.pieces) >= 3 and self.pieces[-3] in self.__binary_operators_float: #Then the move made was second binary operand
            left_operand = self.__tokens_dict[self.pieces[-1]]
            right_operand = self.__tokens_dict[self.pieces[-2]]
            operator = self.__tokens_dict[self.pieces[-3]]
            
            if right_operand == "stack":
                right_operand = Board.stack[-1]
            if left_operand == "stack":
                left_operand = Board.stack[-2]
             
            Board.stack.append(f"{left_operand} {right_operand} {operator}") 
                    
        if len(Board.stack) > self.n:
            del Board.stack[:-2] #delete all but last two expressions in stack

       
        
        
        
