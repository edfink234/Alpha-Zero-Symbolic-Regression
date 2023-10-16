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
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations)
from sympy.utilities.lambdify import implemented_function
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from visualize_tree import *
import matplotlib.pyplot as plt
from copy import copy
import cppyy

cppyy.cppdef(
r'''
std::pair<int, bool> getRPNdepth(const std::vector<std::string>& expression)
{
    if (expression.empty())
    {
        return std::make_pair(0, false);
    }

    std::unordered_set<std::string> operators = {"cos", "+", "-", "*"};
    std::unordered_set<std::string> unary_operators = {"cos"};
    std::stack<int> stack;
    bool complete = true;

    for (const std::string& token : expression)
    {
        if (unary_operators.count(token) > 0)
        {
            stack.top() += 1;
        }
        else if (operators.count(token) > 0)
        {
            int op2 = stack.top();
            stack.pop();
            int op1 = stack.top();
            stack.pop();
            stack.push(std::max(op1, op2) + 1);
        }
        else
        {
            stack.push(1);
        }
    }

    while (stack.size() > 1)
    {
        int op2 = stack.top();
        stack.pop();
        int op1 = stack.top();
        stack.pop();
        stack.push(std::max(op1, op2) + 1);
        complete = false;
    }

    return std::make_pair(stack.top() - 1, complete);
}

std::pair<int, bool> getPNdepth(const std::vector<std::string>& expression)
{
    if (expression.empty() || (expression.size() == 1 && expression[0] == " "))
    {
        return std::make_pair(0, false);
    }

    std::vector<int> stack;
    int depth = 0, num_binary = 0, num_leaves = 0;
    std::unordered_set<std::string> binary_operators = {"+", "-", "*"};
    std::unordered_set<std::string> unary_operators = {"cos"};

    for (const std::string& val : expression)
    {
        if (binary_operators.count(val) > 0)
        {
            stack.push_back(2);  // Number of operands
            num_binary++;
        }
        else if (unary_operators.count(val) > 0)
        {
            stack.push_back(1);
        }
        else
        {
            num_leaves++;
            while (!stack.empty() && stack.back() == 1)
            {
                stack.pop_back();  // Remove fulfilled operators
            }
            if (!stack.empty())
            {
                stack.back()--;  // Indicate an operand is consumed
            }
        }
        depth = std::max(depth, static_cast<int>(stack.size()) + 1);
    }
    return std::make_pair(depth - 1, num_leaves == num_binary + 1);
}

double loss_func(const std::vector<double>& actual, const std::vector<double>& predicted)
{
    if (actual.size() != predicted.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    double mse = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        double error = actual[i] - predicted[i];
        mse += error * error;
    }

    mse /= actual.size(); // Divide by the number of elements to get the mean
    return mse;
}
struct Board
{
    static std::string inline best_expression = "";
    static std::unordered_map<std::string, int> inline expression_dict = {};
    static int inline expression_dict_len = 0;
    static std::vector<std::vector<double>> inline data = {};
    static double inline best_loss = INT_MAX;
    static std::vector<std::string> inline init_expression = {};
    static double inline search_time = 0;
    
    int __num_features;
    std::vector<std::string> __input_vars;
    std::vector<std::string> __unary_operators;
    std::vector<std::string> __binary_operators;
    std::vector<std::string> __operators;
    std::vector<std::string> __other_tokens;
    std::vector<std::string> __tokens;
    std::vector<float> __tokens_float;
    
    std::vector<float> __operators_float;
    std::vector<float> __unary_operators_float;
    std::vector<float> __binary_operators_float;
    std::vector<float> __input_vars_float;
    std::vector<float> __other_tokens_float;
    
    int action_size;
                
    std::unordered_map<float, std::string> __tokens_dict; //Converts number to string
    std::unordered_map<std::string, float> __tokens_inv_dict;

    int n; //depth of RPN tree
    std::string expression_type;
    // Create the empty expression list.
    std::vector<float> pieces;
    bool visualize_exploration;
    
    Board(int n = 3, const std::string& expression_type = "prefix", bool visualize_exploration = true)
    {
        this->__num_features = Board::data[0].size() - 1;
        this->__input_vars.reserve(this->__num_features);
        for (auto i = 0; i < this->__num_features; i++)
        {
            this->__input_vars.push_back("x"+std::to_string(i));
            // std::cout << "x"+std::to_string(i) << '\n';
        }
        this->__unary_operators = {"cos"};
        this->__binary_operators = {"+", "-", "*"};
        this->__operators = {"cos", "+", "-", "*"};
        this->__other_tokens = {"const"};
        this->__tokens = {"cos", "+", "-", "*"};
        for (auto& i: this->__input_vars)
        {
            this->__tokens.push_back(i);
        }
        for (auto& i: this->__other_tokens)
        {
            this->__tokens.push_back(i);
        }
        this->action_size = this->__tokens.size();
        this->__tokens_float.reserve(this->action_size);
        for (int i = 1; i <= this->action_size; ++i)
        {
            this->__tokens_float.push_back(i);
        }
        int num_operators = this->__operators.size();
        int num_unary_operators = this->__unary_operators.size();
        for (int i = 0; i < num_operators; i++)
        {
            this->__operators_float.push_back(i);
        }
        for (int i = 0; i < num_unary_operators; i++)
        {
            this->__unary_operators_float.push_back(i);
        }
        for (int i = num_unary_operators; i < num_operators; i++)
        {
            this->__binary_operators_float.push_back(i);
        }
        int ops_plus_features = num_operators + this->__num_features;
        for (int i = num_operators; i < ops_plus_features; i++)
        {
            this->__input_vars_float.push_back(i);
        }
        for (int i = ops_plus_features; i < this->action_size; i++)
        {
            this->__other_tokens_float.push_back(i);
        }
        for (int i = 0; i < this->action_size; i++)
        {
            this->__tokens_dict[this->__tokens_float[i]] = this->__tokens[i];
            this->__tokens_inv_dict[this->__tokens[i]] = this->__tokens_float[i];
        }
        this->n = n;
        this->expression_type = expression_type;
        this->pieces = {};
        this->visualize_exploration = visualize_exploration;
    }
    
    float operator[](size_t index) const
    {
        if (index < this->__tokens_float.size())
        {
            return this->__tokens_float[index];
        }
        else
        {
            throw std::out_of_range("Index out of range");
        }
    }
    
    int __num_binary_ops()
    {
        int count = 0;
        for (float token : pieces)
        {
            if (std::find(__binary_operators_float.begin(), __binary_operators_float.end(), token) != __binary_operators_float.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_unary_ops()
    {
        int count = 0;
        for (float token : pieces)
        {
            if (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), token) != __unary_operators_float.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_leaves()
    {
        int count = 0;
        std::vector<float> leaves = __input_vars_float;
        leaves.insert(leaves.end(), __other_tokens_float.begin(), __other_tokens_float.end());

        for (float token : pieces)
        {
            if (std::find(leaves.begin(), leaves.end(), token) != leaves.end())
            {
                count++;
            }
        }
        return count;
    }
    
    std::vector<int> get_legal_moves()
    {
        std::vector<int> temp;
        temp.reserve(this->action_size);
        int binary_allowed, unary_allowed, leaves_allowed;
        if (this->expression_type == "prefix")
        {
            if (this->pieces.empty())
            {
                temp.resize(this->__operators.size(), 1);
                temp.resize(this->__operators.size() + this->__num_features, 0);
                temp.push_back(0);
                return temp;
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
            
            //__tokens_dict: converts float to string
            std::vector<std::string> string_pieces;
            string_pieces.reserve(this->pieces.size()+1);
            for (float i: this->pieces)
            {
                string_pieces.push_back(this->__tokens_dict[i]);
            }
            string_pieces.push_back(__binary_operators[0]);
            binary_allowed = (getPNdepth(string_pieces).first <= this->n) ? 1 : 0;
            string_pieces[string_pieces.size() - 1] = __unary_operators[0];
            unary_allowed = (getPNdepth(string_pieces).first <= this->n) ? 1 : 0;
            string_pieces[string_pieces.size() - 1] = __input_vars[0];
            leaves_allowed = ((num_leaves == num_binary + 1) || (getPNdepth(string_pieces).first < this->n && (num_leaves == num_binary))) ? 0 : 1;
            
        }
        else //postfix
        {
            if (this->pieces.empty())
            {
                temp.resize(this->__operators.size(), 0);
                temp.resize(this->__operators.size() + this->__num_features, 1);
                temp.push_back(1);
                return temp;
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();

            binary_allowed = (num_binary == num_leaves - 1) ? 0 : 1;
            std::vector<std::string> string_pieces;
            string_pieces.reserve(this->pieces.size()+1);
            for (float i: this->pieces)
            {
                string_pieces.push_back(this->__tokens_dict[i]);
            }
            string_pieces.push_back(__unary_operators[0]);
            unary_allowed = ((num_leaves >= 1) && (getPNdepth(string_pieces).first <= this->n)) ? 1 : 0;
            string_pieces[string_pieces.size() - 1] = __input_vars[0];

        }
        return temp;
    }
};
''')



def loss_func(y_true, y_pred):
    return 1/(1+mean_squared_error(y_true, y_pred))

# from bkcharts.attributes import color
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    best_expression = None
    expression_dict = {}
    expression_dict_len = 0
    data = None
    best_loss = np.inf
    init_expression = []
    search_time = 0

    def __init__(self, n=3, expression_type="prefix", visualize_exploration = True):
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

        self.n = n #depth of RPN tree
        self.expression_type = expression_type
        # Create the empty expression list.
        self.pieces = []
        self.visualize_exploration = visualize_exploration

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
        if self.expression_type == "prefix":
            if not self.pieces: #At the beginning, self.pieces is empty, so the only legal moves are the operators
                return [1]*len(self.__operators) + [0]*(self.__num_features) + [0]
            
            num_binary, num_leaves = self.__num_binary_ops(), self.__num_leaves()
            
            binary_allowed = 1 if getPNdepth([self.__tokens_dict[i] for i in self.pieces + [self.__binary_operators_float[-1]] ])[0] <= self.n else 0
            
            unary_allowed = 1 if getPNdepth([self.__tokens_dict[i] for i in self.pieces + [self.__unary_operators_float[-1]] ])[0] <= self.n else 0
            
            leaves_allowed = 0 if ((num_leaves == num_binary + 1) or (getPNdepth([self.__tokens_dict[i] for i in self.pieces + [self.__input_vars_float[-1]] ])[0] < self.n and num_leaves == num_binary)) else 1 #The number of leaves can never exceed number of binary + 1 in any RPN expression
        
        else: #postfix
            if not self.pieces: #At the beginning, self.pieces is empty, so the only legal moves are the features and const
                return [0]*len(self.__operators) + [1]*(self.__num_features) + [1]
            
            num_binary, num_leaves = self.__num_binary_ops(), self.__num_leaves()
            
            binary_allowed = 0 if num_binary == num_leaves - 1 else 1 #The number of binary operators can never exceed number of leaves - 1 in any RPN expression
            
            unary_allowed = 1 if num_leaves >= 1 and getRPNdepth([self.__tokens_dict[i] for i in self.pieces + [self.__unary_operators_float[-1]] ])[0] <= self.n else 0
            
            leaves_allowed = 1 if getRPNdepth([self.__tokens_dict[i] for i in self.pieces + [self.__input_vars_float[-1]] ])[0] <= self.n else 0
        return ([unary_allowed]*len(self.__unary_operators) + [binary_allowed]*len(self.__binary_operators) + [leaves_allowed]*(self.__num_features) + [leaves_allowed])
        
    def pn_to_infix(self, pn_expression):
        stack = []
        for token in pn_expression.split()[::-1]:
            if token not in self.__operators: #other
                stack.append(token)
            elif token in self.__unary_operators: #unary operator
                operand = stack.pop()
                result = f'{token}({operand})'
                stack.append(result)
            else: #binary operator
                right_operand = stack.pop()
                left_operand = stack.pop()
                result = f'({right_operand} {token} {left_operand})'
                stack.append(result)
        
        return stack[-1]
    
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
        checks if it is a complete RPN expression. Returns the score of the expression if complete, where 0 <= score <= 1
        and -1 if not complete or if the desired depth has not been reached.
        """
        if self.expression_type == "prefix":
            depth, complete = getPNdepth(expression := [self.__tokens_dict[i] for i in self.pieces])
        else: #postfix
            depth, complete = getRPNdepth(expression := [self.__tokens_dict[i] for i in self.pieces])
        if not complete or depth < self.n: #Expression not complete
            return -1
        else:
            if bytes(self.pieces) not in Board.expression_dict:
                Board.expression_dict[bytes(self.pieces)] = 1
                Board.expression_dict_len += 1
#                    print(f"Board.expression_dict_len = {Board.expression_dict_len}")
            else:
                Board.expression_dict[bytes(self.pieces)] += 1
#                return 0
            if self.visualize_exploration:
                plot_pn_expression_tree(expression) if self.expression_type == "prefix" else plot_rpn_expression_tree(expression)
#                plt.bar(Board.expression_dict.keys(), Board.expression_dict.values())
#                plt.show(block=False)
#                plt.pause(0.01)
            
            grad = implemented_function('grad', lambda x: np.gradient(x))
            
            expression_str = self.pn_to_infix(expression := ' '.join(expression)) if self.expression_type == "prefix" else self.rpn_to_infix(expression := ' '.join(expression))
            
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
                    parameters, covariance = curve_fit(model_selection, X.T, Y, p0 = np.random.random(num_consts))#, xtol = 0.5, ftol = 0.5, gtol = 0.5)
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
                print(f"New best expression ({'PN' if self.expression_type == 'prefix' else 'RPN'}): {expression}")
                print(f"New best expression (infix): {Board.best_expression}")
                print(f"New best expression latex: {latex(Board.best_expression)}")
                print(f"New best loss: {Board.best_loss:.3f}")
            return loss_func(Y, y_pred)

    def execute_move(self, move):
        """Perform the given move
        """
        #make the move
        self.pieces.append(move)
       
        
        
        
