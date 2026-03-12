#A script for visualizing prefix and postfix expression trees
import matplotlib.pyplot as plt
import pydot
from collections import deque
from numpy.random import choice
from time import time
from copy import deepcopy
from matplotlib.animation import FuncAnimation
import os
import dot2tex

class Node:
    def __init__(self, value, unique_id):
        self.value = value
        self.unique_id = unique_id
        
class BinaryNode(Node):
    def __init__(self, value, unique_id):
        super().__init__(value, unique_id)
        self.left = None
        self.right = None

class UnaryNode(Node):
    def __init__(self, value, unique_id):
        super().__init__(value, unique_id)
        self.child = None
        
def is_operator(token):
    return is_binary_operator(token) or is_unary_operator(token)
def is_binary_operator(token):
    return token in {'+', '-', '*', '/', '^', 'MYCDOT', 'MYPOW'}
def is_unary_operator(token):
    return token in {"cos", "exp", "sqrt", "sin", "asin", "arcsin", "log", "tanh", "acos", "arccos", "~", "ln", "MYBRACKETSQRT", "tan", "MYCOS", "MYSIN", "MYTAN", "sech"}
def is_operand(token):
    return (not is_operator(token) and token not in {'(', ')'})

def prec(c):
    if is_unary_operator(c):
        return 4
    if c == '^':
        return 3
    elif c == '/' or c == '*':
        return 2
    elif c == '+' or c == '-':
        return 1
    else:
        return -1

#^ because x^y^z = x^(y^z) #right sub-expression first
#unary because unary unary x = unary(unary(x)) #right sub-expression first
#- is left because x-y-z = (x-y)-z
#/ is left because x/y/z = (x/y)/z
def isRightAssociative(c):
    return (c == '^' or is_unary_operator(c))
    
def tokenize_infix(expr: str):
    """
    Tokenize an infix expression into operators, parentheses, identifiers, and numbers.

    Adds support for '~' as unary minus operator (always tokenized as '~').

    Identifiers: letters/digits/underscore (e.g. x0, var_1, cos)
    Numbers:     12, 12.3, .5, 1e-3, -2.7E+4  (leading sign only for + or -)
    Operators:   + - * / ^ ~
    Parens:      ( )
    """
    tokens = []
    i = 0
    n = len(expr)

    def is_ident_char(ch):
        return ch.isalnum() or ch == '_'

    def scan_number(start):
        """Scan unsigned number starting at index start (digit or '.') with optional exponent."""
        j = start
        saw_digit = False
        saw_dot = False

        # mantissa
        while j < n:
            ch = expr[j]
            if ch.isdigit():
                saw_digit = True
                j += 1
            elif ch == '.' and not saw_dot:
                saw_dot = True
                j += 1
            else:
                break

        if not saw_digit:
            # e.g. "." not followed by digits is not a number
            return None

        # exponent
        if j < n and expr[j] in "eE":
            k = j + 1
            if k < n and expr[k] in "+-":
                k += 1
            exp_start = k
            while k < n and expr[k].isdigit():
                k += 1
            if k > exp_start:  # had at least one exponent digit
                j = k

        return j  # end index (exclusive)

    while i < n:
        ch = expr[i]

        # skip whitespace
        if ch.isspace():
            i += 1
            continue

        # parentheses
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue

        # operators (include '~')
        if ch in "^*/~":
            tokens.append(ch)
            i += 1
            continue

        # + or - can be operator OR can start a signed number
        if ch in "+-":
            # signed number if next char begins a number
            if i + 1 < n and (expr[i + 1].isdigit() or expr[i + 1] == "."):
                end = scan_number(i + 1)
                if end is not None:
                    tokens.append(expr[i:end])  # include the sign
                    i = end
                    continue
            # otherwise treat as operator
            tokens.append(ch)
            i += 1
            continue

        # number starting with digit or '.'
        if ch.isdigit() or ch == ".":
            end = scan_number(i)
            if end is None:
                raise ValueError(f"Invalid number starting at position {i}")
            tokens.append(expr[i:end])
            i = end
            continue

        # identifier / function name
        if is_ident_char(ch):
            j = i + 1
            while j < n and is_ident_char(expr[j]):
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue

        raise ValueError(f"Unexpected character {ch!r} at position {i}")

    return tokens

def infix_to_rpn(infix_expression):
    st = []
    res = []
#    print(f"infix_expression = {infix_expression}")
    if isinstance(infix_expression, str):
        infix_expression = tokenize_infix(infix_expression)
#    print(f"infix_expression = {infix_expression}")
    
#    exit()
    prev_token = None

    for c in infix_expression:

        # Turn unary "-" into "~" (SymPy prints "-sin(...)", "-(...)", etc.)
        if c == '-' and (
            prev_token is None or
            prev_token == '(' or
            is_operator(prev_token)
        ):
            c = '~'
            
        # If operand, add to result
        if is_operand(c):
            res.append(c)

        # If '(', push to stack
        elif c == '(':
            st.append('(')

        # If ')', pop until '('
        elif c == ')':
            while st and st[-1] != '(':
                res.append(st.pop())
            st.pop()

        # If operator
        else:
            while st and st[-1] != '(' and \
                (prec(st[-1]) > prec(c) or (prec(st[-1]) == prec(c) \
                                    and not isRightAssociative(c))):
                res.append(st.pop())
            st.append(c)
        prev_token = c
            
#    print(f"res = {res}")
#    print(f"st = {st}")
    while st:
        res.append(st.pop())

    return ' '.join(res)

def rpn_to_infix(rpn_expression):
    stack = []
    if isinstance(rpn_expression, str):
        rpn_expression = rpn_expression.split()
    for token in rpn_expression:
        if not is_operator(token): #other
            stack.append(token)
        elif is_unary_operator(token): #unary operator
            operand = stack.pop()
            result = f'{token}({operand})'
            stack.append(result)
        else: #binary operator
            right_operand = stack.pop()
            left_operand = stack.pop()
            result = f'({left_operand} {token} {right_operand})'
            stack.append(result)
    
    return stack[-1]
    
def rpn_to_pre(rpn_expression):
    stack = []
    if isinstance(rpn_expression, str):
        rpn_expression = rpn_expression.split()
    for token in rpn_expression:
        if not is_operator(token): #other
            stack.append(token)
        elif is_unary_operator(token): #unary operator
            operand = stack.pop()
            result = f'{token} {operand}'
            stack.append(result)
        else: #binary operator
            right_operand = stack.pop()
            left_operand = stack.pop()
            result = f'{token} {left_operand} {right_operand}'
            stack.append(result)
    
    return stack[-1]

def pn_to_infix(pn_expression):
    stack = []
    if isinstance(pn_expression, str):
        pn_expression = pn_expression.split()
    for token in pn_expression[::-1]:
        if not is_operator(token): #other
            stack.append(token)
        elif is_unary_operator(token): #unary operator
            operand = stack.pop()
            result = f'{token}({operand})'
            stack.append(result)
        else: #binary operator
            right_operand = stack.pop()
            left_operand = stack.pop()
            result = f'({right_operand} {token} {left_operand})'
            stack.append(result)
    
    return stack[-1]

#https://stackoverflow.com/a/77180279/18255427
#Returns two values, depth and if the expression is complete
def getPNdepth(expression: list[str]):
    if not expression: #if it's empty
        return 0, False
    if isinstance(expression, str):
        expression = expression.split()
    stack = []
    depth, num_binary, num_leaves = 0, 0, 0
    for val in expression:
        if is_binary_operator(val):  # all binary operators
            stack.append(2)  # = number of operands
            num_binary += 1
        elif is_unary_operator(val):  # all unary operators
            stack.append(1)
        else:  # an operand (x)
            num_leaves += 1
            while stack and stack[-1] == 1:  # remove fulfilled operators
                stack.pop()
            if stack:  # indicate an operand is consumed
                stack[-1] -= 1
        depth = max(depth, len(stack) + 1)
    return depth-1, num_leaves == num_binary + 1

#https://stackoverflow.com/a/77128902/18255427
#Returns two values, depth and if the expression is complete
def getRPNdepth(expression):
    if not expression: #if it's empty
        return 0, False
    stack = []
    if isinstance(expression, str):
        expression = expression.split()
    for token in expression:
        if is_unary_operator(token):  # all unary operators
            stack[-1] += 1
        elif is_operator(token):  # all binary operators
            stack.append(max(stack.pop(), stack.pop()) + 1)
        else:  # an operand (x)
            stack.append(1)
    complete = True
    while len(stack) > 1:
        stack.append(max(stack.pop(), stack.pop()) + 1)
        complete = False #If the stack length is greater than 1 then expression is an INCOMPLETE RPN expression
    return stack.pop()-1, complete

#https://www.geeksforgeeks.org/dsa/prefix-postfix-conversion/
def prefix_to_postfix(prefix_expr):
    prefix_expr = prefix_expr.split() if (type(prefix_expr) == str) else prefix_expr #Convert prefix_expr to a list if it's a string
    stack = []
    for token_idx in range(len(prefix_expr)-1, -1, -1):
        if is_binary_operator(prefix_expr[token_idx]):
            a = stack.pop()
            b = stack.pop()
            stack.append(f"{a} {b} {prefix_expr[token_idx]}")
        elif is_unary_operator(prefix_expr[token_idx]):
            a = stack.pop()
            stack.append(f"{a} {prefix_expr[token_idx]}")
        else:
            stack.append(prefix_expr[token_idx])
    return stack[-1]

#https://www.geeksforgeeks.org/dsa/postfix-prefix-conversion/
def postfix_to_prefix(postfix_expr):
    postfix_expr = postfix_expr.split() if (type(postfix_expr) == str) else postfix_expr #Convert postfix_expr to a list if it's a string
    stack = []

    for token_idx in range(len(postfix_expr)):
        if is_binary_operator(postfix_expr[token_idx]):
            a = stack.pop()
            b = stack.pop()
            stack.append(f"{postfix_expr[token_idx]} {b} {a}") #a b + -> + a b
        elif is_unary_operator(postfix_expr[token_idx]):
            a = stack.pop()
            stack.append(f"{postfix_expr[token_idx]} {a}")
        else:
            stack.append(postfix_expr[token_idx])
    return stack[-1]

'''
std::vector<std::string> complete_tree(const std::vector<std::string>& expression, const std::string& notation)
{
    int expr_depth;
    bool extended = true;
    if (notation == "prefix")
    {
        expr_depth  = this->getPNdepth(expression).first;
        std::vector<std::string> expr_to_insert = {"+", "0"};
        std::vector<std::string> temp_expression;
        while (extended)
        {
            extended = false;
            for (size_t i = 0; i < expression.size(); i++)
            {
                if (is_const(expression[i]))
                {
                    temp_expression = expression;
                    temp_expression.insert(temp_expression.begin() + i, expr_to_insert.begin(), expr_to_insert.end());
                    if (this->getPNdepth(temp_expression) == expr_depth)
                    {
                        expression = temp_expression;
                        extended = true;
                        break;
                    }
                }
            }
        }
    }
    else //postfix
    {
        expr_depth  = this->getRPNdepth(expression).first;
        while (extended)
        {
            extended = false;
            std::vector<std::string> temp_expression;
            for (size_t i = 0; i < expression.size(); i++)
            {
                if (is_const(expression[i]))
                {
                    temp_expression = expression;
                    temp_expression.insert(temp_expression.begin() + i, "0");
                    temp_expression.insert(temp_expression.begin() + i + 2, "+");
                    if (this->getRPNdepth(temp_expression) == expr_depth)
                    {
                        expression = temp_expression;
                        extended = true;
                        break;
                    }
                }
            }
        }
    }
    return expression;
}

'''

def complete_tree(expression, notation):
    if notation == "prefix":
        expr_depth = getPNdepth(expression) #get depth of tree that we want to keep the same at all times!!
        extended = True
        while extended:
            extended = False
            for token_idx in range(len(expression)):
                if is_operand(expression[token_idx]):
                    temp_expression = deepcopy(expression) #copy the whole expression
                    temp_expression[token_idx:token_idx+1] = ["+", "0", expression[token_idx]] #replace `node` with `["+", "0", node]`
                    if getPNdepth(temp_expression) == expr_depth:
                        expression = deepcopy(temp_expression) #copy the test-substitution-expression into the one we're completing
                        extended = True
                        break
    else: #postfix
        expr_depth = getRPNdepth(expression) #get depth of tree that we want to keep the same at all times!!
        extended = True
        while extended:
            extended = False
            for token_idx in range(len(expression)):
                if is_operand(expression[token_idx]):
                    temp_expression = deepcopy(expression) #copy the whole expression
                    temp_expression[token_idx:token_idx+1] = ["0", expression[token_idx], "+"] #replace `node` with `["0", node, "+"]`
                    if getRPNdepth(temp_expression) == expr_depth:
                        expression = deepcopy(temp_expression) #copy the test-substitution-expression into the one we're completing
                        extended = True
                        break
    return expression
    
called = False
implot = None
def plot_pn_expression_tree(expression: list[str], save = False, include_expression_in_title = True):
    global called, implot

    def build_tree(expression_tokens):
        stack = deque()
        unique_id = 0

        for token in expression_tokens:
            if not is_operator(token):
                unique_id += 1
                node = Node(token, unique_id)
                stack.append(node)
            elif is_unary_operator(token):
                child_operand = stack.pop()
                unique_id += 1
                operator_node = UnaryNode(token, unique_id)
                operator_node.child = child_operand
                stack.append(operator_node)
            else:
                right_operand = stack.pop()
                left_operand = stack.pop()
                unique_id += 1
                operator_node = BinaryNode(token, unique_id)
                operator_node.right = left_operand
                operator_node.left = right_operand
                stack.append(operator_node)
        return stack.pop()

    def plot_tree(node, graph, parent=None):
        if node:
            current_node = pydot.Node(str(node.unique_id), label=str(node.value))
            graph.add_node(current_node)

            if parent:
                edge = pydot.Edge(str(parent.unique_id), str(node.unique_id))
                graph.add_edge(edge)

            if isinstance(node, BinaryNode):
                plot_tree(node.left, graph, node)
                plot_tree(node.right, graph, node)
            elif isinstance(node, UnaryNode):
                plot_tree(node.child, graph, node)

    expression_tree = build_tree(expression[::-1])

    graph = pydot.Dot(graph_type='graph')
    plot_tree(expression_tree, graph)
    
    if save:
        graph.set('label', f"{' '.join(expression)}, depth = {getPNdepth(expression)[0]}" if include_expression_in_title else f"depth = {getPNdepth(expression)[0]}")
        graph.set('labelloc', 't')  # Set label location to "top"
        graph.write_svg('expression_tree_PN_Hemberg2008_expr_5.svg')
    else:
        graph.write_png('expression_tree.png')
#        if called == False or block == True:
        implot = plt.imshow(plt.imread('expression_tree.png'))
#            called = True
#        else:
        implot.set_data(plt.imread('expression_tree.png'))
        plt.axis('off')
        plt.title(f"{' '.join(expression)}, depth = {getPNdepth(expression)[0]}" if include_expression_in_title else f"depth = {getPNdepth(expression)[0]}")
        plt.show(block = True)

def plot_rpn_expression_tree(expression: list[str], save = False, filename = "", title = "", tolatex = False, to_pdf = False, include_expression_in_title = True):
    global called, implot

    def build_tree(expression_tokens):
        stack = deque()
        unique_id = 0
    
        for token in expression_tokens:
            if not is_operator(token):
                unique_id += 1
                node = Node(token, unique_id)
                stack.append(node)
            elif is_unary_operator(token):
                child_operand = stack.pop()
                unique_id += 1
                operator_node = UnaryNode(token, unique_id)
                operator_node.child = child_operand
                stack.append(operator_node)
            else:
                right_operand = stack.pop()
                left_operand = stack.pop()
                unique_id += 1
                operator_node = BinaryNode(token, unique_id)
                operator_node.left = left_operand
                operator_node.right = right_operand
                stack.append(operator_node)
        return stack.pop()

    def plot_tree(node, graph, parent=None):
        if node:
            current_node = pydot.Node(str(node.unique_id), label=str(node.value))
            graph.add_node(current_node)

            if parent:
                edge = pydot.Edge(str(parent.unique_id), str(node.unique_id))
                graph.add_edge(edge)

            if isinstance(node, BinaryNode):
                plot_tree(node.left, graph, node)
                plot_tree(node.right, graph, node)
            elif isinstance(node, UnaryNode):
                plot_tree(node.child, graph, node)

    if isinstance(expression, str):
        expression = expression.split()
    expression_tree = build_tree(expression)

    graph = pydot.Dot(graph_type='graph')
    plot_tree(expression_tree, graph)
    
    
    if save:
        graph.set('label', title)
        graph.set('labelloc', 't')  # Set label location to "top"
        graph.write_svg(filename)
        print(f"Image file saved as {filename}")
        if tolatex:
            # Export to tex
            replace_dict = {"MYTAU": r"\tau", "MYTHETA": r"\theta", "MYETA": r"\eta", "MYCDOT": r"\cdot", "MYNESTEROV": r"\text{Nesterov}", "MYSIGMA": r"\sigma", "MYEPSILON": r"\epsilon", "MYFRAC": r"\frac", "MYBRACKETSQRT": r"\sqrt{}", "MYSQRT": r"\sqrt", "MYHSPACE": r"\hspace", "MYGAMMA": r"\gamma", "MYLEFT": r"\left", "MYRIGHT": r"\right", "MYTEXTA": r"\text{A}", "MYTEXTDADELTA": r"\text{dadelta}", "MYDELTA": r"\Delta ", "MYMUADAM": r"\widehat{\mu}_{j,m,t=\tau}", "MYNUADAM": r"\widehat{\nu}_{j,m,t=\tau}", "MYSIN": r"\sin", "MYCOS": r"\cos", "MYTAN": r"\tan", "MYLAMBDA": r"\lambda", "MYPOW": r"\wedge", "X02": r"x_0^2"}
            texcode = dot2tex.dot2tex(graph.to_string(),format='tikz',texmode='math',crop=True)
            for replacement in replace_dict:
                texcode = texcode.replace(replacement, replace_dict[replacement])
#            texcode = texcode.replace(r"\usepackage{amsmath}", r"\usepackage{amsmath}""\n"r"\usepackage{scalerel}""\n")
            filename = filename[:filename.find('.')]+".tex"
            with open(f"{filename}", "w") as f:
                f.write(texcode)
            print(f"Latex file saved as {filename}")
            if to_pdf:
                status = os.system(f"/Library/TeX/texbin/pdflatex {filename}")
                if not status:
                    print(f"Pdf file saved as {filename.replace('.tex','.pdf')}")
    else:
        graph.write_png('expression_tree.png')
        implot = plt.imshow(plt.imread('expression_tree.png'))
        plt.axis('off')
        plt.title(title if title else f"{' '.join(expression)}, depth = {getRPNdepth(expression)[0]}" if include_expression_in_title else f"depth = {getRPNdepth(expression)[0]}")
        plt.show()

def test_visualize():
    save = False
    if save:
        file_names = ("GradientDescent", "HeavyBall", "Nesterov", "AdaGrad", "RMSProp", "AdaDelta", "Adam", "AdamW",\
         "nasty_edward_equation",\
         "eq:example_pysr_equation"\
         )
        expressions = (r"w_{j,m,t=MYTAU-1} MYETA g_{j,m,t=MYTAU} MYCDOT +", \
                       r"w_{j,m,t=MYTAU-1} MYTHETA v_{j,m,t=MYTAU-1} MYCDOT MYETA g_{j,m,t=MYTAU} MYCDOT + +", \
                       r"w_{j,m,t=MYTAU-1} MYTHETA v_{j,m,t=MYTAU-1} MYCDOT MYETA d_{j}^{MYNESTEROV} y_{i,m,t=MYTAU} MYCDOT MYCDOT + +", \
                       r"w_{j,m,t=MYTAU-1} MYETA g_{j,m,t=MYTAU} MYCDOT MYSIGMA_{MYHSPACE{-.05cm}g^{2}_{j,m}} MYEPSILON + MYBRACKETSQRT / +", \
                       r"w_{j,m,t=MYTAU-1} MYETA g_{j,m,t=MYTAU} MYCDOT EMYLEFT[g_{j,m}^2MYRIGHT]_{t=MYTAU} MYEPSILON + MYBRACKETSQRT / +", \
                       r"w_{j,m,t=MYTAU-1} MYDELTAw^{MYTEXTAMYHSPACE{-.018cm}MYTEXTDADELTA}_{j,m,t=MYTAU} -", \
                       r"w_{j,m,t=MYTAU-1} MYETA MYMUADAM MYCDOT MYNUADAM MYEPSILON + MYBRACKETSQRT / +", \
                       r"w_{j,m,t=MYTAU-1} MYETA  MYLAMBDA w_{j,m,t=MYTAU-1} MYCDOT MYMUADAM MYNUADAM MYBRACKETSQRT MYEPSILON + /  + MYCDOT +",\
                       r"x 3 x MYCOS x MYSIN MYSIN - MYCDOT + MYTAN MYCOS MYSIN", \
                       r"2 x_3 MYCOS * x_0 2 MYPOW + 2 -"
                       )
        titles = (r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYETA MYCDOT g_{j,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYTHETA MYCDOT v_{j,m,t=MYTAU-1} + MYETA MYCDOT g_{j,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYTHETA MYCDOT v_{j,m,t=MYTAU-1} + MYETA MYCDOT d_{j}^{MYNESTEROV} MYCDOT y_{i,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYFRAC{MYETA MYCDOT g_{j,m,t=MYTAU}}{MYSQRT{MYSIGMA_{MYHSPACE{-.05cm}g^{2}_{j,m}} + MYEPSILON}}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYFRAC{MYETA MYCDOT g_{j,m,t=MYTAU}}{MYSQRT{EMYLEFT[g_{j,m}^2MYRIGHT]_{t=MYTAU} + MYEPSILON}}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} - MYDELTAw^{MYTEXTAMYHSPACE{-.018cm}MYTEXTDADELTA}_{j,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYFRAC{MYETA MYCDOT MYMUADAM}{MYSQRT{MYNUADAM} + MYEPSILON}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYETA MYCDOT MYLEFT(MYLAMBDA MYCDOT w_{j,m,t=MYTAU-1} + MYFRAC{MYMUADAM}{MYSQRT{MYNUADAM} + MYEPSILON}MYRIGHT)",\
                  r"f(x) = MYSIN(MYCOS(MYTAN(x+3 MYCDOT (MYCOS(x) - MYSIN(MYSIN(x)))))",\
                  r"2 MYCDOT MYCOS(x_3) + X02 - 2"
                  )
                  
        for file_name, expression, title in list(zip(file_names, expressions, titles))[-1:]:
            plot_rpn_expression_tree(expression = expression, save = True, filename = f"{file_name}.svg", title = title, tolatex=True, to_pdf=True)
            os.system(f"open -a Xcode {file_name}.tex")
            os.system(f"open -a Safari {file_name}.pdf")
    else:
        pn_to_rpn = False
        rpn_to_pn = False
        in_to_rpn = False
        if pn_to_rpn:
            prefix_expr = "+ + * sin * + x0 sin sqrt x0 0.13599420224810638 ~ + * 0.204703 x0 -36.871109 * * x0 0.276493 - -1.868178 sin * 1.095518 sqrt x0 * -5.206597 cos * x0 - 1 sin 0.957523"
            print(prefix_to_postfix(prefix_expr))
        elif rpn_to_pn:
            postfix_expr = "-2.444296 x0 1.026203 x0 sqrt cos ^ ^ * -26.199496 4.207354924039483 168.000000 x0 + sqrt * cos * - x0 25.019477 1.014280 x0 ^ cos tanh * + + -8.058567 -2.633602 x0 0.6931471805599453 ^ + cos * - 11.696530 118.95633426995997 x0 0.995059 / - / + x0 0.3670833851233197 * sin 4.957468 * - 1.502449 x0 0.997889 ^ ^ log cos + -2.060603 44.498356 x0 + 0.2658022288340797 * sin * exp + x0 -0.45018598229727835 * sin -2.701240 * + -1.599485 x0 -1.6880058284590451 / cos * - x0 0.696976831813758 x0 * cos + + 0.032499 4 x0 0.7615941559557649 - * sin / - 2.702079 1.052177 x0 0.992888 ^ ^ + sqrt sin + 1.027621 2 x0 0.8414709848078965 ^ * * cos + 1.225249 2.020931 11.175638 x0 54.598150033144236 - + / - - -6.130043 1.218107 x0 + x0 6.804936 - cos / / - -0.032252 1.134681 x0 + sqrt cos asin / +"
            print(postfix_to_prefix(postfix_expr))
        elif in_to_rpn:
            infix_expr = '-0.285806921654494**(r + 10.0139164646307)*(r + (r**0.999993025405072 - 5.00008333556817e-5)**(r**0.01))**(0.0166848951652189**(6.23978883640503/(r + 2)) + 0.000631778468553939*r + 7.59291602260893)*(-sin(theta + cos(theta) + 1/r) + sin(log(r))) + 0.606923362578475*sqrt(1 - cos(r)**2)*(sech(r + 10) + 0.999884875453817)**((r + 0.02)**4.03*(1.58*(tanh(.59*r)))/(sin(sech(r)) + 0.693147180559945))*sin(theta + 6.28319) + 0.886342906953379'.replace("**","^")
                
            infix_expr_subbed = rpn_to_infix(rpn_expr:=infix_to_rpn(infix_expr))
            
            assert(infix_expr_subbed.replace("(","").replace(")","").replace(" ","").replace("~","-") == infix_expr.replace("(","").replace(")","").replace(" ",""))
            
            rpn1 = infix_to_rpn(infix_expr).split()
            infix2 = rpn_to_infix(' '.join(rpn1))
            rpn2 = infix_to_rpn(infix2).split()
            assert rpn1 == rpn2
            print(f"rpn_expr = {rpn_expr}")
        else:
            expression_type_to_plot = ["prefix", "postfix"][1]
            completeTree = [True, False][1]
            if expression_type_to_plot == "prefix":
                complete_pn_expr = "+ ^ x 3 1"
                if completeTree:
                    complete_pn_expr = complete_tree(complete_pn_expr.split(), "prefix") #returns a list
                else:
                    complete_pn_expr = complete_pn_expr.split()
                print(f"complete_pn_expr = \n{' '.join(complete_pn_expr)}")
                print(f"len(complete_pn_expr) = {len(complete_pn_expr)}")
                plot_pn_expression_tree(complete_pn_expr, save = save, include_expression_in_title = False)

            else:
                complete_rpn_expr = "-17.777416876229204 x0 -5.4145382360994985 * sin * x0 + -15.571405735031114 13.412300028120239 x0 * sin * -3.442896885208855 + + -7.111904853685272 x0 9.840284926874622 * sin * + x0 x0 4 * 8 * + cos + 1.2633853833823223 -3.4758858801587644 1.511154811904955 x0 * + sech + * -92.51479676044292 x0 cos * cos -1.3862403756907975 x0 + * + 0.3473517217795077 x0 -23.531916581060983 - x0 arccos * cos * - -0.787270308886979 x0 * 0.11490012736367276 x0 - sech acos + +"
                print(f"rpn_to_infix = {rpn_to_infix(complete_rpn_expr)}")
                
                
                
                
                
                if completeTree:
                    complete_rpn_expr = complete_tree(complete_rpn_expr.split(), "postfix") #returns a list
                else:
                    complete_rpn_expr = complete_rpn_expr.split()
                print(f"complete_rpn_expr = \n{' '.join(complete_rpn_expr)}")
                print(f"len(complete_rpn_expr) = {len(complete_rpn_expr)}")
                plot_rpn_expression_tree(complete_rpn_expr, save = save, include_expression_in_title = False, title='')

if __name__ == "__main__":
    test_visualize()


