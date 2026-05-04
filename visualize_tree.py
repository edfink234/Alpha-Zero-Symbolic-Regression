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
            prefix_expr = "- + * + - - - - - * + * * * + + + * - * x0 21.98251102023205 0.1696190518730082 * - ~ + * * - * * 13.641233388900535 x0 + x0 1.8818987007940742 - ~ acos sin * x0 4 12.69386693673799 + 0.37164538626278115 arccos sqrt sqrt x0 - x0 cos + -0.3238814038415719 cos x0 5.038385142644697 * * 4 - x0 1 + * - 0.6911266210253895 x0 * -0.1926020595847093 sin * 10.671446617851876 x0 -5.81663141054113 acos sech arccos x0 - * sqrt x0 * -3.944315031368609 4 3.700045228092431 * 83.26636307516488 tanh * 4 x0 * -0.1513537486330349 sin * x0 28.55159091259145 ~ - x0 1.038496248239717 1.2461076172271626 - 1.0045833281395233 sqrt x0 * 11.265804084395839 acos sech - cos x0 0.7184073949666487 4 asin * 0.013010563142755166 cos * 69.41897804095233 x0 * cos * x0 - -61.73609141520944 * x0 4.637872135650893 0.010940773676288767 * arcsin sech - 0.4110442138054884 sqrt x0 3.2298478081246653 * -0.57975678301296 sech arcsin x0 ~ * 3.243379230051854 sech * -11.832940848891166 tanh x0 * 0.001956972023114659 sin * x0 -96.23722600552559 9.570796326794897 sech + 7.3434212372775844 * * asin - 0.7710437902358543 x0 9.769687010344823 -417.04456715206885 - 0.1593079621673323 acos - cos * x0 4.1255761011132455 5.549663731896359e-07"
            print(prefix_to_postfix(prefix_expr))
        elif rpn_to_pn:
            postfix_expr = "x0 21.98251102023205 * 0.1696190518730082 - 13.641072185670984 x0 * x0 -1.9182181920524706 - * x0 4 * sin acos ~ 13.060302602534446 - - 0.37009864755589583 x0 sqrt sqrt arccos + * x0 -0.3238814038415719 x0 cos + cos - * -4.999614381245442 - ~ 4 x0 1 - * x0 -0.6903908419736177 + 0.19264923863345074 10.671446617851876 x0 * sin * * -5.9236947648462115 + * - x0 arccos sech acos * * x0 sqrt 2 8.062197397945244 * * 3.6488068441137274 + - -83.36980539290025 4 x0 * tanh * - -0.15101595617913127 x0 28.55159091259145 * sin * + x0 1.038496248239717 - ~ * 1.2470494087388007 * 1.0045016093755044 x0 sqrt - * -11.264738501159213 x0 cos 0.7184073949666487 - sech acos * - 4 * 0.013010563142755166 69.41897804095233 x0 * cos * asin - x0 -61.73609141520944 x0 4.637872135650893 * - * cos 0.010938233414480353 * - 0.4110442138054884 x0 sqrt - sech arcsin -3.221477401399012 * + 0.5778123512012252 x0 arcsin sech * + -3.1351670114885692 -11.832940848891166 x0 asin * sech * ~ + 0.00195798755829556 x0 -96.23722600552559 * sin * + 9.570796326794897 * 7.3434212372775844 0.7710437902358543 x0 - asin 9.769687010344823 * -417.04456715206885 * + sech + 0.2069850124754552 x0 4.1255761011132455 * cos 5.549663731896359e-07 - acos - -"
            print(postfix_to_prefix(postfix_expr))
            assert(prefix_to_postfix(postfix_to_prefix(postfix_expr)) == postfix_expr)
        elif in_to_rpn:
            infix_expr = '-0.285806921654494**(r + 10.0139164646307)*(r + (r**0.999993025405072 - 5.00008333556817e-5)**(r**0.01))**(0.0166848951652189**(6.23978883640503/(r + 2)) + 0.000631778468553939*r + 7.59291602260893)*(-sin(theta + cos(theta) + 1/r) + sin(log(r))) + 0.606923362578475*sqrt(1 - cos(r)**2)*(sech(r + 10) + 0.999884875453817)**((r + 0.02)**4.03*(1.58*(tanh(.59*r)))/(sin(sech(r)) + 0.693147180559945))*sin(theta + 6.28319) + 0.886342906953379'.replace("**","^")
                
            infix_expr_subbed = rpn_to_infix(rpn_expr:=infix_to_rpn(infix_expr))
            
            assert(infix_expr_subbed.replace("(","").replace(")","").replace(" ","").replace("~","-") == infix_expr.replace("(","").replace(")","").replace(" ",""))
            
            rpn1 = infix_to_rpn(infix_expr).split()
            infix2 = rpn_to_infix(' '.join(rpn1))
            rpn2 = infix_to_rpn(infix2).split()
            assert rpn1 == rpn2
            print(f"rpn_expr = {rpn_expr}")
            
            infix_expr = '3.99983127828682*s*(0.997951315766111 - s)*(1.71943507419579 - s)*(-5.55321096741672*s - 0.0808022569021439)*(sech(3.25576907573188*s - 1.47328476296863) - 0.757036240687343)*asin(cos((-4*s - 37.4148627574806)*(s + tanh(s) + 4.00988486577666))) + 535.791129714912*s*(-10.0438824571593*sqrt(s) - 7.85393482869936*s + 7.2608049778655*tanh(2*s) + tanh(31.1886593496809*s) + 4) - s*acos(sin(15.660895958963*s - 1.07287218013174))*asin(sin(s + 0.530441002019499) - 0.929424794844055)*asin(cos(126.807451395004*s - 1.59904433588435)) + 0.12032331554506*s*asin(cos(1.01394806597437*s*(3.96053881308568*s + 102.332719767656) + 1.28883645260147)) - 1081.87515476891*s*sech(s) + 4.20996960216403*sqrt(1 - 0.997190199730815*tanh((s - 0.680652799864571)*asin(s) + cos(144.066669905664*s) + 4.01542690360951)) - (30.2073738717681 - 46.2830360167089*sin(11.8214021675704*s))*(s - 0.687877353823327)*acos(s) - (0.0253116964212297*s + 0.0133047041476922)*sin((2.00349499924616*s + 5.01363101427281)*(21.8880204857904*s + 34.3782807660179)) + (4.02426531026755*s + 4.24143468386376)*(-sin(s) + asin(s))*(1613.54721263464*acos(s) + 1167.0833877051)*cos(sqrt(s)) + (s + cos(s) + 4.5290985447787)*sin(s**0.253115085305453 - sech((2.26959539551553*sqrt(s) - 1.18184538973023)*asin(2*s - 1) + 0.630159177803622))*asin(sin(6.51293827457812*s**0.253115085305453 - 0.584280255805762)) - 1.45487198432489*sin(s*(43.654901077881 - 2.13010047659353*s)) + 0.0137318149879915*cos(s*(4.19304438906306*s*(s + 11.3900539738965) - 80.7777051072293*s - 327.042784403805)) - tanh((-0.0153757791779974*s - 0.0225413795197801)*sin((6.18348189177986*s - 31.1618822833951)*(9*s + 0.90090984221334))) + sqrt(acos(s)) - 0.373973753491818*acos(sech(asin(cos(6.10728833191319*s + 4*cos(15.9824169394057*s) + 0.39820193313207)))) + asin(s) - 0.578941799827086*asin(cos(1.4397376761908*cos(3.96842459046176*s) + 168.670031135102*sech(s) - 1.3938355403149)) - 71.1672160791384*asin(sech(3.99531513262175*s - 0.596147572686443)) + 0.84473466768219*sech(tanh(6.02767885961134*s)*tanh(tanh(sqrt(acos(-sin(9.59257420842415*sqrt(s)*(s + cos(sqrt(sech(s))) + 1.44723625814732) + 153.450793947131*s - 1.1013914630869))) - 0.878570648952716))) + 52.2640785824516*sech(cos(10.9128178515675*s - 0.267382742428652)) - 1.2024674214492'.replace("**","^")
            
            infix_expr_subbed = rpn_to_infix(rpn_expr:=infix_to_rpn(infix_expr))

            assert(infix_expr_subbed.replace("(","").replace(")","").replace(" ","").replace("~","-") == infix_expr.replace("(","").replace(")","").replace(" ",""))

            rpn1 = infix_to_rpn(infix_expr).split()
            infix2 = rpn_to_infix(' '.join(rpn1))
            rpn2 = infix_to_rpn(infix2).split()
            assert rpn1 == rpn2
            print(f"rpn_expr = {' '.join([i if i != 's' else 'x0' for i in rpn_expr.split()])}")
        else:
            expression_type_to_plot = ["prefix", "postfix"][1]
            completeTree = [True, False][1]
            if expression_type_to_plot == "prefix":
                complete_pn_expr = "8.90965282330888 59.89296187462085 x0 * 17.709631503758562 31.430444972044043 x0 * tanh x0 10.752797498412956 + x0 sqrt * x0 5.872902394660459 * + ~ + + 9.880074292925372 x0 tanh tanh * + * x0 acos 0.5922290901693024 4 + * 0.7764612616244088 x0 - 76.89127635722234 11.8131780914971 x0 * sin - * * + 4 4 -29.14453478867909 * * + * x0 x0 + cos 4.004190002943048 x0 230.35684238247097 * * * - 1.3227253400099304 x0 20.226114315391563 x0 - 22.525655106586292 + * sin * - 50.44170205297188 -12.816822683581922 -10.914784548298355 x0 * - cos sech * + x0 sqrt 5.654621199258812 130.3029718242763 * - x0 arccos tanh * - -277.4668940523251 x0 sin 0.14855711697939927 - sech asin * +"
                if completeTree:
                    complete_pn_expr = complete_tree(complete_pn_expr.split(), "prefix") #returns a list
                else:
                    complete_pn_expr = complete_pn_expr.split()
                print(f"complete_pn_expr = \n{' '.join(complete_pn_expr)}")
                print(f"len(complete_pn_expr) = {len(complete_pn_expr)}")
                plot_pn_expression_tree(complete_pn_expr, save = save, include_expression_in_title = False)

            else:
                complete_rpn_expr = '-16.777616570512944 x0 * 0.8306189265029301 - 38.40359043830019 x0 arcsin * x0 sqrt sqrt sqrt + x0 1.1415951307864396 + sin arccos 1.0223661471440606 x0 cos - x0 -21.13890720974086 * * * - x0 x0 arcsin sech 101.28256646428251 * * + 8.259465504453356 - ~ x0 sqrt 0.6249489763711806 - 64.54590489933697 0.5683023211239526 x0 - * x0 11.330638763373788 * sin + 0.5635392600438554 * * + x0 sin 4.0584308964672235 * 26.77386670959251 4.217961269161968 x0 sqrt * + * + * 6.22514055685562 -0.595915710381489 x0 sqrt + * + 4.0000437572893865 x0 -3.999897110430852 + * * -1.4568317419473362 x0 42.922593655687336 * sin * - 19.83996633272047 -6.683555321462922 x0 * 2.0000119786540065 * cos tanh * + -1.0503963064944712 -36.828669687140874 x0 * cos * + x0 acos * 135.88933938962452 0.8961900453685253 x0 acos - sech acos * + -103.6408388759151 + x0 arccos 0.22470061145876485 * 82.5216355471646 x0 tanh sin x0 x0 * tanh sech - * sin arccos ~ * + 0.044851480438137546 -0.7078732093049552 x0 + x0 x0 * * - -19.867254923459424 x0 * cos -0.2880773968715205 -0.2706328594194209 x0 sqrt * - * + x0 x0 0.9279324129934339 - -15.965336642805138 x0 119.8461111944218 * sin ~ * * * * - -3.238299931836995 1.6562707812807105 x0 sin * sech 3.9998470155400883 3.9999824038768184 1.0124551940002344 1.0594336770154307 4.000044663534493 x0 * * + * * * * cos 0.8063394511057888 x0 0.948739678138467 - x0 6.408877170737145 x0 -9.514769954551296 * + * + sech - arcsin * + x0 -0.5408073272003451 + 4.412171189196541 x0 tanh -0.46525910568678763 2.0054902337430605 x0 - x0 x0 + tanh x0 arcsin x0 1.27721603967268 - * * * - x0 arcsin 0.9640297593509776 + -951.2504369851667 2 x0 acos * * * cos asin * * * * - 0.014612208993260944 -0.7630332746033869 x0 + -0.11179427640609108 x0 * * + 0 arccos 4.367279424945187 0.03304613151477807 x0 x0 * - * 1.1432087294423803 x0 - 2 4 0 - * + x0 4 2 4 * * * * + sin arccos ~ + * +'
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


