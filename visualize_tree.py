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
    return token in {'+', '-', '*', '/', '^', 'MYCDOT'}
def is_unary_operator(token):
    return token in {"cos", "exp", "sqrt", "sin", "asin", "arcsin", "log", "tanh", "acos", "arccos", "~", "ln", "MYBRACKETSQRT", "tan", "MYCOS", "MYSIN", "MYTAN", "sech"}
def is_operand(token):
    return not is_operator(token)

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
    for token_idx in range(len(prefix_expr)-1, -1, -1):
        pass

#https://www.geeksforgeeks.org/dsa/postfix-prefix-conversion/
def postfix_to_prefix(postfix_expr):
    for token_idx in range(len(postfix_expr)):
        pass

#TODO: Get questions about paper that we might want to explore further

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
            replace_dict = {"MYTAU": r"\tau", "MYTHETA": r"\theta", "MYETA": r"\eta", "MYCDOT": r"\cdot", "MYNESTEROV": r"\text{Nesterov}", "MYSIGMA": r"\sigma", "MYEPSILON": r"\epsilon", "MYFRAC": r"\frac", "MYBRACKETSQRT": r"\sqrt{}", "MYSQRT": r"\sqrt", "MYHSPACE": r"\hspace", "MYGAMMA": r"\gamma", "MYLEFT": r"\left", "MYRIGHT": r"\right", "MYTEXTA": r"\text{A}", "MYTEXTDADELTA": r"\text{dadelta}", "MYDELTA": r"\Delta ", "MYMUADAM": r"\widehat{\mu}_{j,m,t=\tau}", "MYNUADAM": r"\widehat{\nu}_{j,m,t=\tau}", "MYSIN": r"\sin", "MYCOS": r"\cos", "MYTAN": r"\tan", "MYLAMBDA": r"\lambda"}
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
#        print(pn_to_infix(" - - + / ^ x 3 5 / ^ y 3 2 y x".split()))
#        print(rpn_to_infix("y y x * * cos y +"))
        file_names = ("GradientDescent", "HeavyBall", "Nesterov", "AdaGrad", "RMSProp", "AdaDelta", "Adam", "AdamW",\
         #"nasty_edward_equation"\
         )
        expressions = (r"w_{j,m,t=MYTAU-1} MYETA g_{j,m,t=MYTAU} MYCDOT +", \
                       r"w_{j,m,t=MYTAU-1} MYTHETA v_{j,m,t=MYTAU-1} MYCDOT MYETA g_{j,m,t=MYTAU} MYCDOT + +", \
                       r"w_{j,m,t=MYTAU-1} MYTHETA v_{j,m,t=MYTAU-1} MYCDOT MYETA d_{j}^{MYNESTEROV} y_{i,m,t=MYTAU} MYCDOT MYCDOT + +", \
                       r"w_{j,m,t=MYTAU-1} MYETA g_{j,m,t=MYTAU} MYCDOT MYSIGMA_{MYHSPACE{-.05cm}g^{2}_{j,m}} MYEPSILON + MYBRACKETSQRT / +", \
                       r"w_{j,m,t=MYTAU-1} MYETA g_{j,m,t=MYTAU} MYCDOT EMYLEFT[g_{j,m}^2MYRIGHT]_{t=MYTAU} MYEPSILON + MYBRACKETSQRT / +", \
                       r"w_{j,m,t=MYTAU-1} MYDELTAw^{MYTEXTAMYHSPACE{-.018cm}MYTEXTDADELTA}_{j,m,t=MYTAU} -", \
                       r"w_{j,m,t=MYTAU-1} MYETA MYMUADAM MYCDOT MYNUADAM MYEPSILON + MYBRACKETSQRT / +", \
                       r"w_{j,m,t=MYTAU-1} MYETA  MYLAMBDA w_{j,m,t=MYTAU-1} MYCDOT MYMUADAM MYNUADAM MYBRACKETSQRT MYEPSILON + /  + MYCDOT +",\
                       #r"x 3 x MYCOS x MYSIN MYSIN - MYCDOT + MYTAN MYCOS MYSIN", \

                       )
        titles = (r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYETA MYCDOT g_{j,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYTHETA MYCDOT v_{j,m,t=MYTAU-1} + MYETA MYCDOT g_{j,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYTHETA MYCDOT v_{j,m,t=MYTAU-1} + MYETA MYCDOT d_{j}^{MYNESTEROV} MYCDOT y_{i,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYFRAC{MYETA MYCDOT g_{j,m,t=MYTAU}}{MYSQRT{MYSIGMA_{MYHSPACE{-.05cm}g^{2}_{j,m}} + MYEPSILON}}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYFRAC{MYETA MYCDOT g_{j,m,t=MYTAU}}{MYSQRT{EMYLEFT[g_{j,m}^2MYRIGHT]_{t=MYTAU} + MYEPSILON}}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} - MYDELTAw^{MYTEXTAMYHSPACE{-.018cm}MYTEXTDADELTA}_{j,m,t=MYTAU}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYFRAC{MYETA MYCDOT MYMUADAM}{MYSQRT{MYNUADAM} + MYEPSILON}", \
                  r"w_{j,m,t=MYTAU} = w_{j,m,t=MYTAU-1} + MYETA MYCDOT MYLEFT(MYLAMBDA MYCDOT w_{j,m,t=MYTAU-1} + MYFRAC{MYMUADAM}{MYSQRT{MYNUADAM} + MYEPSILON}MYRIGHT)",\
                  #r"f(x) = MYSIN(MYCOS(MYTAN(x+3 MYCDOT (MYCOS(x) - MYSIN(MYSIN(x)))))",
                  )
                  
                  
                  
        for file_name, expression, title in zip(file_names, expressions, titles):
#        for file_name, expression, title in [list(zip(file_names, expressions, titles))[-1]]:
            plot_rpn_expression_tree(expression = expression, save = True, filename = f"{file_name}.svg", title = title, tolatex=True, to_pdf=True)
            os.system(f"open -a Xcode {file_name}.tex")
            os.system(f"open -a Safari {file_name}.pdf")
    else:
#        plot_rpn_expression_tree("μ f * ν f * f * f f f * * - + f - 2 ∂^2f/∂r^2 * - ∂^4f/∂r^4 - 2 ∂^3f/∂r^3 * ∂^2f/∂r^2 r / + (∂f/∂r) r r * / - (∂^3f/∂θ^2∂r) r r * / 2 ∂^2f/∂r^2 * r r * r * / - 2 ∂f/∂r * + + r / - 2 ∂^4f/∂θ^2∂r^2 * ∂^3f/∂θ^2∂r r / + (∂^4f/∂θ^4) r r * / + 2 ∂^2f/∂r^2 * - 2 ∂^2f/∂θ^2 * + r r * / - 2 r r * r * / ∂f/∂r 2 ∂^3f/∂θ^2∂r * - 3 r / ∂^2f/∂θ^2 * + * -".split(), save = save, title = r"Swift-Hohenberg 2D Polar Coordinates", tolatex = True, to_pdf = True, filename = "SwiftHohenberg2DPolarCoordinates.pdf")
#        plot_rpn_expression_tree("x30 x24 s * * x30 tau + / 1 1 f ~ exp - / 1 1 1 f ~ exp - / - * * x28 ∂f/∂(x100) * x29 ∂f/∂(x101) * + *".split(), save = save, title = "", tolatex = True, to_pdf = True, filename = "Example.pdf")
#        plot_rpn_expression_tree("0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x30 + + + + 0 0 + 0 0 + + 0 0 + 0 x24 + + + 0 0 + 0 0 + + 0 0 + 0 s + + + * * 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x30 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 tau + + + + + / 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 1 + + + + 0 0 + 0 0 + + 0 0 + 0 1 + + + 0 f + ~ exp - / 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 1 + + + + 0 0 + 0 0 + + 0 0 + 0 1 + + + 0 0 + 0 1 + + f ~ exp - / - * * 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x28 + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 ∂f/∂(x100) + + + + + * 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x29 + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 ∂f/∂(x101) + + + + + * + *".split(), save = save, title = "", tolatex = True, to_pdf = True, filename = "Example.pdf", include_expression_in_title = False)
#        test_expr = "+ + - - 9736 x22 / x7 x20 / -100.051731 ^ x5 x15 * + x20 * 1075.000000 x5 -17064.107062"
#        print(f"test_expr = {test_expr}")
#        test_expr = ' '.join(complete_tree(test_expr.split(), 'prefix'))
#        print(f"Completed test_expr = {test_expr}")
#        plot_pn_expression_tree("* * / * + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x30 * + + + 0 0 + 0 0 + + 0 0 + 0 x24 + + + 0 0 + 0 0 + + 0 0 + 0 s + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x30 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 tau * / + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 1 - + + + 0 0 + 0 0 + + 0 0 + 0 1 exp ~ + 0 f - + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 1 / + + + 0 0 + 0 0 + + 0 0 + 0 1 - + + 0 0 + 0 1 exp ~ f + * + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x28 + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 ∂f/∂(x100) * + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 x29 + + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + + 0 0 + 0 0 + + 0 0 + 0 0 + + + 0 0 + 0 0 + + 0 0 + 0 ∂f/∂(x101)".split(), save = save, include_expression_in_title = False)
#        print(complete_tree("".split(), "prefix"))

        complete_rpn_expr = complete_tree("0 0 0 + + 0 x71 + + 0 0 + 0 x69 + + * 0 0 + 0 0.823570 + + 0 335.000000 + 0.018720 x31 + / ^ + 0 0.496312 + x67 -0.6901253057600001 * + 0 0 + 0 6.2216003257007526e-24 + + + 0 -1.278025 + 0 x14 + / 0 0 + 0 -2431.4844 + + + + + 0 0 + 0 2838.0127038380915 + + 0 -0.286767 + 0 x76 + + - 0 0 + 0 0 + + 0 0 + 0 144.49968 + + + + 0 0 + 0 x18 + + 0 0 + 0 2.063746 + + + 0 0 + 0 x100 + + 12187.586 x28 ^ x101 -14640.288932156 + + - * - /".split(), "postfix")
        print(f"complete_rpn_expr = \n{' '.join(complete_rpn_expr)}")
        plot_rpn_expression_tree(complete_rpn_expr, save = save, include_expression_in_title = False)

#    do
#        operand -> operand 0 + -> depth:getDepth(expr)
#    while depth == original_depth
#        print(c)
#        plot_rpn_expression_tree(complete_tree("9736 1.000000 x13 / + x22 x22 * x6 cos ^ - 8.851731000000001 x6 / 4.372938 x7 + 279.200012 - ^ - -3.225653 1075 x10 ^ - x5 x18 2 - ^ / + x24 58.000000 * x0 x2 - ^ 20200101.000000 x15 ^ x20 + + x22 x18 + sqrt 64 x23 - ^ + x19 0.006210 + 25.4 * -820.627062 + x23 -330 + x22 * x21 acos -18327.436844999997 + + + * +".split(), "postfix"), save = save, tolatex = True, to_pdf = True, filename = "Example.pdf", include_expression_in_title = False)
#                               0.148475282221305*x1 - 1.00001327588926*sin(x0)*sin(x1) - 0.0922858190550785
if __name__ == "__main__":
    test_visualize()


#.35 * sin(x1) * sech( (x0 - 5.0)/3.0 ) * sin(1.00*x0)




