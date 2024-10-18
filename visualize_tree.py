#A script for visualizing prefix and postfix expression trees
import matplotlib.pyplot as plt
import pydot
from collections import deque
from numpy.random import choice
from time import time
from matplotlib.animation import FuncAnimation
import os
import sys

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
    return token in {"cos", "exp", "sqrt", "sin", "asin", "arcsin", "log", "tanh", "acos", "arccos", "~", "+", "-", "*", "/", "^", "ln", "sech", "conj"}
def is_binary_operator(token):
    return token in {'+', '-', '*', '/', '^'}
def is_unary_operator(token):
    return token in {"cos", "exp", "sqrt", "sin", "asin", "arcsin", "log", "tanh", "acos", "arccos", "~", "ln", "sech", "conj"}


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

called = False
implot = None
def plot_pn_expression_tree(expression: list[str], block = False, save = False):
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
        graph.set('label', f"{' '.join(expression)}, depth = {getPNdepth(expression)[0]}")
        graph.set('labelloc', 't')  # Set label location to "top"
        graph.write_svg('expression_tree_PN_Hemberg2008_expr_5.svg')
    else:
        graph.write_png('expression_tree.png')
        if called == False or block == True:
            implot = plt.imshow(plt.imread('expression_tree.png'))
            called = True
        else:
            implot.set_data(plt.imread('expression_tree.png'))
        plt.axis('off')
        plt.title(f"{' '.join(expression)}, depth = {getPNdepth(expression)[0]}")
        plt.show(block = block)

def plot_rpn_expression_tree(expression: list[str], block = False, save = False, filename = "", title = ""):
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
    else:
        graph.write_png('expression_tree.png')
        if called == False or block == True:
            implot = plt.imshow(plt.imread('expression_tree.png'))
            called = True
        else:
            implot.set_data(plt.imread('expression_tree.png'))
        plt.axis('off')
        plt.title(f"{' '.join(expression)}, depth = {getRPNdepth(expression)[0]}")
        plt.show(block = block)

# Example usage:
def test_visualize():
#    # Example usage:
    save = False
    
    if save:
#        plot_pn_expression_tree("- + + - + - + / * 30 ^ x 2 * - 10 x y ^ x 4 * / 4 5 ^ x 3 / ^ y 2 2 * 2 y / 8 + + 2 ^ x 2 ^ y 2 / ^ y 3 2 x", block=False, save = save)
#        os.system("rsvg-convert -f pdf -o expression_tree_PN_Hemberg2008_expr_5.pdf expression_tree_PN_Hemberg2008_expr_5.svg")
        
        plot_rpn_expression_tree("I 0 sech ^ y 4 I ^ x exp / / y 4 I ^ x exp / / 0.2 t 20.0 - - tanh * - -", block=False, save = save, filename = "/Users/edwardfinkelstein/AIFeynmanExpressionTrees/AE601/Case1BestTree.svg", title = r"((I ^ sech(0)) - ((y / ((4 ^ I) / exp(x))) - ((y / ((4 ^ I) / exp(x))) * tanh((0.2 - (t - 20.000000)))))), depth = 6")
#        os.system("rsvg-convert -f pdf -o /Users/edwardfinkelstein/AIFeynmanExpressionTrees/AE601/Case1BestTree.pdf Case1BestTree.svg")
        plot_rpn_expression_tree("I 0.2 t ^ π log log sech 0.2 0.0 x y 0.1 + / / + + / ^", block=False, save = save, filename = "/Users/edwardfinkelstein/AIFeynmanExpressionTrees/AE601/Case2BestTree.svg", title = r"(I ^ ((0.2 ^ t) / (sech(log(log(π))) + (0.2 + (0.000000 / (x / (y + 0.100000))))))), depth = 7")
#        os.system("rsvg-convert -f pdf -o /Users/edwardfinkelstein/AIFeynmanExpressionTrees/AE601/Case2BestTree.pdf Case2BestTree.svg")
        
    else:
#        print("hi")
#        print(pn_to_infix(" - - + / ^ x 3 5 / ^ y 3 2 y x".split()))
#        print(rpn_to_infix("y y x * * cos y +"))
#        while True:
        try:
#                plot_pn_expression_tree("+ cos cos x * 1.031240 + 0.008202 * 1.919085 - cos I - cos x cos cos * x + I I".split(), block=False, save = save)
#                plot_rpn_expression_tree("I cos 0.427738 * 4.779139 y - 0.390789 x 0.637794 t * - + 0.598703 t cos 1.463665 cos t + 1.063828 I + x 0.031570 x + 1.493230 - * * + - * * + *".split(), block=False, save = save)
#                plot_rpn_expression_tree("q Ef * m omega_0 2 ^ omega 2 ^ - *  /".split(), block=False, save = save)
#                plot_pn_expression_tree("- * ~ sin x 1 * ^ x 1 / 1 x".split(), block=True, save = save)
#            plot_rpn_expression_tree("i u_t * 1 2 / u_xx u_yy + * + u conj u * u * -".split(), block=True, save = save)
#                x_real = "+ x y"
#                x_imag = "* x cos y"
#                plot_pn_expression_tree(("+ " + x_real + " * i " + x_imag).split(), block=True, save = save)
                '''
                depth_real  depth_imag  total_depth
                2           2           4
                3           2           4
                4           2           5
                4           3           5
                4           4           6 
                2           3           5
                2           4           6
                3           3           5
                3           4           6 
                '''
#                x_real = "x y +"
#                x_imag = "x y cos *"
#                plot_rpn_expression_tree((x_real + " i " + x_imag + " * +").split(), block=True, save = save)
#                plot_pn_expression_tree(("/ * - x_0 x0 x3 * sigma sigma").split(), block=True, save = save) #✅
#                plot_pn_expression_tree(("/ * - y_0 x1 x3 * sigma sigma").split(), block=True, save = save) #✅
#                plot_rpn_expression_tree(("x_0 x0 - x3 * sigma sigma * /").split(), block=True, save = save) #✅
#                plot_rpn_expression_tree(("y_0 x1 - x0 x_0 - x_0 x0 - * x1 y_0 - x1 y_0 - * - exp * sigma sigma * /").split(), block=True, save = save) ✅
#                plot_rpn_expression_tree(("y_0 x1 - x3 * sigma sigma * /").split(), block=True, save = save) #✅
                plot_pn_expression_tree(("* - y_0 x1 * 2 x3").split(), block=True, save = save)
                
            
#                plot_rpn_expression_tree("T_t 1 y y * - T_x * + kappa T_{xx} T_{yy} + * -".split(), block=True, save = save)
#                plot_pn_expression_tree(("- + T_t * - 1 * y y T_x * kappa + T_{xx} T_{yy}").split(), block=True, save = save)
#                plot_rpn_expression_tree("T_t 4 y * sin T_x * + 4 x * cos T_y * + kappa T_{xx} T_{yy} + * -".split(), block=True, save = save)
#                plot_pn_expression_tree(("- + + T_t * sin * 4 y T_x * cos * 4 x T_y * kappa + T_{xx} T_{yy}").split(), block=True, save = save)
#                plot_rpn_expression_tree("I I ^ y cos x tanh - + t I I ^ + I I tanh ^ / /".split(), block=True, save = save)


        except KeyboardInterrupt:
            plt.close()
            exit()

if __name__ == "__main__":

    test_visualize()


#-((x - x_0)*(x - x_0)) = -(x^2 - 2*x*x_0 + x_0^2)
#(x - x_0)*(x_0 - x) = x*x_0 - x^2 - x_0^2 + x*x_0 = -(x^2 - 2*x*x_0 + x_0^2)
