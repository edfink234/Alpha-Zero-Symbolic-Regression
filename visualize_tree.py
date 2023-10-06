import matplotlib.pyplot as plt
import pydot
from collections import deque
from numpy.random import choice
from time import time
from matplotlib.animation import FuncAnimation
from symreg import *
import os

def generate_random_rpn_expression(operators, max_depth=5):
    stack = []

    for _ in range(max_depth):
        operator = choice(operators)

        if operator in {'cos', 'exp', 'grad'}:
            if stack:
                operand = choice(('x','c', stack[-1]), p = (0, 0, 1)) #p = [0.1, 0.1, 0.8])
            else:
                operand = choice(('x','c'))
            stack.append(f'{operand} {operator}')
        elif operator in {'+', '-', '*'}:
            if stack:
                right_operand = choice(('x','c', stack[-1]), p = (0, 0, 1)) #p = [0.1, 0.1, 0.8])
            else:
                right_operand = choice(('x','c'))
            left_operand = choice(('x','c'))
            stack.append(f'{left_operand} {right_operand} {operator}')

    return stack[-1]

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
    return token in {'+', '-', '*', '/', '^', 'cos', 'grad', 'exp'}
def is_binary_operator(token):
    return token in {'+', '-', '*', '/', '^'}
def is_unary_operator(token):
    return token in {'cos', 'grad', 'exp'}

#https://stackoverflow.com/a/77180279/18255427
#Returns two values, depth and if the expression is complete
def getPNdepth(expression):

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


called = False
implot = None
def plot_pn_expression_tree(expression, block = False, save = False):
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

    
    expression_tokens = expression.split()[::-1] if isinstance(expression, str) else expression[::-1]
    expression_tree = build_tree(expression_tokens)

    graph = pydot.Dot(graph_type='graph')
    plot_tree(expression_tree, graph)
    
    if save:
        graph.set('label', f"{expression}, depth = {getPNdepth(expression)[0]}")
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
        plt.title(f"{expression}, depth = {getPNdepth(expression)[0]}")
        plt.show(block = block)
        plt.pause(0.01)

# Example usage:
def test_visualize():
#    # Example usage:
    operators = ['+', '-', '*', 'exp', 'cos', 'grad']
    save = True
    
    if save:
        plot_pn_expression_tree("- + + - + - + / * 30 ^ x 2 * - 10 x y ^ x 4 * / 4 5 ^ x 3 / ^ y 2 2 * 2 y / 8 + + 2 ^ x 2 ^ y 2 / ^ y 3 2 x", block=False, save = save)
        os.system("rsvg-convert -f pdf -o expression_tree_PN_Hemberg2008_expr_5.pdf expression_tree_PN_Hemberg2008_expr_5.svg")


    else:
        while True:
            try:
    #            print(expression:=generate_random_pn_expression(operators, max_depth=3))
#                plot_pn_expression_tree(' '.join(['+', 'cos', '+', 'x1', 'x2', '+', '+', 'x1', 'x2', 'cos', 'x3'], block=False, save = save)
#                plot_pn_expression_tree(' '.join(['+', 'cos', '+', 'x1', 'x2', '+', 'x1', 'x2']), block=False, save = save)
                plot_pn_expression_tree(' '.join(['+', '*', '2.5382', 'cos', 'x3', '-', '*', 'x0', 'x0', '0.5']), block=False, save = save)
#                print(getPNdepth("+")[0] <= 3)
#                print(getPNdepth("+ +")[0] <= 3)
#                print(getPNdepth("+ + const")[0] <= 3)
#                print(getPNdepth("+ + const cos")[0] <= 3)
#                print(getPNdepth("+ + const cos x3")[0] <= 3)
#                print(getPNdepth("+ + const cos x3 -")[0] <= 3)
#                print(getPNdepth("+ + const cos x3 - *")[0] <= 3)
#                print(getPNdepth("+ + const cos x3 - * x0")[0] <= 3)
#                print(getPNdepth("+ + const cos x3 - * x0 x0")[0] <= 3)
#                print(getPNdepth("+ + const cos x3 - * x0 x0 0.5")[0] <= 3)
                
            except KeyboardInterrupt:
                plt.close()
                exit()

if __name__ == "__main__":
    test_visualize()
