import matplotlib.pyplot as plt
import pydot
from collections import deque
from numpy.random import choice
from time import time
from matplotlib.animation import FuncAnimation
from symreg import *

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

called = False
implot = None
def plot_rpn_expression_tree(expression, block = False):
    global called, implot
    def is_operator(token):
        return token in {'+', '-', '*', 'cos', 'grad', 'exp'}
    def is_unary_operator(token):
        return token in {'cos', 'grad', 'exp'}
    
    #https://stackoverflow.com/a/77128902/18255427
    def getRPNdepth(expression):
        stack = []
        for token in expression.split():
            if is_unary_operator(token):  # all unary operators
                stack[-1] += 1
            elif is_operator(token):  # all binary operators
                stack.append(max(stack.pop(), stack.pop()) + 1)
            else:  # an operand (x)
                stack.append(1)
        return stack.pop()

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

    def plot_tree(node, graph, parent=None, depth = 0):
        if node:
            current_node = pydot.Node(str(node.unique_id), label=str(node.value))
            graph.add_node(current_node)

            if parent:
                edge = pydot.Edge(str(parent.unique_id), str(node.unique_id))
                graph.add_edge(edge)

            if isinstance(node, BinaryNode):
                plot_tree(node.left, graph, node, depth+1)
                plot_tree(node.right, graph, node, depth+1)
            elif isinstance(node, UnaryNode):
                plot_tree(node.child, graph, node, depth+1)

    expression_tokens = expression.split()
    expression_tree = build_tree(expression_tokens)

    graph = pydot.Dot(graph_type='graph')
    depth = plot_tree(expression_tree, graph)
    
    graph.write_png('expression_tree.png')
    if called == False or block == True:
        implot = plt.imshow(plt.imread('expression_tree.png'))
        called = True
    else:
        implot.set_data(plt.imread('expression_tree.png'))
    plt.axis('off')
    plt.title(f"{expression}, depth = {getRPNdepth(expression)}")
    plt.show(block = block)
    plt.pause(0.01)

# Example usage:
def test_visualize():
#    # Example usage:
    operators = ['+', '-', '*', 'exp', 'cos', 'grad']

    while True:
        try:
            print(expression:=generate_random_rpn_expression(operators, max_depth=3))
            plot_rpn_expression_tree("x x * c *", block=False)
            
        except KeyboardInterrupt:
            plt.close()
            exit()

if __name__ == "__main__":
    test_visualize()
