import matplotlib.pyplot as plt
import pydot
from collections import deque
from numpy.random import choice
from time import time
from matplotlib.animation import FuncAnimation
from symreg import *
import os
from numba import njit, types, typed, jit
from numba.experimental import jitclass
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

class Node:
    def __init__(self, value, unique_id):
        self.value = value
        self.unique_id = unique_id

class BinaryNode(Node):
    def __init__(self, value, unique_id):
        super().__init__(value, unique_id)
        self.left = ""
        self.right = ""

class UnaryNode(Node):
    def __init__(self, value, unique_id):
        super().__init__(value, unique_id)
        self.child = ""

@njit(types.boolean(types.unicode_type), fastmath=True, cache=True)
def is_operator(token):
    return token in {'+', '-', '*', '/', '^', 'cos', 'grad', 'exp'}
@njit(types.boolean(types.unicode_type), fastmath=True, cache=True)
def is_binary_operator(token):
    return token in {'+', '-', '*', '/', '^'}
@njit(types.boolean(types.unicode_type), fastmath=True, cache=True)
def is_unary_operator(token):
    return token in {'cos', 'grad', 'exp'}
#print(f"inspect_types(is_unary_operator) = {is_unary_operator.inspect_types()}")

#https://stackoverflow.com/a/77180279/18255427
#Returns two values, depth and if the expression is complete
@njit(types.Tuple((types.int32, types.boolean))(types.List(types.unicode_type,True)), fastmath=True, cache=True)
def getPNdepth(expression: list[str]) -> tuple[int, bool]:
    if not expression or expression == [" "]: #if it's empty
        return 0, False
    stack = []
    depth, num_binary, num_leaves = 0, 0, 0
    for val in expression:
        if val in {'+', '-', '*', '/', '^'}:#is_binary_operator(val):  # all binary operators
            stack.append(2)  # = number of operands
            num_binary += 1
        elif val in {'cos', 'grad', 'exp'}:#is_unary_operator(val):  # all unary operators
            stack.append(1)
        else:  # an operand (x)
            num_leaves += 1
            while stack and stack[-1] == 1:  # remove fulfilled operators
                stack.pop()
            if stack:  # indicate an operand is consumed
                stack[-1] -= 1
        depth = max(depth, len(stack) + 1)
    return depth-1, num_leaves == num_binary + 1

#print(getPNdepth.inspect_types())
#https://stackoverflow.com/a/77128902/18255427
#Returns two values, depth and if the expression is complete
@njit(types.Tuple((types.int32, types.boolean))(types.List(types.unicode_type,True)), fastmath=True, cache=True)
def getRPNdepth(expression):
    if not expression or expression == [" "]: #if it's empty
        return 0, False
    stack = []
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

    
    expression_tree = build_tree(expression[::-1])

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
        plt.title(f"{' '.join(expression)}, depth = {getPNdepth(expression)[0]}")
        plt.show(block = block)
        plt.pause(0.01)

def plot_rpn_expression_tree(expression, block = False, save = False):
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

    expression_tree = build_tree(expression)

    graph = pydot.Dot(graph_type='graph')
    plot_tree(expression_tree, graph)
    
    if save:
        graph.set('label', f"{expression}, depth = {getRPNdepth(expression)[0]}")
        graph.set('labelloc', 't')  # Set label location to "top"
        graph.write_svg('expression_tree_RPN_Hemberg2008_expr_5.svg')
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
        plt.pause(0.01)

# Example usage:
def test_visualize():
#    # Example usage:
    operators = ['+', '-', '*', 'exp', 'cos', 'grad']
    save = False
    
    if save:
        plot_pn_expression_tree("- + + - + - + / * 30 ^ x 2 * - 10 x y ^ x 4 * / 4 5 ^ x 3 / ^ y 2 2 * 2 y / 8 + + 2 ^ x 2 ^ y 2 / ^ y 3 2 x".split(), block=False, save = save)
        os.system("rsvg-convert -f pdf -o expression_tree_PN_Hemberg2008_expr_5.pdf expression_tree_PN_Hemberg2008_expr_5.svg")
        plot_rpn_expression_tree("30 x 2 ^ * 10 x - y * / x 4 ^ + 4 5 / x 3 ^ * - y 2 ^ 2 / + 2 y * - 8 2 x 2 ^ + y 2 ^ + / + y 3 ^ 2 / + x -".split(), block=False, save = save)
        os.system("rsvg-convert -f pdf -o expression_tree_RPN_Hemberg2008_expr_5.pdf expression_tree_RPN_Hemberg2008_expr_5.svg")

    else:
        while True:
            try:
                plot_pn_expression_tree("+ * 2.583 cos x3 - * x0 x0 0.5".split(), block=False, save = save)
                plot_rpn_expression_tree("2.583 x3 cos * x0 x0 * 0.5 - +".split(), block=False, save = save)

            except KeyboardInterrupt:
                plt.close()
                exit()

if __name__ == "__main__":
    start = time()
        
    test_visualize()
    print(f"Time taken = {time() - start}")

