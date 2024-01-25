#A script for visualizing prefix and postfix expression trees
import matplotlib.pyplot as plt
import pydot
from collections import deque
from numpy.random import choice
from time import time
from matplotlib.animation import FuncAnimation
import os

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
    return token in {'+', '-', '*', '/', '^', 'cos', 'grad', 'exp', '-sin', 'sin'}
def is_binary_operator(token):
    return token in {'+', '-', '*', '/', '^'}
def is_unary_operator(token):
    return token in {'cos', 'grad', 'exp', '-sin', 'sin'}


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
        plt.pause(0.01)

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
        plt.pause(0.01)

# Example usage:
def test_visualize():
#    # Example usage:
    save = False
    
    if save:
#        plot_pn_expression_tree("- + + - + - + / * 30 ^ x 2 * - 10 x y ^ x 4 * / 4 5 ^ x 3 / ^ y 2 2 * 2 y / 8 + + 2 ^ x 2 ^ y 2 / ^ y 3 2 x", block=False, save = save)
#        os.system("rsvg-convert -f pdf -o expression_tree_PN_Hemberg2008_expr_5.pdf expression_tree_PN_Hemberg2008_expr_5.svg")
        
        plot_rpn_expression_tree("8 2 x 2 ^ + y 2 ^ + /", block=False, save = save, filename = "Hemberg2008Expressions/expression_tree_Hemberg2008_expr_1.svg", title = r"(8 / ((2 + (x ^ 2)) + (y ^ 2)))")
        os.system("rsvg-convert -f pdf -o Hemberg2008Expressions/expression_tree_Hemberg2008_expr_1.pdf Hemberg2008Expressions/expression_tree_Hemberg2008_expr_1.svg")
        plot_rpn_expression_tree("x 3 ^ x 1 + * y y 2 / 1 - * +", block=False, save = save, filename = "Hemberg2008Expressions/expression_tree_Hemberg2008_expr_2.svg", title = r"(((x ^ 3) * (x + 1)) + (y * ((y / 2) - 1)))")
        os.system("rsvg-convert -f pdf -o Hemberg2008Expressions/expression_tree_Hemberg2008_expr_2.pdf Hemberg2008Expressions/expression_tree_Hemberg2008_expr_2.svg")
        plot_rpn_expression_tree("x 3 ^ 5 / y 3 ^ 2 / + y - x -", block=False, save = save, filename = "Hemberg2008Expressions/expression_tree_Hemberg2008_expr_3.svg", title = r"(((((x ^ 3) / 5) + ((y ^ 3) / 2)) - y) - x)")
        os.system("rsvg-convert -f pdf -o Hemberg2008Expressions/expression_tree_Hemberg2008_expr_3.pdf Hemberg2008Expressions/expression_tree_Hemberg2008_expr_3.svg")
        plot_rpn_expression_tree("30 x 2 ^ * 10 x - y * / x 4 ^ + x 3 ^ - y 2 ^ 2 / + y - 8 2 x 2 ^ + y 2 ^ + / + x +", block=False, save = save, filename = "Hemberg2008Expressions/expression_tree_Hemberg2008_expr_4.svg", title = r"((((((((30 * (x ^ 2)) / ((10 - x) * y)) + (x ^ 4)) - (x ^ 3)) + ((y ^ 2) / 2)) - y) + (8 / ((2 + (x ^ 2)) + (y ^ 2)))) + x)")
        os.system("rsvg-convert -f pdf -o Hemberg2008Expressions/expression_tree_Hemberg2008_expr_4.pdf Hemberg2008Expressions/expression_tree_Hemberg2008_expr_4.svg")
        plot_rpn_expression_tree("30 x 2 ^ * 10 x - y * / x 4 ^ + 4 5 / x 3 ^ * - y 2 ^ 2 / + 2 y * - 8 2 x 2 ^ + y 2 ^ + / + y 3 ^ 2 / + x -", block=False, save = save, filename = "Hemberg2008Expressions/expression_tree_Hemberg2008_expr_5.svg", title = r"(((((((((30 * (x ^ 2)) / ((10 - x) * y)) + (x ^ 4)) - ((4 / 5) * (x ^ 3))) + ((y ^ 2) / 2)) - (2 * y)) + (8 / ((2 + (x ^ 2)) + (y ^ 2)))) + ((y ^ 3) / 2)) - x)")
        os.system("rsvg-convert -f pdf -o Hemberg2008Expressions/expression_tree_Hemberg2008_expr_5.pdf Hemberg2008Expressions/expression_tree_Hemberg2008_expr_5.svg")
        
        rpn_1 = rpn_to_infix(r_1:="8 2 x 2 ^ + y 2 ^ + /")
        rpn_2 = rpn_to_infix(r_2:="x 3 ^ x 1 + * y y 2 / 1 - * +")
        rpn_3 = rpn_to_infix(r_3:="x 3 ^ 5 / y 3 ^ 2 / + y - x -")
        rpn_4 = rpn_to_infix(r_4:="30 x 2 ^ * 10 x - y * / x 4 ^ + x 3 ^ - y 2 ^ 2 / + y - 8 2 x 2 ^ + y 2 ^ + / + x +")
        rpn_5 = rpn_to_infix(r_5:="30 x 2 ^ * 10 x - y * / x 4 ^ + 4 5 / x 3 ^ * - y 2 ^ 2 / + 2 y * - 8 2 x 2 ^ + y 2 ^ + / + y 3 ^ 2 / + x -")
        
        pn_1 = pn_to_infix(p_1:="/ 8 + + 2 ^ x 2 ^ y 2")
        pn_2 = pn_to_infix(p_2:="+ * ^ x 3 + x 1 * y - / y 2 1")
        pn_3 = pn_to_infix(p_3:=" - - + / ^ x 3 5 / ^ y 3 2 y x")
        pn_4 = pn_to_infix(p_4:="+ + - + - + / * 30 ^ x 2 * - 10 x y ^ x 4 ^ x 3 / ^ y 2 2 y / 8 + + 2 ^ x 2 ^ y 2 x")
        pn_5 = pn_to_infix(p_5:="- + + - + - + / * 30 ^ x 2 * - 10 x y ^ x 4 * / 4 5 ^ x 3 / ^ y 2 2 * 2 y / 8 + + 2 ^ x 2 ^ y 2 / ^ y 3 2 x")
        
        print(rpn_1==pn_1, rpn_2==pn_2, rpn_3==pn_3, rpn_4==pn_4, rpn_5==pn_5, sep='\n')
        
        
        print(rpn_1, rpn_2, rpn_3, rpn_4, rpn_5, '\n', sep='\n')
        print(r_1, r_2, r_3, r_4, r_5, '\n', p_1, p_2, p_3, p_4, p_5, sep='\n')
        
        
        

    else:
        print(pn_to_infix(" - - + / ^ x 3 5 / ^ y 3 2 y x".split()))
#        print(rpn_to_infix("y y x * * cos y +"))
        while True:
            try:
#                plot_pn_expression_tree("+ cos cos x0 * 1.031240 + 0.008202 * 1.919085 - cos x3 - cos x0 cos cos * x0 + x3 x3".split(), block=False, save = save)
#                plot_rpn_expression_tree("x3 cos 0.427738 * 4.779139 x1 - 0.390789 x0 0.637794 x2 * - + 0.598703 x2 cos 1.463665 cos x2 + 1.063828 x3 + x0 0.031570 x0 + 1.493230 - * * + - * * + *".split(), block=False, save = save)
                plot_rpn_expression_tree("x 2 x * + cos 8 9 cos * / ".split(), block=False, save = save)


            except KeyboardInterrupt:
                plt.close()
                exit()

if __name__ == "__main__":
    test_visualize()
