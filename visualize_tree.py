import matplotlib.pyplot as plt
import pydot
from collections import deque
from numpy.random import choice
from time import time

def generate_random_rpn_expression(operators, max_depth=5):
    stack = []

    for _ in range(max_depth):
        operator = choice(operators)

        if operator in {'cos', 'exp'}:
            if stack:
                operand = stack.pop()#choice(('x','c', stack[-1]), p = (0, 0, 1)) #p = [0.1, 0.1, 0.8])
            else:
                operand = choice(('x','c'))
            stack.append(f'{operand} {operator}')
        elif operator in {'+', '-', '*'}:
            if stack:
                right_operand = stack.pop()#choice(('x','c', stack[-1]), p = (0, 0, 1)) #p = [0.1, 0.1, 0.8])
            else:
                right_operand = choice(('x','c'))
            left_operand = choice(('x','c'))
            stack.append(f'{left_operand} {right_operand} {operator}')

    # print(*stack,len(stack),sep="\n")
    return stack[-1]

class Node:
    def __init__(self, value, unique_id):
        self.value = value
        self.unique_id = unique_id
        self.left = None
        self.right = None

def plot_rpn_expression_tree(expression):
    def is_operator(token):
        return token in {'+', '-', '*', '/', 'cos', 'exp'}

    def build_tree(expression_tokens):
        stack = deque()
        unique_id = 0

        for token in expression_tokens:
            if not is_operator(token):
                unique_id += 1
                node = Node(token, unique_id)
                stack.append(node)
            else:
                right_operand = stack.pop()
                left_operand = stack.pop()
                unique_id += 1
                operator_node = Node(token, unique_id)
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

            plot_tree(node.left, graph, node)
            plot_tree(node.right, graph, node)

    expression_tokens = expression.split()
    expression_tree = build_tree(expression_tokens)

    graph = pydot.Dot(graph_type='graph')
    plot_tree(expression_tree, graph)

    graph.write_png('expression_tree.png')
    plt.imshow(plt.imread('expression_tree.png'))
    plt.axis('off')
    plt.show()

# Example usage:
rpn_expression = "x x + x *"
plot_rpn_expression_tree(rpn_expression)


# Example usage:
operators = ['+', '-', '*']

while time() - start < 10:
    try:
        plot_rpn_expression_tree(x:=generate_random_rpn_expression(operators, max_depth=5))
        print(x)
    except KeyboardInterrupt:
        plt.close()
        exit()
#rpn_expression = "x x exp + cos"
#plot_rpn_expression_tree(rpn_expression)

