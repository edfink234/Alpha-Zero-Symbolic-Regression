# https://www.geeksforgeeks.org/infix-to-postfix-using-different-precedence-values-for-in-stack-and-out-stack/
import pandas as pd
import os
from visualize_tree import *
operators = {"(", ")", "cos", "exp", "sqrt", "sin", "asin", "arcsin", "log", "ln", "tanh", "acos", "arccos", "~", "+", "-", "*", "/", "^"}

# To check if the input character
# is an operator or a '('
def isOperator(input):
    return input in (operators - {")"})

# To check if the input character is an operand
def isOperand(input):
    return input != ')' and not isOperator(input)
    
# Function to return precedence value
# if operator is present in stack
def inPrec(input):
    switch = {
        '+': 2,
        '-': 2,
        '*': 4,
        '/': 4,
        '%': 4,
        '^': 6,
        '(': 0,
        'cos': 99,
        'exp': 99,
        'sqrt': 99,
        'sin': 99,
        'asin': 99,
        'arcsin': 99,
        'log': 99,
        'ln': 99,
        'tanh': 99,
        'acos': 99,
        'arccos': 99,
        '~': 99,
    }
    
    return switch.get(input, 0)

# Function to return precedence value
# if operator is present outside stack.
def outPrec(input):
    
    switch = {
        '+': 1,
        '-': 1,
        '*': 3,
        '/': 3,
        '%': 3,
        '^': 6,
        '(': 100,
        'cos': 101,
        'exp': 101,
        'sqrt': 101,
        'sin': 101,
        'asin': 101,
        'arcsin': 101,
        'log': 101,
        'ln': 101,
        'tanh': 101,
        'acos': 101,
        'arccos': 101,
        '~': 101,
    }
    
    return switch.get(input, 0)

# Function to convert infix to postfix
def inToPost(input, end):
    i = 0
    s = []
    string = ""

    # While input is not NULL iterate
    while (len(input) != i):
        # If character an operand
        if isOperand(input[i]):
            string += input[i] + end

        # If input is an operator, push
        elif isOperator(input[i]):
            if (len(s) == 0 or
                outPrec(input[i]) >
                inPrec(s[-1])):
                s.append(input[i])
            
            else:
                while(len(s) > 0 and
                    outPrec(input[i]) <
                    inPrec(s[-1])):
                    string += s.pop() + end
                s.append(input[i])

        # Condition for opening bracket
        elif(input[i] == ')'):
            while(s[-1] != '('):
                string += s.pop() + end
                # If opening bracket not present
                if(len(s) == 0):
                    print('Wrong input')
                    exit(1)
            # pop the opening bracket.
            s.pop()
        i += 1
        
    # pop the remaining operators
    while(len(s) > 0):
        if(s[-1] == '('):
            print('Wrong input')
            exit(1)
        string += s.pop() + end
    
    return string

FeynmanEquations = pd.read_csv("FeynmanEquationsAll.csv")
print(FeynmanEquations.columns)
Formulae = FeynmanEquations['Formula']
Parsed_Postfix_Formulae = []
Parsed_Prefix_Formulae = []
Expression_Depths = []

for Formula in Formulae:
    Parsed_Formula = []
    temp_token = ""
    Formula = Formula.replace("**","^")
    for char in Formula:
        if char in operators:
            if temp_token:
                Parsed_Formula.append(temp_token)
                temp_token = ""
            Parsed_Formula.append(char)
        else:
            temp_token += char
    Parsed_Formula.append(temp_token)
    if ''.join(Parsed_Formula) != Formula:
        print(''.join(Parsed_Formula), Formula)
    
    Parsed_Postfix_Formulae.append(postfix_formula:=inToPost(Parsed_Formula, end=" "))
    Parsed_Prefix_Formulae.append(prefix_formula:=rpn_to_pre(postfix_formula))
    assert(rpn_to_infix(postfix_formula)==pn_to_infix(prefix_formula))
    Expression_Depths.append(getRPNdepth(postfix_formula)[0])
    
FeynmanEquations["PostfixFormula"] = Parsed_Postfix_Formulae
FeynmanEquations["PrefixFormula"] = Parsed_Prefix_Formulae
FeynmanEquations["ExpressionDepth"] = Expression_Depths
print(f"Depths of expressions = {set(FeynmanEquations['ExpressionDepth'])}")

depth = {1, 2, 3, 4, 5, 6, 7, 8}
#equn_nums = FeynmanEquations.loc[FeynmanEquations['ExpressionDepth']==depth]["Eqn. No."].index
#print(equn_nums)
#
#for i in equn_nums:
#    os.system(rf"open -a Google\ Chrome AIFeynmanTrees/Formula{i+1}.pdf")

for depth in set(FeynmanEquations['ExpressionDepth']) - ({1, 2, 3, 4, 5, 6, 7, 8} - {*depth}):
    print(FeynmanEquations.loc[FeynmanEquations['ExpressionDepth']==depth].sort_values(by="# variables", ascending = False)[["PostfixFormula", "ExpressionDepth", "PrefixFormula"]])
    
    print()



if not os.path.isdir("AIFeynmanTrees"):
    os.mkdir("AIFeynmanTrees")

for i, [Formula, PostfixFormula, ExpressionDepth] in enumerate(FeynmanEquations[["Formula", "PostfixFormula", "ExpressionDepth"]].values):
    plot_rpn_expression_tree(expression = PostfixFormula, save = True, filename = f"AIFeynmanTrees/Formula{i+1}.svg", title = f"{Formula}, depth = {ExpressionDepth}")
    os.system(f"rsvg-convert -f pdf -o AIFeynmanTrees/Formula{i+1}.pdf AIFeynmanTrees/Formula{i+1}.svg")

N = 101
#
#plot_rpn_expression_tree(expression = FeynmanEquations["PostfixFormula"][N], save = True, filename = f"AIFeynmanTrees/Formula{N+1}.svg", title = f"{FeynmanEquations['Formula'][N]}, depth = {FeynmanEquations['ExpressionDepth'][N]}")
#os.system(f"rsvg-convert -f pdf -o AIFeynmanTrees/Formula{N+1}.pdf AIFeynmanTrees/Formula{N+1}.svg")

print(FeynmanEquations["PrefixFormula"][N])
print(FeynmanEquations["PostfixFormula"][N])

#FeynmanEquations = FeynmanEquations.sort_values(by="ExpressionDepth")

#os.system(f"rm AIFeynmanTrees/*svg")
