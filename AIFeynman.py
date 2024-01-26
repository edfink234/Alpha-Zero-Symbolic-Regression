import pandas as pd
operators = "(", ")", "cos", "exp", "sqrt", "sin", "asin", "arcsin", "log", "tanh", "acos", "arccos", "~", "+", "-", "*", "/", "^"
FeynmanEquations = pd.read_csv("FeynmanEquations.csv")
Formulae = FeynmanEquations['Formula']
Parsed_Formulae = []
for Formula in Formulae:
    Parsed_Formula = ""
    temp_token = ""
    Formula = Formula.replace("**","^")
    for char in Formula:
        if char in operators:
            if temp_token:
                Parsed_Formula.append(temp_token)
                
            Parsed_Formula.append(char)
            
        else:
            temp_token += char
            if temp_token in
