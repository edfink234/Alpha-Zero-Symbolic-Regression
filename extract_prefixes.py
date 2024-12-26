
import re

def extract_prefixes(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regex pattern to match lines with 'prefix = {...}'
    prefix_pattern = r'prefix\s*=\s*\{[^}]+\};'
    
    # Find all matches of the pattern in the file content
    matches = re.findall(prefix_pattern, content)
    
    # Print each match line by line
    for match in matches:
        print(match.replace("prefix", "test_expr"))
        print('''printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");
''')

# Replace 'your_file.cpp' with the actual path to your C++ file
extract_prefixes('PrefixDifferentiationSymbolic.cpp')

test_expr = {"+","x","x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+","-","x","x","x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+","-","-","x","x","x","y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+","cos","/","*","y","y","x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "cos", "*", "*","y","x","y","y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+","*","x","x","y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "+", "x", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+","+","cos","x","x","y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "y", "+","cos","x","x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-","y","x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "x", "cos", "cos", "-","y","x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "x", "/", "x", "sin", "-", "y", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "x", "/", "x", "*", "y", "cos", "sin", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "sin", "~", "~", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"sqrt", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "sqrt", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "ln", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "ln", "~", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "ln", "sqrt", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"asin", "*", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"arcsin", "*", "ln", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"arcsin", "*", "ln", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"arcsin", "/", "acos", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "arcsin", "*", "ln", "x", "y", "acos", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"acos", "*", "acos", "acos", "x", "~", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "exp", "x", "exp", "cos", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "exp", "~", "x", "*", "*", "x", "y", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"arccos", "*", "exp", "arcsin", "y", "~", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"^", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "^", "cos", "x", "cos", "y", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "^", "cos", "x", "cos", "y", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "^", "^", "x", "x", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "^", "^", "x", "x", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "^", "tanh", "sech", "x", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "x", "^", "tanh", "/", "x", "y", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"sech", "sin", "sin", "^", "sech", "sin", "x", "*", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"sin", "~", "sech", "/", "arccos", "ln", "x", "*", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "0", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "*", "0", "x", "+", "x", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "~", "*", "0", "x", "tanh", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "1", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "*", "1", "x", "+", "x", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "~", "*", "1", "x", "tanh", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "x", "0"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "*", "x", "0", "+", "x", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "~", "*", "x", "0", "tanh", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "*", "x", "x", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "+", "sin", "x", "x", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "~", "tanh", "*", "x", "1", "*", "1", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "-", "sin", "x", "sin", "x", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "x", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "*", "x", "x", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "*", "x", "cos", "x", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "0", "*", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "0", "*", "x", "cos", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "0", "*", "sin", "x", "sech", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "1", "*", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "1", "cos", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "1", "*", "cos", "x", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "x", "sin", "~", "~", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "tanh", "~", "~", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "^", "0", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "^", "0", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "cos", "x", "^", "0", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "x", "^", "x", "0"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "^", "x", "0", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "cos", "x", "^", "x", "0"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "x", "^", "1", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "^", "1", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "cos", "x", "^", "1", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "x", "^", "x", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "^", "x", "1", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "cos", "x", "^", "x", "1"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"ln", "*", "1", "exp", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"-", "x", "ln", "*", "1", "exp", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"cos", "-", "x", "ln", "*", "1", "exp", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"ln", "exp", "*", "y", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"*", "exp", "*", "x", "x", "ln", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"+", "exp", "exp", "-", "y", "y", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "sin", "*", "x", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"/", "sin", "cos", "x", "y"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"cos", "sqrt", "-", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"sin", "tanh", "sqrt", "-", "*", "x", "x", "*", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"sqrt", "sqrt", "-", "*", "x", "cos", "x", "*", "x", "cos", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"cos", "arcsin", "-", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"sin", "tanh", "asin", "-", "^", "x", "x", "^", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"asin", "arcsin", "-", "*", "x", "sin", "x", "*", "x", "sin", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"exp", "acos", "-", "tanh", "x", "tanh", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"sech", "sech", "arccos", "-", "/", "x", "x", "/", "x", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"acos", "arccos", "-", "-", "x", "sech", "x", "-", "x", "sech", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"acos", "tanh", "-", "*", "x", "exp", "x", "*", "x", "exp", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"asin", "sech", "-", "*", "x", "exp", "x", "*", "x", "exp", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");

test_expr = {"acos", "sech", "-", "-", "x", "sech", "x", "-", "x", "sech", "x"};
printf("before: ");print_container(test_expr);
simplifyPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");
