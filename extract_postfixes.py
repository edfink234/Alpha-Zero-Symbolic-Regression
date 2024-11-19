
import re

def extract_postfixes(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regex pattern to match lines with 'postfix = {...}'
    postfix_pattern = r'postfix\s*=\s*\{[^}]+\};'
    
    # Find all matches of the pattern in the file content
    matches = re.findall(postfix_pattern, content)
    
    # Print each match line by line
    for match in matches:
        print(match.replace("postfix", "test_expr"))
        print('''printf("before: ");print_container(test_expr);
simplifyRPN(test_expr);
printf("after: ");print_container(test_expr);
puts("");
''')

# Replace 'your_file.cpp' with the actual path to your C++ file
extract_postfixes('PostfixDifferentiationSymbolic.cpp')
