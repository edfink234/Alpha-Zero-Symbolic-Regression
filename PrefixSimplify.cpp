#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <cmath>

const std::unordered_set<std::string> unary_operators = {"cos", "~", "sin", "log", "ln", "asin", "arcsin", "acos", "arccos", "exp", "sech", "tanh", "sqrt"};
const std::unordered_set<std::string> binary_operators = {"+", "-", "*", "/", "^"};

bool is_unary(const std::string& token)
{
    return (unary_operators.find(token) != unary_operators.end());
}

bool is_binary(const std::string& token)
{
    return (binary_operators.find(token) != binary_operators.end());
}

bool is_const(const std::string& token)
{
    return ((!is_unary(token)) && (!is_binary(token)));
}

void print_container(const std::vector<std::string>& c)
{
    for (const std::string& i : c)
        std::cout << i << ' ';
    std::cout << '\n';
}

bool isFloat(const std::string& x)
{
//    std::cout << x << '\n';
    try
    {
        std::stof(x);
        return true;
    }
    catch (std::invalid_argument& e)
    {
        return false;
    }
}

std::string simplifyString(const std::string& x)
{
    if ((x.size() == 2) && (x[0] == '-') && (x[1] == '0'))
    {
        return "0";
    }
    unsigned long jdx = x.find(".");
    if (jdx == std::string::npos)
    {
        return x;
    }
    for (unsigned long i = jdx + 1; i < x.size(); i++)
    {
        if (x[i] != '0')
        {
            return x;
        }
    }
    std::string temp = x.substr(0, jdx);
    if ((temp.size() == 2) && (temp[0] == '-') && (temp[1] == '0'))
    {
        return "0";
    }
    
    return temp;
}

void simplifyPN(std::vector<std::string>& expression)
{
    bool simplified = true;
    bool isFloat1, isFloat2;
    while (simplified)
    {
        simplified = false;
        if (expression.size() > 1)
        {
            for (size_t i = 0; i < expression.size() - 1; i++)
            {
                if (is_binary(expression[i]))
                {
                    isFloat1 = isFloat(expression[i+1]);
                    isFloat2 = isFloat(expression[i+2]);
                    
                    if (isFloat1 && isFloat2)
                    {
                        if (expression[i] == "+")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) + std::stof(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "-")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) - std::stof(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "*")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) * std::stof(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "/")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i+1]) / std::stof(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "^")
                        {
                            expression[i] = simplifyString(std::to_string(std::powf(std::stof(expression[i+1]), std::stof(expression[i+2]))));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "-")
                    {
                        if ((is_const(expression[i+1]) && is_const(expression[i+2])) && (expression[i+1] == expression[i+2])) //- x x => 0
                        {
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "0") //- 0 x -> ~ x
                        {
                            expression[i] = "~";
                            expression.erase(expression.begin() + i + 1);
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0") //- x 0 -> x
                        {
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "*")
                    {
                        if (expression[i+1] == "0" && is_const(expression[i+2])) //* 0 x -> 0
                        {
                            puts("hi 131");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0" && is_const(expression[i+1])) //* x 0 -> 0
                        {
                            puts("hi 139");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "1" && is_const(expression[i+2])) //* 1 x -> x
                        {
                            puts("hi 147");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && is_const(expression[i+1])) //* x 1 -> x
                        {
                            puts("hi 155");
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "+")
                    {
                        if (expression[i+1] == "0" && is_const(expression[i+2])) //+ 0 x -> x
                        {
                            puts("hi 167");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0" && is_const(expression[i+1])) //+ x 0 -> x
                        {
                            puts("hi 175");
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "/")
                    {
                        if (expression[i+1] == "0" && is_const(expression[i+2])) // / 0 x -> 0
                        {
                            puts("hi 187");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && is_const(expression[i+1])) // / x 1 -> x
                        {
                            puts("hi 195");
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (is_const(expression[i+1]) && is_const(expression[i+2]) && (expression[i+1] == expression[i+2])) // / x x -> 1
                        {
                            puts("hi 203");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "^")
                    {
                        if (expression[i+1] == "0" && is_const(expression[i+2])) // ^ 0 x -> 0
                        {
                            puts("hi 215");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0" && is_const(expression[i+1])) // ^ x 0 -> 1
                        {
                            puts("hi 223");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "1" && is_const(expression[i+2])) // ^ 1 x -> 1
                        {
                            puts("hi 231");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && is_const(expression[i+1])) // ^ x 1 -> x
                        {
                            puts("hi 239");
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                }
                
                else if (is_unary(expression[i]) && isFloat(expression[i+1]))
                {
                    if (expression[i] == "cos")
                    {
                        expression[i] = simplifyString(std::to_string(cos(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "~")
                    {
                        expression[i] = simplifyString(std::to_string(-(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sin")
                    {
                        expression[i] = simplifyString(std::to_string(sin(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if ((expression[i] == "ln") || (expression[i] == "log"))
                    {
                        expression[i] = simplifyString(std::to_string(log(std::stof(expression[i+1])))); // Natural log (ln)
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "asin" || expression[i] == "arcsin")
                    {
                        expression[i] = simplifyString(std::to_string(asin(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "acos" || expression[i] == "arccos")
                    {
                        expression[i] = simplifyString(std::to_string(acos(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "exp")
                    {
                        expression[i] = simplifyString(std::to_string(exp(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sech")
                    {
                        expression[i] = simplifyString(std::to_string(1 / cosh(std::stof(expression[i+1])))); // sech(x) = 1 / cosh(x)
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "tanh")
                    {
                        expression[i] = simplifyString(std::to_string(tanh(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sqrt")
                    {
                        expression[i] = simplifyString(std::to_string(sqrt(std::stof(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                }
                
                else if (is_unary(expression[i]))
                {
                    if (expression[i] == "~" && expression[i+1] == "~")
                    {
                        expression[i] = expression[i+2];
                        expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "exp" && (expression[i+1] == "ln" || expression[i+1] == "log"))
                    {
                        puts("hi 361");
                        expression[i] = expression[i+2];
                        expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                        simplified = true;
                        break;
                    }
                    //TODO: Need to add ln(exp(x)) here and a corresponding test case:
                }
            }
        }
    }
}

int main()
{
    std::vector<std::string> test_expr = {"-", "-", "-", "x1", "x1", "0", "+", "x1", "x1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    test_expr = {"-", "-", "-", "2.33", "1.222", "0", "x1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    test_expr = {"+", "-", "0", "x", "-", "0", "-", "0", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    test_expr = {"+", "-", "x", "0", "-", "0", "-", "y", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    test_expr = {"cos", "+", "-", "3", "0", "-", "0", "-", "4", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    test_expr = {"+", "+", "-", "*", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "*", "ln", "exp", "log", "20.000000", "/", "-", "-", "~", "0", "exp", "x0", "*", "x1", "~", "0", "*", "-", "~", "0", "exp", "x0", "-", "~", "0", "exp", "x0", "*", "-0.214359", "*", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "*", "ln", "exp", "log", "20.000000", "/", "~", "*", "x1", "-", "~", "0", "exp", "x0", "*", "-", "~", "0", "exp", "x0", "-", "~", "0", "exp", "x0", "/", "*", "0.001370", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "+", "1.244282", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "*", "*", "1.238819", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "sech", "exp", "*", "0.805109", "+", "x0", "x1"}; //TODO: check if equal
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    
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
    
    
    

    test_expr = {"+", "~", "*", "0", "tanh", "tanh", "x", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "*", "y", "1", "*", "x2", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "+", "0", "y", "+", "0", "x2"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "+", "x", "0", "+", "0", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "+", "x3", "0", "+", "0", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "/", "0", "x3", "/", "1", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "/", "0", "w", "/", "y", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
}

//g++ -std=c++20 -o PrefixSimplify PrefixSimplify.cpp
//TODO: Link with PrefixDifferentiationSymbolic.cpp and make sure results are correct!


//https://stackoverflow.com/questions/20153412/simplification-algorithm-for-reverse-polish-notation
//https://dl.acm.org/
//simplification of polish notation expressions articles
