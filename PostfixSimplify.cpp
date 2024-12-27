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

void print_container(const std::vector<std::string>& c) //TODO: Fortran
{
    for (const std::string& i : c)
        std::cout << i << ' ';
    std::cout << '\n';
}

//https://medium.com/@ryan_forrester_/c-check-if-string-is-number-practical-guide-c7ba6db2febf
bool isFloat(const std::string& s) //TODO: Fortran
{
    enum State { START, INT, FRAC, EXP, EXP_NUM };
    State state = START;
    bool has_digits = false;

    for (char c : s)
    {
        switch (state)
        {
            case START:
                if (c == '+' || c == '-') state = INT;
                else if (std::isdigit(c)) { state = INT; has_digits = true; }
                else if (c == '.') state = FRAC;
                else return false;
                break;
            case INT:
                if (std::isdigit(c)) has_digits = true;
                else if (c == '.') state = FRAC;
                else if (c == 'e' || c == 'E') state = EXP;
                else return false;
                break;
            case FRAC:
                if (std::isdigit(c)) has_digits = true;
                else if (c == 'e' || c == 'E') state = EXP;
                else return false;
                break;
            case EXP:
                if (c == '+' || c == '-' || std::isdigit(c)) state = EXP_NUM;
                else return false;
                break;
            case EXP_NUM:
                if (!std::isdigit(c)) return false;
                break;
        }
    }
    return has_digits && (state == INT || state == FRAC || state == EXP_NUM);
}

/*
 Converts:
  - "-0" -> "0"
  - "x.000000" (repeating) -> "x"
  - "-0.0000" (repeating) -. "0"
 */
std::string simplifyString(const std::string& x)
{
    if ((x.size() == 2) && (x[0] == '-') && (x[1] == '0')) //"-0" -> "0"
    {
        return "0";
    }
    unsigned long jdx = x.find(".");
    if (jdx == std::string::npos) //if there's no '.' in x
    {
        return x;
    }
    for (unsigned long i = jdx + 1; i < x.size(); i++) //checking if all the characters after the decimal in x are 0; if not, then return x
    {
        if (x[i] != '0')
        {
            return x;
        }
    }
    std::string temp = x.substr(0, jdx);
    if ((temp.size() == 2) && (temp[0] == '-') && (temp[1] == '0')) //"-0.0000" (repeating) -> "0"
    {
        return "0";
    }
    
    return temp; //"x.0000000" (repeating) -> "x"
}

void simplifyRPN(std::vector<std::string>& expression)
{
    bool simplified = true;
    bool isFloat1, isFloat2;
    while (simplified)
    {
        simplified = false;
        if (expression.size() > 1)
        {
            for (size_t i = 1; i < expression.size(); i++)
            {
                if (is_binary(expression[i]))
                {
                    isFloat1 = isFloat(expression[i-1]);
                    isFloat2 = isFloat(expression[i-2]);
                    
                    if (isFloat1 && isFloat2)
                    {
                        if (expression[i] == "+")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) + std::stof(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "-")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) - std::stof(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "*")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) * std::stof(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "/")
                        {
                            expression[i] = simplifyString(std::to_string(std::stof(expression[i-2]) / std::stof(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "^")
                        {
                            expression[i] = simplifyString(std::to_string(std::powf(std::stof(expression[i-2]), std::stof(expression[i-1]))));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "-")
                    {
                        if ((is_const(expression[i-1]) && is_const(expression[i-2])) && (expression[i-1] == expression[i-2])) //x x - => 0
                        {
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-2] == "0") //"0 x -" -> "x ~"
                        {
                            expression[i] = "~";
                            expression.erase(expression.begin() + i - 2);
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "0") //"x 0 -" -> "x"
                        {
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "*")
                    {
                        if (expression[i-2] == "0" && is_const(expression[i-1])) //"0 x *" -> "0"
                        {
                            puts("hi 131");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "0" && is_const(expression[i-2])) //"x 0 *" -> "0"
                        {
                            puts("hi 139");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-2] == "1" && is_const(expression[i-1])) //"1 x *" -> "x"
                        {
                            puts("hi 147");
                            expression[i] = expression[i-1];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "1" && is_const(expression[i-2])) //"x 1 *" -> "x"
                        {
                            puts("hi 155");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "+")
                    {
                        if (expression[i-2] == "0" && is_const(expression[i-1])) //"0 x +" -> "x"
                        {
                            puts("hi 167");
                            expression[i] = expression[i-1];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "0" && is_const(expression[i-2])) //"x 0 +" -> "x"
                        {
                            puts("hi 175");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "/")
                    {
                        if (expression[i-2] == "0" && is_const(expression[i-1])) // "0 x /" -> "0"
                        {
                            puts("hi 187");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "1" && is_const(expression[i-2])) // "x 1 /" -> "x"
                        {
                            puts("hi 195");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (is_const(expression[i-1]) && is_const(expression[i-2]) && (expression[i-1] == expression[i-2])) // "x x /" -> "1"
                        {
                            puts("hi 203");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "^")
                    {
                        if (expression[i-2] == "0" && is_const(expression[i-1])) // "0 x ^" -> "0"
                        {
                            puts("hi 215");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "0" && is_const(expression[i-2])) // "x 0 ^" -> "1"
                        {
                            puts("hi 223");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-2] == "1" && is_const(expression[i-1])) // "1 x ^" -> "1"
                        {
                            puts("hi 231");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "1" && is_const(expression[i-2])) // "x 1 ^" -> "x"
                        {
                            puts("hi 239");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                }
                
                else if (is_unary(expression[i]) && isFloat(expression[i-1]))
                {
                    if (expression[i] == "cos")
                    {
                        expression[i] = simplifyString(std::to_string(cos(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "~")
                    {
                        expression[i] = simplifyString(std::to_string(-(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sin")
                    {
                        expression[i] = simplifyString(std::to_string(sin(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if ((expression[i] == "ln") || (expression[i] == "log"))
                    {
                        expression[i] = simplifyString(std::to_string(log(std::stof(expression[i-1])))); // Natural log (ln)
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "asin" || expression[i] == "arcsin")
                    {
                        expression[i] = simplifyString(std::to_string(asin(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "acos" || expression[i] == "arccos")
                    {
                        expression[i] = simplifyString(std::to_string(acos(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "exp")
                    {
                        expression[i] = simplifyString(std::to_string(exp(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sech")
                    {
                        expression[i] = simplifyString(std::to_string(1 / cosh(std::stof(expression[i-1])))); // sech(x) = 1 / cosh(x)
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "tanh")
                    {
                        expression[i] = simplifyString(std::to_string(tanh(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sqrt")
                    {
                        expression[i] = simplifyString(std::to_string(sqrt(std::stof(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                }
                
                else if (is_unary(expression[i]))
                {
                    if (expression[i] == "~" && expression[i-1] == "~")
                    {
                        expression[i] = expression[i-2];
                        expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "exp" && (expression[i-1] == "ln" || expression[i-1] == "log"))
                    {
                        puts("hi 360");
                        expression[i] = expression[i-2];
                        expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                        simplified = true;
                        break;
                    }
                    else if (expression[i-1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                    {
                        puts("hi 368");
                        expression[i] = expression[i-2];
                        expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                        simplified = true;
                        break;
                    }
                }
            }
        }
    }
}

int main()
{
    std::vector<std::string> test_expr = {"x1", "x1", "-", "0", "-", "x1", "x1", "+", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"2.33", "1.222", "-", "0", "-", "x1", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "-", "0", "0", "y", "-", "-", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "0", "-", "0", "y", "0", "-", "-", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"3", "0", "-", "0", "4", "0", "-", "-", "+", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x0", "cos", "x0", "x0", "sin", "~", "*", "-", "x0", "cos", "x0", "cos", "*", "/", "x0", "x0", "cos", "/", "sech", "x0", "x0", "cos", "/", "sech", "*", "*", "1", "x0", "x0", "cos", "/", "tanh", "x0", "x0", "cos", "/", "tanh", "*", "-", "sqrt", "/", "~", "x0", "x0", "cos", "/", "tanh", "acos", "sin", "~", "*"}; //TODO: check if equal
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    //========================================================================================================================
    
    test_expr = {"x","x","+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x","x","x","-","+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x","x","-","x","-","y","+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y","y","x","/","*","cos", "y", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y","y","x","*","*","cos", "y", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y","x","x","*","+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y","x","x","+","+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y","x","cos","x","+","+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y","x","cos","x","+","-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y","x","-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "y","x","-", "cos", "cos", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "y", "x", "-", "sin", "/", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "y", "y", "sin", "cos", "*", "/", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "~", "~", "sin", "y", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "sqrt"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "sqrt", "y", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "ln", "y", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "~", "ln", "x", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "sqrt", "ln", "y", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "*", "asin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "ln", "y", "*", "asin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "ln", "y", "*", "asin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "acos", "y", "/", "asin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "ln", "y", "*", "asin", "y", "acos", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "acos", "acos", "x", "~", "*", "acos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "exp", "x", "cos", "exp", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "~", "exp", "x", "x", "y", "*", "*", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y", "arcsin", "exp", "x", "~", "*", "acos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "y", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "cos", "y", "cos", "^", "x", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "cos", "y", "cos", "^", "x", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "^", "x", "^", "y", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "^", "x", "^", "y", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "sech", "tanh", "x", "^", "y", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "y", "/", "tanh", "x", "sin", "^", "x", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "sin", "sech", "x", "y", "*", "^", "sin", "sin", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "ln", "arccos", "x", "y", "*", "/", "sech", "~", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "x", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "x", "*", "x", "x", "sin", "+", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "x", "*", "~", "x", "tanh", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "*", "x", "x", "sin", "+", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "*", "~", "x", "tanh", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "0", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "0", "*", "x", "x", "sin", "+", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "0", "*", "~", "x", "tanh", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "*", "1", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "sin", "+", "1", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "1", "*", "tanh", "~", "1", "*", "1", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "sin", "x", "sin", "-", "x", "sin", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "1", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "*", "1", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "cos", "*", "1", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "x", "x", "*", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "x", "x", "cos", "*", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "x", "sin", "x", "sech", "*", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "x", "*", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "cos", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "cos", "x", "sin", "*", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "~", "~", "sin", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "~", "~", "tanh", "x", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "0", "x", "^", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "x", "^", "x", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "cos", "0", "x", "^", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "0", "^", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "0", "^", "x", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "cos", "x", "0", "^", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "1", "x", "^", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "^", "x", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "cos", "1", "x", "^", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "1", "^", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "1", "^", "x", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "cos", "x", "1", "^", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "exp", "*", "ln"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "1", "x", "exp", "*", "ln", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "1", "x", "exp", "*", "ln", "-", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y", "y", "*", "exp", "ln"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "*", "exp", "y", "ln", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"y", "y", "-", "exp", "exp", "x", "sin", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "*", "sin", "y", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "cos", "sin", "y", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "-", "sqrt", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "*", "x", "x", "*", "-", "sqrt", "tanh", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "cos", "*", "x", "x", "cos", "*", "-", "sqrt", "sqrt"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "-", "arcsin", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "sin", "*", "x", "x", "sin", "*", "-", "arcsin", "asin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "tanh", "x", "tanh", "-", "acos", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "/", "x", "x", "/", "-", "arccos", "sech", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "sech", "-", "x", "x", "sech", "-", "-", "arccos", "acos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "exp", "*", "x", "x", "exp", "*", "-", "tanh", "acos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "exp", "*", "x", "x", "exp", "*", "-", "sech", "asin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x", "x", "sech", "-", "x", "x", "sech", "-", "-", "sech", "acos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"x0", "x0", "cos", "/", "tanh", "acos", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    //========================================================================================================================
    
    test_expr = {"0", "x", "tanh", "tanh", "*", "~", "x", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"y", "1", "*", "x2", "1", "*", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"0", "y", "+", "0", "x2", "+", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "0", "+", "0", "y", "+", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x3", "0", "+", "0", "y", "+", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x3", "/", "1", "y", "/", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "w", "/", "y", "1", "/", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "w", "/", "y", "1", "/", "/", "exp", "ln", "ln", "exp", "ln", "ln", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
}

//g++ -std=c++20 -o PostfixSimplify PostfixSimplify.cpp
//TODO: Link with PostfixDifferentiationSymbolic.cpp and make sure results are correct!


//https://stackoverflow.com/questions/20153412/simplification-algorithm-for-reverse-polish-notation
//https://dl.acm.org/
//simplification of polish notation expressions articles
