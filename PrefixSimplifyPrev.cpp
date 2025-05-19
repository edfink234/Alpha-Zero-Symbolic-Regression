#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <cmath>
#include <cassert>

const std::unordered_set<std::string> unary_operators = {"cos", "~", "sin", "log", "ln", "asin", "arcsin", "acos", "arccos", "exp", "sech", "tanh", "sqrt"};
const std::unordered_set<std::string> binary_operators = {"+", "-", "*", "/", "^"};
const std::string expression_type = "prefix";
std::vector<int> grasp;

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

//https://medium.com/@ryan_forrester_/c-check-if-string-is-number-practical-guide-c7ba6db2febf
bool isFloat(const std::string& s)
{
    enum State { START, INT, FRAC, EXP, EXP_NUM };
    State state = START;
    bool has_digits = false;
    
    if ((s.rfind("nan", 0) != std::string::npos)
        || (s.rfind("inf", 0) != std::string::npos)
        || (s.rfind("-inf", 0) != std::string::npos))
    {
        return true;
    }

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

bool areExpressionRangesEqual(int start_idx_1, int start_idx_2, int num_steps, const std::vector<std::string>& expression)
{
    int stop_idx_1 = start_idx_1 + num_steps;
    
    for (int i = start_idx_1, j = start_idx_2; i < stop_idx_1; i++, j++)
    {
        if (expression[i] != expression[j])
        {
            return false;
        }
    }
    return true;
}

//Function to compute the LGB or RGB, from https://www.jstor.org/stable/43998756
//(top of pg. 165)
void GB(size_t z, size_t& ind, const std::vector<std::string>& individual)
{
    do
    {
        ind = ((expression_type == "prefix") ? ind+1 : ind-1);
        if (is_unary(individual[ind]))
        {
            GB(1, ind, individual);
        }
        else if (is_binary(individual[ind]))
        {
            GB(2, ind, individual);
        }
        --z;
    } while (z);
}

//Computes the grasp of an arbitrary element pieces[i],
//from https://www.jstor.org/stable/43998756 (bottom of pg. 165)
int GR(size_t i, const std::vector<std::string>& individual)
{
    size_t start = i;
    size_t& ptr_lgb = start;
    if (is_unary(individual[i]))
    {
        GB(1, ptr_lgb, individual);
    }
    else if (is_binary(individual[i]))
    {
        GB(2, ptr_lgb, individual);
    }
    return ((expression_type == "prefix") ? ( ptr_lgb - i) : (i - ptr_lgb));
}

void print_container(const std::vector<std::string>& c, int low, int up)
{
    for (int i = low; i <= up; i++)
        std::cout << c[i] << ' ';
    std::cout << '\n';
}

void setPrefixGR(const std::vector<std::string>& prefix, std::vector<int>& grasp)
{
    grasp.reserve(prefix.size());
    for (size_t k = 0; k < prefix.size(); ++k)
    {
        grasp.push_back(GR(k, prefix));
    }
}

void graspSimplifyPrefixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
{
    if (!setGRvar)
    {
        grasp.clear();
        setPrefixGR(expression, grasp);
    }
//    print_container(expression, low, up);
    if (expression[low] == "+" || expression[low] == "-") // +/- x y
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]);
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true);
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true);
        int second_arg_idx_high = new_expression.size();
        int step;
        
        if (new_expression[first_arg_idx_high] == "0") //+/- x 0 -> x
        {
            //puts("hi 177");
            if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
            {
                new_expression.pop_back();
            }
            else
            {
                new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
            }
            new_expression.erase(new_expression.begin() + op_idx); //remove +/- operator at beginning
        }
        
        else if (new_expression[first_arg_idx_low] == "0")
        {
            if (expression[low] == "+") //+ 0 y -> y
            {
                //puts("hi 176");
                new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + first_arg_idx_high); //remove '+' and '0'
            }
            else //- 0 y -> ~ y
            {
                //puts("hi 184");
                new_expression[op_idx] = "~";
                new_expression.erase(new_expression.begin() + first_arg_idx_low); //'0'
            }
        }
        
        else if ((expression[low] == "-") && ((step = (second_arg_idx_high - first_arg_idx_high)) == (first_arg_idx_high - first_arg_idx_low)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) //- x x
        {
            //puts("hi 221");
            assert(new_expression[op_idx] == expression[low]);
            new_expression[op_idx] = "0"; //change "-" to "0";
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
        }
    }
    else if (expression[low] == "*") //* x y
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); //*
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); //* x
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); //* x y
        int second_arg_idx_high = new_expression.size();
        int step;
        if (new_expression[first_arg_idx_high] == "0") //* x 0 -> 0 (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to * x 0, which is 0)
        {
            //puts("hi 239");
            new_expression[op_idx] = "0"; //change '*' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "0") //* 0 x -> 0
        {
            //puts("hi 245");
            new_expression[op_idx] = "0"; //change '*' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_high] == "1") //* x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to * x 1, which is 1)
        {
            //puts("hi 251");
            //erase the '1' at the end
            if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
            {
                new_expression.pop_back();
            }
            else
            {
                new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
            }
            new_expression.erase(new_expression.begin() + op_idx); //erase the '*'
        }
        else if (new_expression[first_arg_idx_low] == "1") //* 1 x -> x
        {
            //puts("hi 265");
            new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + op_idx + 2); //erase the '*' and the '1'
        }
    }
    else if (expression[low] == "/") // / x y
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // /
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // / x
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // / x y
        int second_arg_idx_high = new_expression.size();
        int step;
        if (new_expression[first_arg_idx_high] == "0") // / x 0 -> inf (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to / x 0, which is 0)
        {
            //puts("hi 282");
            new_expression[op_idx] = (new_expression[first_arg_idx_low] != "~") ? "inf": "-inf"; //change '/' to 'inf' or '-inf'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "0") // / 0 x -> 0
        {
            //puts("hi 295");
            new_expression[op_idx] = "0"; //change '/' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_high] == "1") // / x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to / x 1, which is 1)
        {
            //puts("hi 301");
            //erase the '1' at the end
            if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
            {
                new_expression.pop_back();
            }
            else
            {
                new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
            }
            new_expression.erase(new_expression.begin() + op_idx); //erase the '/'
        }
        else if ((expression[low] == "/") && ((step = (second_arg_idx_high - first_arg_idx_high)) == (first_arg_idx_high - first_arg_idx_low)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) // / x x
        {
            //puts("hi 315");
            assert(new_expression[op_idx] == expression[low]);
            new_expression[op_idx] = "1"; //change "-" to "1";
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
        }
    }
    else if (expression[low] == "^") // ^ x y
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // /
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // / x
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // / x y
        //int second_arg_idx_high = new_expression.size();
        int step;
        if (new_expression[first_arg_idx_high] == "0") //^ x 0 -> 1 (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to ^ x 0, which is 1)
        {
            //puts("hi 334");
            new_expression[op_idx] = "1"; //change '^' to '1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "0") // ^ 0 x -> 0 (x > 0 assumed)
        {
            //puts("hi 340");
            new_expression[op_idx] = "0"; //change '^' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_high] == "1") // ^ x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to ^ x 1, which is 1)
        {
            //puts("hi 346");
            //erase the '1' at the end
            if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
            {
                new_expression.pop_back();
            }
            else
            {
                new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
            }
            new_expression.erase(new_expression.begin() + op_idx); //erase the '^'
        }
        else if (new_expression[first_arg_idx_low] == "1") // ^ 1 x -> 1
        {
            //puts("hi 360");
            new_expression[op_idx] = "1"; //change '^' to '1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else if (expression[low] == "cos") // cos x
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // cos
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // cos x
        if (new_expression[first_arg_idx_low] == "0") // cos 0 -> 1
        {
            //puts("hi 374");
            new_expression[op_idx] = "1"; //change 'cos' to '1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else if (expression[low] == "sin") // sin x
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // sin
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sin x
        if (new_expression[first_arg_idx_low] == "0") // sin 0 -> 0
        {
            //puts("hi 388");
            new_expression[op_idx] = "0"; //change 'sin' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else if (expression[low] == "tanh") // tanh x
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // tanh
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // tanh x
        if (new_expression[first_arg_idx_low] == "0") // tanh 0 -> 0
        {
            puts("hi 402");
            new_expression[op_idx] = "0"; //change 'tanh' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else
    {
        for (int i = low; i <= up; i++)
        {
            //assert(i < expression.size() && i >= 0);
            new_expression.push_back(expression[i]);
        }
    }
}

void graspSimplifyPrefix(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp)
{
    std::vector<std::string> new_expression;
    new_expression.reserve(expression.size());
    graspSimplifyPrefixHelper(expression, low, up, grasp, new_expression, false);
    expression = new_expression;
}

void simplifyPN_Helper(std::vector<std::string>& expression)
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
                        else if (expression[i+2] == "0" && is_const(expression[i+1])) //- x 0 -> x
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
                            //puts("hi 131");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0" && is_const(expression[i+1])) //* x 0 -> 0
                        {
                            //puts("hi 139");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "1" && is_const(expression[i+2])) //* 1 x -> x
                        {
                            //puts("hi 147");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && is_const(expression[i+1])) //* x 1 -> x
                        {
                            //puts("hi 155");
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
                            //puts("hi 167");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0" && is_const(expression[i+1])) //+ x 0 -> x
                        {
                            //puts("hi 175");
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
                            //puts("hi 187");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && is_const(expression[i+1])) // / x 1 -> x
                        {
                            //puts("hi 195");
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (is_const(expression[i+1]) && is_const(expression[i+2]) && (expression[i+1] == expression[i+2])) // / x x -> 1
                        {
                            //puts("hi 203");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    else if (expression[i] == "^")
                    {
                        if (expression[i+2] == "0" && is_const(expression[i+1])) // ^ x 0 -> 1
                        {
                            //puts("hi 223");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "0" && is_const(expression[i+2])) // ^ 0 x -> 0 (x > 0)
                        {
                            //puts("hi 215");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "1" && is_const(expression[i+2])) // ^ 1 x -> 1
                        {
                            //puts("hi 231");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && is_const(expression[i+1])) // ^ x 1 -> x
                        {
                            //puts("hi 239");
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
                        expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "exp" && (expression[i+1] == "ln" || expression[i+1] == "log"))
                    {
                        //puts("hi 361");
                        expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                        simplified = true;
                        break;
                    }
                    else if (expression[i+1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                    {
                        //puts("hi 369");
                        expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "cos" && (expression[i+1] == "acos" || expression[i+1] == "arccos"))
                    {
                        //puts("hi 403");
                        expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                        simplified = true;
                        break;
                    }
                    else if ((expression[i] == "cos") && (expression[i+1] == "~")) //cos(-x) = cos(x)
                    {
                        //puts("hi 708");
                        expression.erase(expression.begin() + i + 1); // Remove the '~'
                        simplified = true;
                        break;
                    }
                    else if (expression[i+1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                    {
                        //puts("hi 411");
                        expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sin" && (expression[i+1] == "asin" || expression[i+1] == "arcsin"))
                    {
                        //puts("hi 419");
                        expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                        simplified = true;
                        break;
                    }
                    else if (expression[i+1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                    {
                        //puts("hi 427");
                        expression.erase(expression.begin() + i, expression.begin() + i + 2); // Remove elements at i and i + 1
                        simplified = true;
                        break;
                    }
                }
            }
        }
    }
}

void simplifyPN(std::vector<std::string>& expression)
{
    simplifyPN_Helper(expression);
    graspSimplifyPrefix(expression, 0, expression.size() - 1, grasp);
    simplifyPN_Helper(expression);
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
    
    test_expr = {"+", "+", "-", "*", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "*", "ln", "exp", "log", "20.000000", "/", "-", "-", "~", "0", "exp", "x0", "*", "x1", "~", "0", "*", "-", "~", "0", "exp", "x0", "-", "~", "0", "exp", "x0", "*", "-0.214359", "*", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "*", "ln", "exp", "log", "20.000000", "/", "~", "*", "x1", "-", "~", "0", "exp", "x0", "*", "-", "~", "0", "exp", "x0", "-", "~", "0", "exp", "x0", "/", "*", "0.001370", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "+", "1.244282", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "*", "*", "1.238819", "^", "exp", "log", "20.000000", "/", "x1", "-", "~", "0", "exp", "x0", "sech", "exp", "*", "0.805109", "+", "x0", "x1"};
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
    
    test_expr = {"cos", "acos", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"ln", "exp", "*", "cos", "arccos", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"arccos", "cos", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"ln", "exp", "*", "arccos", "cos", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"sin", "arcsin", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"ln", "exp", "*", "sin", "asin", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"arcsin", "sin", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"ln", "exp", "*", "asin", "sin", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    
    test_expr = {"+", "0", "+", "x", "+", "0", "+", "x", "+", "x", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"-", "0", "+", "x", "-", "0", "+", "x", "+", "x", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"+", "tanh", "cos", "x", "^", "0", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"-", "tanh", "cos", "x", "tanh", "cos", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "tanh", "cos", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "+", "tanh", "cos", "x", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "0", "tanh", "tanh", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"+", "0", "*", "0", "cos", "tanh", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "1", "*", "x", "+", "x", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"+", "0", "*", "1", "+", "tanh", "x", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "tanh", "cos", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sin", "arcsin", "/", "~", "*", "y", "y", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "sin", "arcsin", "/", "~", "*", "y", "y", "x", "sin", "arcsin", "/", "~", "*", "y", "y", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"+", "*", "x", "x", "/", "sin", "arcsin", "/", "~", "*", "y", "y", "x", "sin", "arcsin", "/", "~", "*", "y", "y", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"^", "sin", "arcsin", "/", "~", "*", "y", "y", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"^", "*", "1", "+", "tanh", "x", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"^", "0", "*", "tanh", "cos", "x", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sin", "arcsin", "^", "0", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"^", "sin", "arcsin", "^", "x", "*", "y", "y", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"^", "+", "x", "-", "0", "+", "x", "+", "x", "x", "1"}; //(x + (0 - (x+x+x))) ^ 1 = (-2x)
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr); //+ x ~ + x + x x = -(x+x+x)+x = (-2x)
    puts("");
    
    test_expr = {"^", "1", "+", "*", "x", "x", "/", "sin", "arcsin", "/", "~", "*", "y", "y", "x", "sin", "arcsin", "/", "~", "*", "y", "y", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"+", "^", "1", "sin", "arcsin", "^", "x", "*", "y", "y", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"cos", "sin", "arcsin", "^", "0", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"cos", "^", "0", "*", "tanh", "cos", "x", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"cos", "~", "^", "0", "*", "tanh", "cos", "x", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sin", "cos", "~", "arcsin", "^", "0", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sin", "sin", "arcsin", "^", "0", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sin", "^", "sin", "sin", "arcsin", "^", "0", "*", "y", "y", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "sin", "sin", "arcsin", "^", "0", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "^", "sin", "sin", "arcsin", "^", "0", "*", "y", "y", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
}

//g++ -std=c++20 -o PrefixSimplifyPrev PrefixSimplifyPrev.cpp

//https://stackoverflow.com/questions/20153412/simplification-algorithm-for-reverse-polish-notation
//https://dl.acm.org/
//simplification of polish notation expressions articles
// ! objdump -d -M intel PrefixSimplify
