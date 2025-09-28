//TODO: Include `bin_op nan x` = `bin_op x nan` = `un_op nan` = `nan`; needs to be first case to test for each "block"
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
    return ((unary_operators.find(token) != unary_operators.end()) || (token == "abs"));
}

std::string to_string_general(double v)
{
    if (std::isnan(v))
    {
        return "nan";
    }
    if (std::isinf(v))
    {
        return (v < 0 ? "-inf" : "inf");
    }
    thread_local std::string out(32, '\0'); // start with a small buffer
    while (true)
    {
        auto [p, ec] = std::to_chars(out.data(), out.data() + out.size(), v, std::chars_format::general);
        if (ec == std::errc{})
        {
            out.resize(p - out.data());
            return out;
        }
        out.resize(out.size() * 2);            // grow and retry (rare)
    }
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

double Stod(const std::string& param)
{
    try
    {
        double val = std::stod(param);
        return val;
    }
    catch (const std::out_of_range&)
    {
        if (!param.empty() && param[0] == '-')
        {
            return -std::numeric_limits<double>::infinity();
        }
        else
        {
            return std::numeric_limits<double>::infinity();
        }
    }
}

//https://medium.com/@ryan_forrester_/c-check-if-string-is-number-practical-guide-c7ba6db2febf
bool isdouble(const std::string& s)
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
//((1+5) * 1) -> 1 5 1 * +, + 1 * 5 1 -> traverse into a tree data-structure -> apply simplification algorithm
//-> traverse tree again to get prefix/postfix simplified expression -> convert to infix
//prefix/postifx -> simplified prefix/postfix
void graspSimplifyPrefixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
{
    if (!setGRvar)
    {
        grasp.clear();
        setPrefixGR(expression, grasp);
    }
//    print_container(expression, low, up);
//    print_container(new_expression, 0, new_expression.size() - 1);
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
        
        else if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) //+/- nan x -> nan, +/- x nan -> nan, +/- nan nan -> nan
        {
            //puts("hi 273");
            assert(new_expression[op_idx] == expression[low]);
            new_expression[op_idx] = "nan"; //change "+/-" to "nan";
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
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
        //int second_arg_idx_high = new_expression.size();
        //int step;
        if ((new_expression[first_arg_idx_high] == "nan") || (new_expression[first_arg_idx_low] == "nan")) //* nan x -> nan, * x nan -> nan, * nan nan -> nan
        {
            //puts("hi 300");
            new_expression[op_idx] = "nan"; //change '*' to 'nan'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_high] == "0") //* x 0 -> 0 (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to * x 0, which is 0)
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
        
        if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) // / nan x -> nan, / x nan -> nan, / nan nan -> nan
        {
            //puts("hi 350");
            new_expression[op_idx] = "nan"; //change '/' to 'nan'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if ((new_expression[first_arg_idx_low] == "0") && (new_expression[first_arg_idx_high] == "0")) // / 0 0 -> nan
        {
            //puts("hi 290");
            new_expression[op_idx] = "nan"; //change '/' to 'nan'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_high] == "0") // / x 0 -> nan (for now, because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to / x 0, which is, for now, assumed to be nan for simplicity)
        {
            //puts("hi 282");
            //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
            //for now we weed it out because annoying not to; giving this up seems like the better deal...
            //TODO: need to retest this in simplification script
            new_expression[op_idx] = "nan";//(new_expression[first_arg_idx_low] != "~") ? "inf": "-inf"; //change '/' to 'inf' or '-inf'
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

        //TODO:
            /*
            x*y       y
            ---  -->  -
            x*z       z
            */
    }
    else if (expression[low] == "^") // ^ x y
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // ^
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ^ x
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // ^ x y
        //int second_arg_idx_high = new_expression.size();
        //int step;
        if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) // ^ nan x -> nan, ^ x nan -> nan, ^ nan nan -> nan
        {
            //puts("hi 417");
            new_expression[op_idx] = "nan"; //change '^' to 'nan'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_high] == "0") //^ x 0 -> 1 (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to ^ x 0, which is 1)
        {
            //puts("hi 334");
            new_expression[op_idx] = "1"; //change '^' to '1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "0") // ^ 0 x -> nan (for now)
        {
            //puts("hi 340");
            //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
            //for now we weed it out because annoying not to; giving this up seems like the better deal...
            //TODO: need to retest this in simplification script
            new_expression[op_idx] = "nan";//new_expression[op_idx] = "0"; //change '^' to '0'
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
            //puts("hi 402");
            new_expression[op_idx] = "0"; //change 'tanh' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "inf") // tanh inf -> 1
        {
            //puts("hi 408");
            new_expression[op_idx] = "1"; //change 'tanh' to '1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "-inf") // tanh -inf -> -1
        {
            //puts("hi 414");
            new_expression[op_idx] = "-1"; //change 'tanh' to '-1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // tanh ~ inf -> 1
        {
//            puts("hi 421");
            new_expression[op_idx] = "-1"; //change 'tanh' to '-1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else if (expression[low] == "sech") // sech x
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // sech
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sech x
//        if (new_expression[first_arg_idx_low] == "nan") // sech nan -> nan
//        {
////            puts("hi 466");
//            new_expression[op_idx] = "nan"; //change 'sech' to 'nan'
//            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
//        }
        if (new_expression[first_arg_idx_low] == "0") // sech 0 -> 1
        {
            //puts("hi 416");
            new_expression[op_idx] = "1"; //change 'sech' to '1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "inf") // sech inf -> 0
        {
            //puts("hi 440");
            new_expression[op_idx] = "0"; //change 'sech' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "-inf") // sech -inf -> 0
        {
//            puts("hi 446");
            new_expression[op_idx] = "0"; //change 'sech' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // sech ~ inf -> 0
        {
//            puts("hi 452");
            new_expression[op_idx] = "0"; //change 'sech' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else if (expression[low] == "~") // ~ x
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // ~
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ~ x
//        if (new_expression[first_arg_idx_low] == "nan") // ~ nan -> nan
//        {
////            puts("hi 466");
//            new_expression[op_idx] = "nan"; //change '~' to 'nan'
//            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
//        }
        if (new_expression[first_arg_idx_low] == "0") // ~ 0 -> 0
        {
//            puts("hi 466");
            new_expression[op_idx] = "0"; //change '~' to '0'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "inf") // ~ inf -> -inf
        {
            //puts("hi 507");
            new_expression[op_idx] = "-inf"; //change '~' to '-inf'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else if (expression[low] == "exp") // exp x
    {
        int op_idx = new_expression.size();
        new_expression.push_back(expression[low]); // exp
        int temp = low+1+grasp[low+1];
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // exp x
        if (new_expression[first_arg_idx_low] == "0") // exp 0 -> 1
        {
            //puts("hi 521");
            new_expression[op_idx] = "1"; //change 'exp' to '1'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
        else if (new_expression[first_arg_idx_low] == "inf") // exp inf -> inf
        {
            //puts("hi 527");
            new_expression[op_idx] = "inf"; //change 'exp' to 'inf'
            new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
        }
    }
    else if (expression[low] == "abs") // abs x
    {
        new_expression.push_back(expression[low]); // abs
        int temp = low+1+grasp[low+1];
        graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // abs x
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
//scans entire `expression` for the following:
//     bin_op number1 number2 -> numberResult
//     un_op number1 -> numberResult
//the function repeatedly iterates over `expression` until no more
//instances of the above are found.



//+ cos - + 1 3 x + * cos 2 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
//+ cos - 4 x + * cos 2 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
//+ cos - 4 x + * -0.42 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
//+ cos - 4 x + * -0.42 0.41 + arcsin - 8 7 sin tanh cos * 3 4
void simplifyPN_Helper(std::vector<std::string>& expression)
{
    bool simplified = true;
    bool isdouble1, isdouble2, isConst1, isConst2;
    while (simplified)
    {
        simplified = false;
        if (expression.size() > 1)
        {
            for (size_t i = 0; i < expression.size() - 1; i++)
            {
                if (is_binary(expression[i]))
                {
                    isdouble1 = isdouble(expression[i+1]);
                    isdouble2 = isdouble(expression[i+2]);
                    
                    if (isdouble1 && isdouble2)
                    {
                        if (expression[i] == "+")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i+1]) + Stod(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "-")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i+1]) - Stod(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "*")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i+1]) * Stod(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "/")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i+1]) / Stod(expression[i+2])));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "^")
                        {
                            expression[i] = simplifyString(to_string_general(std::powf(Stod(expression[i+1]), Stod(expression[i+2]))));
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    isConst1 = is_const(expression[i+1]);
                    isConst2 = is_const(expression[i+2]);
                    
                    if ((isConst1 && isConst2) && ((expression[i+1].find("nan") != std::string::npos) || (expression[i+2].find("nan") != std::string::npos))) //binary_op nan x = binary_op x nan = nan
                    {
                        //puts("hi 570");
                        expression[i] = "nan";
                        expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "-")
                    {
                        if ((isConst1 && isConst2) && (expression[i+1] == expression[i+2])) //- x x => 0
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
                        else if (expression[i+2] == "0" && isConst1) //- x 0 -> x
                        {
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                    else if (expression[i] == "*")
                    {
                        if (expression[i+1] == "0" && isConst2) //* 0 x -> 0
                        {
                            //puts("hi 131");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0" && isConst1) //* x 0 -> 0
                        {
                            //puts("hi 139");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "1" && isConst2) //* 1 x -> x
                        {
                            //puts("hi 147");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && isConst1) //* x 1 -> x
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
                        if (expression[i+1] == "0" && isConst2) //+ 0 x -> x
                        {
                            //puts("hi 167");
                            expression[i] = expression[i+2];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "0" && isConst1) //+ x 0 -> x
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
                        if (expression[i+1] == "0" && isConst2) // / 0 x -> 0
                        {
                            //puts("hi 187");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && isConst1) // / x 1 -> x
                        {
                            //puts("hi 195");
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (isConst1 && isConst2 && (expression[i+1] == expression[i+2])) // / x x -> 1
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
                        if (expression[i+2] == "0" && isConst1) // ^ x 0 -> 1
                        {
                            //puts("hi 223");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "0" && isConst2) // ^ 0 x -> 0 (x > 0)
                        {
                            //puts("hi 215");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+1] == "1" && isConst2) // ^ 1 x -> 1
                        {
                            //puts("hi 231");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i+2] == "1" && isConst1) // ^ x 1 -> x
                        {
                            //puts("hi 239");
                            expression[i] = expression[i+1];
                            expression.erase(expression.begin() + i + 1, expression.begin() + i + 3); // Remove elements at i + 1 and i + 2
                            simplified = true;
                            break;
                        }
                    }
                }
                
                else if (is_unary(expression[i]) && isdouble(expression[i+1]))
                {
                    if (expression[i] == "cos")
                    {
                        expression[i] = simplifyString(to_string_general(cos(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "~")
                    {
                        expression[i] = simplifyString(to_string_general(-(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sin")
                    {
                        expression[i] = simplifyString(to_string_general(sin(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if ((expression[i] == "ln") || (expression[i] == "log"))
                    {
                        expression[i] = simplifyString(to_string_general(log(Stod(expression[i+1])))); // Natural log (ln)
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "asin" || expression[i] == "arcsin")
                    {
                        expression[i] = simplifyString(to_string_general(asin(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "acos" || expression[i] == "arccos")
                    {
                        expression[i] = simplifyString(to_string_general(acos(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "exp")
                    {
                        expression[i] = simplifyString(to_string_general(exp(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sech")
                    {
                        expression[i] = simplifyString(to_string_general(1 / cosh(Stod(expression[i+1])))); // sech(x) = 1 / cosh(x)
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "tanh")
                    {
                        expression[i] = simplifyString(to_string_general(tanh(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sqrt")
                    {
                        expression[i] = simplifyString(to_string_general(sqrt(Stod(expression[i+1]))));
                        expression.erase(expression.begin() + i + 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "abs")
                    {
                        expression[i] = simplifyString(to_string_general(abs(Stod(expression[i+1]))));
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
                    //TODO: Add 0 ~ -> 0
                    //TODO: Add inf ~ -> -inf
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
                    //TODO: uncomment the below!
//                    else if ((expression[i] == "sech") && (expression[i+1] == "~")) //sech(-x) = sech(x)
//                    {
//                        //puts("hi 708");
//                        expression.erase(expression.begin() + i + 1); // Remove the '~'
//                        simplified = true;
//                        break;
//                    }
                }
            }
        }
    }
}

void simplifyPN(std::vector<std::string>& expression)
{
    simplifyPN_Helper(expression);
//    print_container(expression);
    graspSimplifyPrefix(expression, 0, expression.size() - 1, grasp);
//    print_container(expression);
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
    
    test_expr = {"sech", "tanh", "^", "sin", "sin", "arcsin", "^", "0", "*", "y", "y", "1"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sech", "sin", "sin", "arcsin", "^", "0", "*", "y", "y"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "/", "tanh", "cos", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "/", "sech", "cos", "+", "z", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "/", "~", "sech", "cos", "+", "z", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "/", "~", "tanh", "cos", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "~", "inf"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "-", "0", "inf"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sech", "/", "~", "sech", "cos", "+", "z", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sech", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sech", "~", "/", "~", "sech", "cos", "+", "z", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sech", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"sech", "+", "~", "*", "0", "tanh", "tanh", "x", "x"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "-", "cos", "x", "cos", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "-", "sech", "cos", "x", "sech", "cos", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "-", "+", "z", "nan", "-", "nan", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "/", "~", "-", "x", "nan", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "sech", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
        
    test_expr = {"~", "/", "sech", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"~", "/", "tanh", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"exp", "/", "tanh", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"exp", "/", "sech", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"exp", "*", "tanh", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"exp", "*", "sech", "~", "/", "~", "tanh", "cos", "x", "sin", "+", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"exp", "*", "sech", "~", "/", "~", "tanh", "cos", "x", "sin", "/", "0", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"cos", "+", "tanh", "~", "/", "~", "tanh", "cos", "x", "sin", "/", "x", "0", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"tanh", "/", "sin", "~", "/", "~", "tanh", "cos", "x", "sin", "^", "0", "cos", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"exp", "/", "sech", "~", "/", "~", "tanh", "cos", "x", "sin", "^", "0", "+", "x", "x", "0"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"+", "nan", "cos", "sin", "^", "x", "tanh", "2"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"-", "cos", "sin", "^", "x", "tanh", "2", "nan"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"-", "nan", "cos", "sin", "^", "x", "tanh", "2"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"+", "cos", "sin", "^", "x", "tanh", "2", "nan"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "nan", "cos", "sin", "^", "x", "tanh", "2"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "cos", "sin", "^", "x", "tanh", "2", "nan"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "nan", "tanh", "sech", "+", "x", "tanh", "-", "2", "^", "x", "2"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"*", "sin", "asin", "/", "x", "tanh", "sech", "acos", "arccos", "+", "apple", "2", "nan"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "nan", "asin", "acos", "/", "x", "tanh", "2"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"/", "acos", "tanh", "-", "y", "tanh", "2", "nan"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"^", "nan", "sech", "+", "1", "+", "x", "tanh", "-", "2", "^", "x", "2"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"^", "sech", "asin", "-", "apple", "tanh", "sech", "acos", "arccos", "+", "apple", "2", "nan"};
    printf("before: ");print_container(test_expr);
    simplifyPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
}
//g++ -std=c++20 -o PrefixSimplify PrefixSimplify.cpp
//MARK: Number of non-production-tested simplifications so far: 4
//https://stackoverflow.com/questions/20153412/simplification-algorithm-for-reverse-polish-notation
//https://dl.acm.org/
//simplification of polish notation expressions articles
// ! objdump -d -M intel PrefixSimplify
