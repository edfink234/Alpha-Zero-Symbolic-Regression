//TODO: Include `nan x bin_op` = `x nan bin_op` = `nan un_op` = `nan`; needs to be first case to test for each "block"
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <cmath>
#include <charconv>
#include <cassert>

const std::unordered_set<std::string> unary_operators = {"cos", "~", "sin", "log", "ln", "asin", "arcsin", "acos", "arccos", "exp", "sech", "tanh", "sqrt"};
const std::unordered_set<std::string> binary_operators = {"+", "-", "*", "/", "^"};
const std::string expression_type = "postfix";
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
        double val = std::stof(param);
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

void setPostfixGR(const std::vector<std::string>& postfix, std::vector<int>& grasp)
{
    grasp.reserve(postfix.size()); //grasp[k] = GR( postfix[k]), k = 1, ... ,i.
    //In the paper they do `k = 1;` instead of `k = 0;`, presumably because GR(postfix[0]) always is 0, but it works
    //if you set k = 0 too.
    for (size_t k = 0; k < postfix.size(); ++k)
    {
        grasp.push_back(GR(k, postfix));
    }
}

//0 x x tanh sin cos + * -> 0 * (cos(sin(tanh(x)+x)
//e.g. 0*x, 1*x, x*0, x*1, x-x, x+0, 0+x, 0-x, x-0, ...
void graspSimplifyPostfixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
{
    if (!setGRvar)
    {
        grasp.clear();
        setPostfixGR(expression, grasp);
    }
//    print_container(expression, low, up);
    if (expression[up] == "+" || expression[up] == "-") // x y +/-
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true);
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true);
        int second_arg_idx_high = new_expression.size();
        int step;
        
        if (new_expression.back() == "0") // x 0 +/- -> x
        {
            //puts("hi 181");
            new_expression.pop_back();
        }
        
        else if (new_expression[first_arg_idx_high - 1] == "0")
        {
            //puts("hi 184");
            //erase elements from new_expression[first_arg_idx_low] to new_expression[first_arg_idx_high-1] inclusive
            new_expression.erase(new_expression.begin() + first_arg_idx_low, new_expression.begin() + first_arg_idx_high); //0 y + -> y
            if (expression[up] == "-")
            {
                //puts("hi 187");
                new_expression.push_back("~"); //0 y - -> y ~
            }
        }
        
        else if ((expression[up] == "-") && ((step = (first_arg_idx_high - first_arg_idx_low)) == (second_arg_idx_high - first_arg_idx_high)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) //x x - -> 0
        {
            //puts("hi 215");
            new_expression[first_arg_idx_low] = "0"; //change first symbol of x to 0
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.begin() + second_arg_idx_high); //erase the rest of x and y
        }
        
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "*") //x y *
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true); //x
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true); //y
        //int second_arg_idx_high = new_expression.size();
        //int step;
        
        if (new_expression.back() == "0") // x 0 * -> 0 (because, since postfix operators come at the end, if the end of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 *, which is 0)
        {
            //puts("hi 235");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression[first_arg_idx_high - 1] == "0") //0 x * -> 0 (because, since postfix operators come at the end, if the end of the first argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x *, which is 0)
        {
            //puts("hi 241");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "1") // x 1 * -> x (because, since postfix operators come at the end, if the end of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 *, which is x)
        {
            //puts("hi 247");
            new_expression.pop_back(); //erase the '1'
        }
        else if (new_expression[first_arg_idx_high - 1] == "1") //1 x * -> x (because, since postfix operators come at the end, if the end of the first argument of '*' is 1, then the whole first argument MUST be 1, therefore the expression reduces to 1 x *, which is x)
        {
            //puts("hi 252");
            new_expression.erase(new_expression.begin() + first_arg_idx_high - 1); //erase the '1'
        }
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "/") //x y /
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true); //x
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true); //y
        int second_arg_idx_high = new_expression.size();
        int step;
        if ((new_expression.back() == "0") && (new_expression[first_arg_idx_high - 1] == "0")) // 0 0 / -> nan
        {
            //puts("hi 279");
            new_expression[first_arg_idx_low] = "nan";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "0") // x 0 / -> nan (for now, because, since postfix operators come at the end, if the end of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 /, which is, for now, assumed to be nan for simplicity)
        {
            //puts("hi 280");
            //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
            //for now we weed it out because annoying not to; giving this up seems like the better deal...
            //TODO: need to retest this in simplification script
            new_expression[first_arg_idx_low] = "nan";//(new_expression[first_arg_idx_high - 1] == "~") ? "-inf" : "inf";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression[first_arg_idx_high - 1] == "0") //0 x / -> 0 (because, since postfix operators come at the end, if the end of the first argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x /, which is 0)
        {
            //puts("hi 286");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "1") // x 1 / -> x (because, since postfix operators come at the end, if the end of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 /, which is x)
        {
            //puts("hi 292");
            new_expression.pop_back(); //erase the '1'
        }
        else if ((expression[up] == "/") && ((step = (first_arg_idx_high - first_arg_idx_low)) == (second_arg_idx_high - first_arg_idx_high)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) // / x x -> 1
        {
            //puts("hi 297");
            new_expression[first_arg_idx_low] = "1"; //change first symbol of x to 1
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.begin() + second_arg_idx_high); //erase the rest of x and y
        }
        //TODO:
            /*
            x*y       y
            ---  -->  -
            x*z       z
            */
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "^") //x y ^
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-2-grasp[up-1], grasp, new_expression, true); //x
        int first_arg_idx_high = new_expression.size();
        graspSimplifyPostfixHelper(expression, up-1-grasp[up-1], up-1, grasp, new_expression, true); //y
        //int second_arg_idx_high = new_expression.size();
        //int step;
        
        if (new_expression.back() == "0") // x 0 ^ -> 1 (because, since postfix operators come at the end, if the end of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 ^, which is 1)
        {
            //puts("hi 318");
            new_expression[first_arg_idx_low] = "1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression[first_arg_idx_high - 1] == "0") //0 x ^ -> nan (for now, because, since postfix operators come at the end, if the end of the first argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x ^, which is, for now, assumed to be nan for simplicity)
        {
            //puts("hi 324");
            //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
            //for now we weed it out because annoying not to; giving this up seems like the better deal...
            //TODO: need to retest this in simplification script
            new_expression[first_arg_idx_low] = "nan";// "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "1") // x 1 ^ -> x (because, since postfix operators come at the end, if the end of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 ^, which is x)
        {
            //puts("hi 330");
            new_expression.pop_back(); //erase the '1'
        }
        else if (new_expression[first_arg_idx_high - 1] == "1") //1 x ^ -> 1 (because, since postfix operators come at the end, if the end of the first argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to 1 x ^, which is 1)
        {
            //puts("hi 335");
            new_expression[first_arg_idx_low] = "1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "cos") //x cos
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
        if (new_expression.back() == "0") // 0 cos -> 1 (because, since postfix operators come at the end, if the end of the argument of 'cos' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 cos, which is 1)
        {
            //puts("hi 350");
            new_expression[first_arg_idx_low] = "1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        //TODO: Add cos(inf) -> nan
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "sin") //x sin
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
        if (new_expression.back() == "0") // 0 sin -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sin' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sin, which is 0)
        {
            //puts("hi 365");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        //TODO: Add sin(inf) -> nan
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "tanh") //x tanh
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
        if (new_expression.back() == "0") // 0 tanh -> 0 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 tanh, which is 0)
        {
            //puts("hi 380");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "inf") // inf tanh -> 1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf tanh, which is 1)
        {
            //puts("hi 386");
            new_expression[first_arg_idx_low] = "1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "-inf") // -inf tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
        {
            //puts("hi 392");
            new_expression[first_arg_idx_low] = "-1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
        {
            //puts("hi 392");
            new_expression[first_arg_idx_low] = "-1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "sech") //x sech
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
        if (new_expression.back() == "0") // 0 sech -> 1 (because, since postfix operators come at the end, if the end of the argument of 'sech' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sech, which is 1)
        {
            //puts("hi 395");
            new_expression[first_arg_idx_low] = "1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "inf") // inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf sech, which is 0)
        {
//            puts("hi 419");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if (new_expression.back() == "-inf") // -inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
        {
            //puts("hi 425");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
        {
            //puts("hi 431");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "~") //x ~
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
        if (new_expression.back() == "0") // 0 ~ -> 0 (because, since postfix operators come at the end, if the end of the argument of '~' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 ~, which is 0)
        {
//            puts("hi 445");
            new_expression[first_arg_idx_low] = "0";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        if (new_expression.back() == "inf") // inf ~ -> -inf (because, since postfix operators come at the end, if the end of the argument of '~' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf ~, which is -inf)
        {
            //puts("hi 507");
            new_expression[first_arg_idx_low] = "-inf";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "exp") //x exp
    {
        int first_arg_idx_low = new_expression.size();
        graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
        if (new_expression.back() == "0") // 0 exp -> 1 (because, since postfix operators come at the end, if the end of the argument of 'exp' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 exp, which is 1)
        {
            //puts("hi 524");
            new_expression[first_arg_idx_low] = "1";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        if (new_expression.back() == "inf") // inf exp -> inf (because, since postfix operators come at the end, if the end of the argument of '~' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf ~, which is -inf)
        {
            //puts("hi 530");
            new_expression[first_arg_idx_low] = "inf";
            new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
        }
        else
        {
            new_expression.push_back(expression[up]);
        }
    }
    else if (expression[up] == "abs") //x abs
    {
        graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
        new_expression.push_back(expression[up]); //abs
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

void graspSimplifyPostfix(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp)
{
    std::vector<std::string> new_expression;
    new_expression.reserve(expression.size());
    graspSimplifyPostfixHelper(expression, low, up, grasp, new_expression, false);
    expression = new_expression;
}
// 0 0 0 1 + - * tanh sech 1 /
// 0 0 1 - * tanh sech 1 /
// 0 -1 * tanh sech 1 /
// 0 tanh sech 1 /
// 0 sech 1 /
// 1 1 /
// 1
void simplifyRPN_Helper(std::vector<std::string>& expression)
{
    bool simplified = true;
    bool isdouble1, isdouble2, isConst1, isConst2;
    while (simplified)
    {
        simplified = false;
        if (expression.size() > 1)
        {
            for (size_t i = 1; i < expression.size(); i++)
            {
                if (is_binary(expression[i]))
                {
                    isdouble1 = isdouble(expression[i-1]);
                    isdouble2 = isdouble(expression[i-2]);
                    
                    if (isdouble1 && isdouble2)
                    {
                        if (expression[i] == "+")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i-2]) + Stod(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "-")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i-2]) - Stod(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "*")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i-2]) * Stod(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "/")
                        {
                            expression[i] = simplifyString(to_string_general(Stod(expression[i-2]) / Stod(expression[i-1])));
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i] == "^")
                        {
                            expression[i] = simplifyString(to_string_general(std::powf(Stod(expression[i-2]), Stod(expression[i-1]))));
//                            printf("hi 566, res = %s\n", expression[i].c_str());
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                    
                    isConst1 = is_const(expression[i-1]);
                    isConst2 = is_const(expression[i-2]);
                    
                    if ((isConst1 && isConst2) && ((expression[i-1].find("nan") != std::string::npos) || (expression[i-2].find("nan") != std::string::npos))) //x nan binary_op = nan x binary_op = nan
                    {
//                        puts("hi 549");
                        expression[i] = "nan";
                        expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "-")
                    {
                        if ((isConst1 && isConst2) && (expression[i-1] == expression[i-2])) //x x - => 0
                        {
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-2] == "0" && isConst1) //"0 x -" -> "x ~"
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
                        if (expression[i-2] == "0" && isConst1) //"0 x *" -> "0"
                        {
                            //puts("hi 131");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "0" && isConst2) //"x 0 *" -> "0"
                        {
                            //puts("hi 139");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-2] == "1" && isConst1) //"1 x *" -> "x"
                        {
                            //puts("hi 147");
                            expression[i] = expression[i-1];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "1" && isConst2) //"x 1 *" -> "x"
                        {
                            //puts("hi 155");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }

                    else if (expression[i] == "+")
                    {
                        if (expression[i-2] == "0" && isConst1) //"0 x +" -> "x"
                        {
                            //puts("hi 167");
                            expression[i] = expression[i-1];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "0" && isConst2) //"x 0 +" -> "x"
                        {
                            //puts("hi 175");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }

                    else if (expression[i] == "/")
                    {
                        if (expression[i-2] == "0" && isConst1) // "0 x /" -> "0"
                        {
                            //puts("hi 187");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "1" && isConst2) // "x 1 /" -> "x"
                        {
                            //puts("hi 195");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (isConst1 && isConst2 && (expression[i-1] == expression[i-2])) // "x x /" -> "1"
                        {
                            //puts("hi 203");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }

                    else if (expression[i] == "^")
                    {
                        if (expression[i-1] == "0" && isConst2) // "x 0 ^" -> "1"
                        {
                            //puts("hi 223");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-2] == "0" && isConst1) // "0 x ^" -> "0" (x > 0)
                        {
//                            puts("hi 215");
                            expression[i] = "0";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-2] == "1" && isConst1) // "1 x ^" -> "1"
                        {
                            //puts("hi 231");
                            expression[i] = "1";
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                        else if (expression[i-1] == "1" && isConst2) // "x 1 ^" -> "x"
                        {
                            //puts("hi 239");
                            expression[i] = expression[i-2];
                            expression.erase(expression.begin() + i - 2, expression.begin() + i); // Remove elements at i - 1 and i - 2
                            simplified = true;
                            break;
                        }
                    }
                }
                
                else if (is_unary(expression[i]) && isdouble(expression[i-1]))
                {
                    if (expression[i] == "cos")
                    {
                        expression[i] = simplifyString(to_string_general(cos(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "~")
                    {
                        expression[i] = simplifyString(to_string_general(-(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sin")
                    {
                        expression[i] = simplifyString(to_string_general(sin(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if ((expression[i] == "ln") || (expression[i] == "log"))
                    {
                        expression[i] = simplifyString(to_string_general(log(Stod(expression[i-1])))); // Natural log (ln)
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "asin" || expression[i] == "arcsin")
                    {
                        expression[i] = simplifyString(to_string_general(asin(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "acos" || expression[i] == "arccos")
                    {
                        expression[i] = simplifyString(to_string_general(acos(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "exp")
                    {
                        expression[i] = simplifyString(to_string_general(exp(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sech")
                    {
                        expression[i] = simplifyString(to_string_general(1 / cosh(Stod(expression[i-1])))); // sech(x) = 1 / cosh(x)
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "tanh")
                    {
                        expression[i] = simplifyString(to_string_general(tanh(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sqrt")
                    {
                        expression[i] = simplifyString(to_string_general(sqrt(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "abs")
                    {
                        expression[i] = simplifyString(to_string_general(abs(Stod(expression[i-1]))));
                        expression.erase(expression.begin() + i - 1);
                        simplified = true;
                        break;
                    }
                }
                
                else if (is_unary(expression[i]))
                {
                    if (expression[i] == "~" && expression[i-1] == "~")
                    {
                        expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                        simplified = true;
                        break;
                    }
                    //TODO: Add 0 ~ -> 0
                    else if (expression[i] == "exp" && (expression[i-1] == "ln" || expression[i-1] == "log"))
                    {
                        //puts("hi 360");
                        expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                        simplified = true;
                        break;
                    }
                    else if (expression[i-1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                    {
                        //puts("hi 368");
                        expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "cos" && (expression[i-1] == "acos" || expression[i-1] == "arccos"))
                    {
                        //puts("hi 408");
                        expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "cos" && expression[i-1] == "~") //cos(-x) -> cos(x)
                    {
                        //puts("hi 688");
                        expression.erase(expression.begin() + i - 1); // Remove the '~'
                        simplified = true;
                        break;
                    }
                    else if (expression[i-1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                    {
                        //puts("hi 416");
                        expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                        simplified = true;
                        break;
                    }
                    else if (expression[i] == "sin" && (expression[i-1] == "asin" || expression[i-1] == "arcsin"))
                    {
                        //puts("hi 424");
                        expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                        simplified = true;
                        break;
                    }
                    else if (expression[i-1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                    {
                        //puts("hi 432");
                        expression.erase(expression.begin() + i - 1, expression.begin() + i + 1); // Remove elements at i - 1 and i
                        simplified = true;
                        break;
                    }
                }
            }
        }
    }
}

void simplifyRPN(std::vector<std::string>& expression)
{
    simplifyRPN_Helper(expression);
    graspSimplifyPostfix(expression, 0, expression.size() - 1, grasp);
    simplifyRPN_Helper(expression);
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
    
    test_expr = {"x0", "cos", "x0", "x0", "sin", "~", "*", "-", "x0", "cos", "x0", "cos", "*", "/", "x0", "x0", "cos", "/", "sech", "x0", "x0", "cos", "/", "sech", "*", "*", "1", "x0", "x0", "cos", "/", "tanh", "x0", "x0", "cos", "/", "tanh", "*", "-", "sqrt", "/", "~", "x0", "x0", "cos", "/", "tanh", "acos", "sin", "~", "*"};
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
    
    test_expr = {"1", "w", "/", "y", "1", "/", "/", "asin", "sin", "sin", "asin", "sin", "sin", "arcsin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "w", "/", "y", "1", "/", "/", "arccos", "cos", "cos", "acos", "cos", "cos", "acos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "w", "/", "y", "1", "/", "/", "sin", "asin", "asin", "sin", "asin", "asin", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "w", "/", "y", "1", "/", "/", "cos", "acos", "acos", "cos", "acos", "arccos", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "0", "x", "x", "x", "+", "+", "+", "+", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "+", "cos", "cos", "sin", "tanh", "0", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "+", "cos", "cos", "sin", "tanh", "x", "x", "+", "cos", "cos", "sin", "tanh", "-"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "w", "/", "y", "1", "/", "/", "cos", "acos", "acos", "cos", "acos", "arccos", "cos", "0", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "+", "sin", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "y", "x", "x", "+", "tanh", "-", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "+", "tanh", "x", "*", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "1", "x", "x", "+", "asin", "x", "*", "*", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "1", "x", "x", "+", "asin", "x", "*", "*", "*", "~", "0", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "+", "tanh", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "1", "x", "x", "+", "asin", "x", "*", "*", "*", "0", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "0", "x", "x", "x", "+", "+", "+", "+", "+", "1", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "1", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "1", "x", "x", "+", "asin", "x", "*", "*", "*", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "1", "w", "/", "y", "1", "/", "/", "cos", "acos", "acos", "cos", "acos", "arccos", "cos", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "^", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*", "^", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*", "^", "~", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "~", "cos", "x", "x", "-", "*", "0", "/", "^", "cos"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "~", "cos", "x", "x", "-", "*", "0", "/", "^", "1", "-", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*", "^", "cos", "1", "-", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "~", "cos", "x", "x", "-", "*", "0", "/", "^", "1", "-", "sin", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*", "^", "cos", "1", "-", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*", "^", "cos", "1", "-", "tanh", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "x", "x", "*", "1", "x", "x", "+", "tanh", "0", "^", "x", "*", "*", "1", "x", "x", "+", "tanh", "x", "*", "*", "/", "*", "^", "cos", "1", "-", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "1", "^", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "+", "x", "x", "-", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "+", "x", "x", "-", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "~", "0", "/", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "~", "0", "/", "1", "^", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"inf", "~", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "inf", "-", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "1", "^", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "+", "x", "x", "-", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "tanh", "0", "/", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "1", "^", "~", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "x", "+", "x", "x", "-", "-", "asin", "tanh", "sin", "x", "x", "-", "*", "0", "/", "tanh", "0", "/", "~", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"nan", "x", "+", "x", "nan", "-", "-", "asin", "tanh", "sin", "x", "nan", "-", "*", "0", "/", "tanh", "0", "/", "~", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "nan", "^", "nan", "x", "^", "-", "asin", "tanh", "sin", "x", "nan", "-", "*", "0", "/", "1", "^", "~", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x", "nan", "^", "nan", "x", "^", "-", "asin", "tanh", "sin", "x", "nan", "-", "*", "0", "/", "1", "^", "~", "sech"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x0", "x1", "^", "6.283190", "cos", "0.000000", "0.000100", "6.283190", "^", "^", "-", "^"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"x1", "10.000000", "*", "43.47847366333008", "-", "185.5887837532312", "/", "0.4000400020000667", "x1", "sin", "x0", "sin", "*", "1.5707963267948966", "*", "-", "+"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "*", "+", "0", "/", "~"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "sin", "x", "*", "+", "0", "/", "~"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");

    test_expr = {"1", "x", "x", "*", "+", "1", "x", "x", "*", "+", "-", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "sin", "x", "*", "+", "1", "x", "sin", "x", "*", "+", "-", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "*", "+", "0", "/", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "sin", "x", "*", "+", "0", "/", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "*", "+", "0", "/", "~", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "sin", "x", "*", "+", "0", "/", "~", "exp"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "x", "*", "+", "0", "/", "~", "cos", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"1", "x", "sin", "x", "*", "+", "0", "/", "~", "exp", "tanh"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "1", "x", "x", "*", "+", "^", "~", "cos", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
    
    test_expr = {"0", "1", "x", "x", "tanh", "*", "+", "^", "~", "cos", "sin"};
    printf("before: ");print_container(test_expr);
    simplifyRPN(test_expr);
    printf("after: ");print_container(test_expr);
    puts("");
}

//g++ -std=c++20 -o PostfixSimplifyPrev PostfixSimplifyPrev.cpp

//https://stackoverflow.com/questions/20153412/simplification-algorithm-for-reverse-polish-notation
//https://dl.acm.org/
//simplification of polish notation expressions articles
