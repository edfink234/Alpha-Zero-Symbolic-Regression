//implementation of https://scindeks-clanci.ceon.rs/data/pdf/0354-0243/2001/0354-02430101061K.pdf
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <fstream>
#define ASSERT false

std::stringstream sout;
std::vector<std::string> derivat;
int Index = 0; //global integer variable initially equal to zero, which represents the Index of the array derivat

//"cos", "exp", "sqrt", "sin", "asin", "ln", "tanh", "acos", "~", "sech"
// ✅     ✅      ✅     ✅     ✅     ✅     ✅      ✅    ✅    ✅
//"+", "-", "*", "/", "^"
//✅   ✅   ✅   ✅   ✅

const std::vector<std::string> unary_operators = {"cos", "~", "sin", "log", "ln", "asin", "arcsin", "acos", "arccos", "exp", "sech", "tanh", "sqrt"};
const std::vector<std::string> binary_operators = {"+", "-", "*", "/", "^"};

bool is_unary(const std::string& token)
{
    return (std::find(unary_operators.begin(), unary_operators.end(), token) != unary_operators.end());
}

bool is_binary(const std::string& token)
{
    return (std::find(binary_operators.begin(), binary_operators.end(), token) != binary_operators.end());
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        out << vec[i];
        if (i < (vec.size() - 1))
        {
            out <<  " ";
        }
    }
    return out;
}

template<typename T>
std::stringstream& operator<<(std::stringstream& out, const std::vector<T>& vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        out << vec[i];
        if (i < (vec.size() - 1))
        {
            out <<  " ";
        }
    }
    return out;
}

bool string_in_file(const std::string& str, const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << '\n';
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.find(str) != std::string::npos)
        {
            return true;
        }
    }

    return false;
}

//Function to compute the LGB, from https://www.jstor.org/stable/43998756 (top of pg. 165)
void LGB(int z, int& ind, const std::vector<std::string>& postfix)
{
    do
    {
        --ind;
        if (is_unary(postfix[ind]))
        {
            LGB(1, ind, postfix);
        }
        else if (is_binary(postfix[ind]))
        {
            LGB(2, ind, postfix);
        }
        --z;
    } while (z);
}

//Computes the grasp of an arbitrary element postfix[i], from https://www.jstor.org/stable/43998756 (bottom of pg. 165)
int GR(int i, const std::vector<std::string>& postfix)
{
    int start = i;
    int& ptr_lgb = start;
    if (is_unary(postfix[i]))
    {
        LGB(1, ptr_lgb, postfix);
    }
    else if (is_binary(postfix[i]))
    {
        LGB(2, ptr_lgb, postfix);
    }
    return (i - ptr_lgb);
}

std::vector<int> getLGBs(const std::vector<std::string>& postfix)
{
    std::vector<int> temp(postfix.size());
    for (size_t k = 0; k < postfix.size(); ++k)
    {
        int start = k;
        int& ptr_lgb = start;
        if (is_unary(postfix[k]))
        {
            LGB(1, ptr_lgb, postfix);
        }
        else if (is_binary(postfix[k]))
        {
            LGB(2, ptr_lgb, postfix);
        }
        temp[k]=(ptr_lgb);
    }
    return temp;
}


void setGR(const std::vector<std::string>& postfix, std::vector<int>& grasp)
{
    grasp.reserve(postfix.size()); //grasp[k] = GR( postfix[k]), k = 1, ... ,i.
    //In the paper they do `k = 1;` instead of `k = 0;`, presumably because GR(postfix[0]) always is 0, but it works
    //if you set k = 0 too.
    for (size_t k = 0; k < postfix.size(); ++k)
    {
        grasp.push_back(GR(k, postfix));
    }
}

bool areDerivatRangesEqual(int start_idx_1, int start_idx_2, int num_steps)
{
    int stop_idx_1 = start_idx_1 + num_steps;
    
    for (int i = start_idx_1, j = start_idx_2; i < stop_idx_1; i++, j++)
    {
        if (derivat[i] != derivat[j])
        {
            return false;
        }
    }
    return true;
}

/*
low and up: lower and upper Index bounds, respectively, for the piece of the array postfix which is to be the subject of the processing.
dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
*/
void derivePostfixHelper(int low, int up, const std::string& dx, const std::vector<std::string>& postfix, std::vector<int>& grasp, bool setGRvar = false, bool trace_derivat = false)
{
    if (!setGRvar)
    {
        grasp.clear();
        derivat.clear();
        // std::cout << derivat.size();
        derivat.reserve(100);
//        Index = 0;
        setGR(postfix, grasp);
    }
    //allowed ops: +, -, *, /, ^, unary +, unary -, sin(), cos(), tan(), ctg(), log(), sqrt(), const, x0, x1, ..., x_numFeatures
    //Define `grasp` of postfix[i], i.e., the number of elements forming operands of postfix[i] (grasp(operand) = 0)
    //The grasped elements of postfix[i] are the elements forming operands of postfix[i]
    //The left-grasp-bound (LGB) of postfix[i] is the Index of the left-most grasped element of postfix[i] in the array postfix
    //For the expression formed by postfix[i] and its grasped elements, the element postfix[i] is termed the `head` or `main element`
    //ANY element postfix[i] is an N-arity operator acting on operands arg1, ..., argN, the heads of which are op1, ..., opN, where op1 is the left-neighest operator of postfix[i] in the array postfix (so operands are 0-arity operators)
    //For an N-arity operator postfix[i]:
        //The `grasp` of postfix[i]  is equal to i - LGB(postfix[i])
        //grasp(postfix[i]) = N + sum(grasp(op1), ..., grasp(op_k), ..., grasp(opN))
        //grasp(postfix[i]) = N + sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //LGB(postfix[i]) = i - N - sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //op_(N-j) = postfix[i - sum(grasp(op_(N-1)), ..., grasp(op_(N-j-1))) - j - 1], where j = 0, ..., N-1
    //If the grasp of an arbitrary postfix[i] is greater than N, then at least one of its argument heads is also an operator.
        //Example: If the grasp of any binary operator postfix[i] is greater than 2, then at least one of the two preceding elements in the RPN of the expression (postfix[i-1] and postfix[i-2]) is also an operator (unary or binary).
    //postfix[numElements] is certainly an operator (unary or binary)
    
    //if postfix[up] is a binary operator, then:
        //the head of its second argument (let's call it op2) is equal to postfix[up-1]
        //then the grasped elements of op2 are the elements from postfix[up-1-grasp[up-1]] to postfix[up-1]
            //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "*", "+"}, up = 7 -> postfix[up] = "+" is binary
            //so postfix[up-1] = "*" is the head of the second argument of "+" and so the grasped elements
            //of postfix[up-1] are the elements [(postfix[up-1-grasp[up-1]] = postfix[6-3] = postfix[3]), postfix[up-1] = postfix[6]]
            //i.e., the elements {"x", "cos", "x", "*"}
        //the head of its first argument (lets call it op1) is equal to postfix[up-grasp(op2)-2] which is equal to postfix[up-2-grasp[up-1]].
        //then the grasped elements of op1 are the elements from postfix[low = 0] to postfix[up-2-grasp[up-1]]
            //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}, up = 9 ->postfix[up] = "+" is binary
            //so postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2] = "*" is the head of the first argument of "+" and so the grasped elements
            //of postfix[up-grasp(op2)-2] are the elements [(postfix[low] = postfix[0], postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2]]
            //i.e., the elements {"x", "x", "*"}

    if (trace_derivat)
    {
        std::cout << "derivat = {" << derivat << "}, low = " << low << ", up = " << up
        << ", postfix[up] = " << postfix[up] << ", postfix[low] = " << postfix[low] << '\n';
    }
    
    if (postfix[up] == "+" || postfix[up] == "-")
    {
        int x_prime_low = derivat.size();
        derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true);  /*Putting x'*/
        int x_prime_high = derivat.size();
        derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true, trace_derivat); /*Putting y'*/
        int y_prime_high = derivat.size();
        int step;
                
        /*
         Simplification cases:
            
          1.) y' == 0, +/-, x'
          2.) x' == 0,   +, y'
          3.) x' == 0,   -, y' ~
         
         */
        
        if (derivat.back() == "0") //1.) x' 0 + -> x'
        {
//            puts("hi 162");
            derivat.pop_back();
        }
        
        else if (derivat[x_prime_high - 1] == "0")
        {
//            puts("hi 168");
            //erase elements from derivat[x_prime_low] to derivat[x_prime_high-1] inclusive
            derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + x_prime_high); //0 y + -> y
            if (postfix[up] == "-") //3.)
            {
//                puts("hi 173");
                derivat.push_back("~"); //0 y - -> y ~
            }
        }
        
        else if ((postfix[up] == "-") && ((step = (x_prime_high - x_prime_low)) == (y_prime_high - x_prime_high)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
        {
//            puts("hi 180");
            derivat[x_prime_low] = "0"; //change first symbol of x' to 0
            derivat.erase(derivat.begin() + x_prime_low + 1, derivat.begin() + y_prime_high); //erase the rest of x' and y'
        }
        
        else
        {
            derivat.push_back(postfix[up]);
        }

    }
    else if (postfix[up] == "*")
    {
        int x_low = derivat.size();
        for (int k = low; k <= up-2-grasp[up-1]; k++) /* x */
        {
            derivat.push_back(postfix[k]);
        }
        if (derivat.back() == "0") //0 y' * -> 0
        {
//            puts("hi 200");
        }
        else
        {
            int x_high = derivat.size();
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true, trace_derivat); /* x y' */
            if (derivat.back() == "0") //x 0 * -> 0
            {
//                puts("hi 208");
                derivat[x_low] = "0"; //change first symbol of x to 0
                derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase rest of x and y'
            }
            else if (derivat[x_high - 1] == "1") //1 y' * -> y'
            {
//                puts("hi 214");
                assert(x_low == x_high - 1);
                derivat.erase(derivat.begin() + x_low); //erase the x since it's 1
            }
            else if (derivat.back() == "1") //x 1 * -> x
            {
//                puts("hi 220");
                derivat.pop_back(); //remove the y' since it's 1
            }
            else
            {
                derivat.push_back("*"); /* x y' "*" */
            }
        }

        int x_prime_low = derivat.size();
        derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x y' "*" x' */
        if (derivat.back() == "0") //0 y * -> 0
        {
//            puts("hi 233");
        }
        else
        {
            int y_low = derivat.size();
            for (int k = up-1-grasp[up-1]; k <= up - 1; k++)
            {
                derivat.push_back(postfix[k]); /* x y' "*" x' y */
            }
            if (derivat.back() == "0") //x' 0 * -> 0
            {
//                puts("hi 244");
                derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + y_low); //erase x'
            }
            else if (derivat[y_low - 1] == "1") //1 y * -> y
            {
//                puts("hi 249");
                assert(y_low - 1 == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low); //remove the 1
            }
            else if (derivat.back() == "1") //x' 1 * -> x'
            {
//                puts("hi 255");
                derivat.pop_back(); //remove the "1"
            }
            else
            {
                derivat.push_back("*"); /* x y' "*" x' y "*" */
            }
        }
        if (derivat[x_prime_low - 1] == "0") // 0 x' y "*" + -> x' y "*"
        {
//            puts("hi 265");
            derivat.erase(derivat.begin() + x_prime_low - 1); //erase 0
        }
        else if (derivat.back() == "0") //x y' "*" 0 + -> x y' "*"
        {
//            puts("hi 270");
            derivat.pop_back();
        }
        else
        {
            derivat.push_back("+"); /* x y' "*" x' y "*" + */
        }
    }
    
    else if (postfix[up] == "/")
    {
        int x_prime_low = derivat.size();
        derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x' */
        int k;
        if (derivat.back() == "0") //0 y * -> 0
        {
//            puts("hi 286");
        }
        else
        {
            int y_low = derivat.size();
            for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x' y */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //x' 0 * -> 0
            {
//                puts("hi 297");
                derivat.erase(derivat.begin() + x_prime_low, derivat.end() - 1); //erase x'
            }
            else if (derivat.back() == "1") //x' 1 * -> x'
            {
//                puts("hi 302");
                derivat.pop_back(); //remove the "1"
            }
            else if (derivat[y_low-1] == "1") //1 y * -> y
            {
//                puts("hi 307");
                derivat.erase(derivat.begin() + y_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*"); /* x' y *  */
            }
        }
        int x_low = derivat.size();
        for (k = low; k <= up-2-grasp[up-1]; k++) /* x' y * x */
        {
            derivat.push_back(postfix[k]);
        }
        if (derivat.back() == "0") //0 y' * -> 0
        {
//            puts("hi 322");
        }
        else
        {
            int y_prime_low = derivat.size();
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true, trace_derivat); /* x' y * x y' */
            if (derivat.back() == "0") //x 0 * -> 0
            {
//                puts("hi 330");
                derivat.erase(derivat.begin() + x_low, derivat.begin() + y_prime_low); //erase x
            }
            else if (derivat.back() == "1") //x 1 * -> x
            {
//                puts("hi 335");
                derivat.pop_back(); //erase the 1
            }
            else if (derivat[y_prime_low - 1] == "1") //1 y' * -> y'
            {
//                puts("hi 340");
                derivat.erase(derivat.begin() + y_prime_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back("*"); /* x' y * x y' * */
            }
        }
        if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - x_low)) && (areDerivatRangesEqual(x_prime_low, x_low, k))) //thing1 thing1 - -> 0
        {
//            puts("hi 350");
            derivat[x_prime_low] = "0"; //change first symbol of x' to 0
            derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest of x' y * and x y' *
        }
        else
        {
            if (derivat[x_low - 1] == "0") //0 x y' * - -> x y' * ~
            {
//                puts("hi 358");
                derivat.erase(derivat.begin() + x_low - 1); //remove "0"
                derivat.push_back("~"); //add "~" at the end
            }
            else if (derivat.back() == "0") //x' y * 0 - -> x' y *
            {
//                puts("hi 364");
                derivat.pop_back(); //remove "0"
            }
            else
            {
                derivat.push_back("-"); /* x' y * x y' * - */
            }
            for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "1") //"1 1 * /" -> ""
            {
//                puts("hi 377");
                derivat.pop_back(); //remove the "1"
            }
            else
            {
                for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y y */
                {
                    derivat.push_back(postfix[k]);
                }
                derivat.push_back("*"); /* x' y * x y' * - y y * */
                derivat.push_back("/"); /* x' y * x y' * - y y * / */
            }
        }
    }
    
    else if (postfix[up] == "^")
    {
        int k;
        int x_low = derivat.size();
        for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
        {
            derivat.push_back(postfix[k]);
        }
        if (derivat.back() == "0") //0 y ^ (0 ln y *)' * -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
        {
//            puts("hi 402");
            return;
        }
        else if (derivat.back() == "1") //1 y ^ (1 ln y *)' * -> 0 (because ln(1) is 0)
        {
            derivat.back() = "0";
//            puts("hi 407");
            return;
        }
        else
        {
//            int y_low = derivat.size();
            for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x y */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == "0") //x 0 ^ (x ln 0 *)' * -> 0
            {
//                puts("hi 419");
                derivat[x_low] = "0"; //change the first symbol of x to "0"
                derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
                return;
            }
            else if (derivat.back() == "1") //x 1 ^ -> x
            {
//                puts("hi 426");
                derivat.pop_back(); //erase the 1
            }
            else
            {
                derivat.push_back("^"); /* x y ^ */
            }
        }

        std::vector<std::string> postfix_temp;
        std::vector<int> grasp_temp;
        size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> x ln y *
        postfix_temp.reserve(reserve_amount);
        grasp_temp.reserve(reserve_amount);
        for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
        {
            postfix_temp.push_back(postfix[k]);
        }
        postfix_temp.push_back("ln"); /* x ln  */
        for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x ln y */
        {
            postfix_temp.push_back(postfix[k]);
        }
        if (postfix_temp.back() == "1") //x ln 1 * -> x ln
        {
//            puts("hi 452");
            postfix_temp.pop_back();
        }
        else
        {
            postfix_temp.push_back("*"); /* x ln y * */
        }
        setGR(postfix_temp, grasp_temp);
        derivePostfixHelper(0, postfix_temp.size() - 1, dx, postfix_temp, grasp_temp, true, trace_derivat); /* x y ^ (x ln y *)' */
        if (derivat.back() == "0") //x y ^ 0 * -> 0
        {
//            puts("hi 455");
            derivat[x_low] = "0"; //change the first symbol of x to "0"
            derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
        }
        else if (derivat.back() == "1") //x y ^ 1 * -> x y ^
        {
//            puts("hi 460");
            derivat.pop_back(); //erase (x ln y *)'
        }
        else
        {
            derivat.push_back("*"); /* x y ^ (x ln y *)' * */
        }
    }

    else if (postfix[up] == "cos")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "0") //0 x sin ~ * -> 0
        {
//            puts("hi 514");
            return;
        }
        int x_low = derivat.size();
        for (int k = low; k <= up-1; k++)
        {
            derivat.push_back(postfix[k]); /* x' x */
        }
        derivat.push_back("sin"); /* x' x sin */
        derivat.push_back("~"); /* x' x sin ~ */
        if (derivat[x_low - 1] == "1") //1 x sin ~ * -> x sin ~
        {
//            puts("hi 526");
            derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
        }
        else
        {
            derivat.push_back("*"); /* x' x sin ~ * */
        }
    }
    
    else if (postfix[up] == "sin")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "0") //0 x cos * -> 0
        {
//            puts("hi 540");
            return;
        }
        int x_low = derivat.size();
        for (int k = low; k <= up-1; k++)
        {
            derivat.push_back(postfix[k]); /* x' x */
        }
        derivat.push_back("cos"); /* x' x cos */
        if (derivat[x_low - 1] == "1") //1 x cos * -> x cos
        {
//            puts("hi 551");
            derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
        }
        else
        {
            derivat.push_back("*"); /* x' x cos * */
        }
    }
    
    else if (postfix[up] == "sqrt")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "0") //0 2 x sqrt * / -> 0
        {
//            puts("hi 565");
            return;
        }
        derivat.push_back("2"); /* x' 2 */
        for (int k = low; k <= up-1; k++) /* x' 2 x */
        {
            derivat.push_back(postfix[k]);
        }
        derivat.push_back("sqrt");    /* x' 2 x sqrt */
        derivat.push_back("*");       /* x' 2 x sqrt * */
        derivat.push_back("/");       /* x' 2 x sqrt * / */
    }
    
    else if (postfix[up] == "log" || postfix[up] == "ln")
    {
        int x_prime_low = derivat.size();
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "0") //0 x / -> 0
        {
//            puts("hi 551");
            assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
            return;
        }
        int x_low = derivat.size();
        for (int k = low; k <= up-1; k++)
        {
            derivat.push_back(postfix[k]);      /* x' x */
        }
        int step = derivat.size() - x_low;
        if ((step == (x_low - x_prime_low)) && areDerivatRangesEqual(x_prime_low, x_low, step)) //something something / -> 1
        {
//            puts("hi 563");
            derivat[x_prime_low] = "1"; //replace first symbol of x' with "1"
            derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest
            return;
        }
        
        derivat.push_back("/");               /* x' x / */
    }
    
    else if (postfix[up] == "asin" || postfix[up] == "arcsin")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "0") //0 1 x x * - sqrt / -> 0
        {
//            puts("hi 610");
            return;
        }
        derivat.push_back("1"); /* x' 1 */
        for (int k = low; k <= up-1; k++) /* x' 1 x */
        {
            derivat.push_back(postfix[k]);
        }
        for (int k = low; k <= up-1; k++) /* x' 1 x x */
        {
            derivat.push_back(postfix[k]);
        }
        derivat.push_back("*");   /* x' 1 x x * */
        derivat.push_back("-");   /* x' 1 x x * - */
        derivat.push_back("sqrt");   /* x' 1 x x * - sqrt */
        derivat.push_back("/");   /* x' 1 x x * - sqrt / */
    }
    
    else if (postfix[up] == "acos" || postfix[up] == "arccos")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "0") //0 1 x x * - sqrt / ~ -> 0
        {
//            puts("hi 633");
            return;
        }
        derivat.push_back("1"); /* x' 1 */
        for (int k = low; k <= up-1; k++) /* x' 1 x */
        {
            derivat.push_back(postfix[k]);
        }
        for (int k = low; k <= up-1; k++) /* x' 1 x x */
        {
            derivat.push_back(postfix[k]);
        }
        derivat.push_back("*");   /* x' 1 x x * */
        derivat.push_back("-");   /* x' 1 x x * - */
        derivat.push_back("sqrt");   /* x' 1 x x * - sqrt */
        derivat.push_back("/");   /* x' 1 x x * - sqrt / */
        derivat.push_back("~");   /* x' 1 x x * - sqrt / ~ */
    }
    
    else if (postfix[up] == "tanh")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); //x'
        if (derivat.back() == "0") //0 x sech x sech * * -> 0
        {
//            puts("hi 657");
            return;
        }
        int x_low = derivat.size();
        for (int k = low; k <= up-1; k++) //x' x
        {
            derivat.push_back(postfix[k]);
        }
        derivat.push_back("sech"); //x' x sech
        for (int k = low; k <= up-1; k++) //x' x sech x
        {
            derivat.push_back(postfix[k]);
        }
        derivat.push_back("sech"); //x' x sech x sech
        derivat.push_back("*"); //x' x sech x sech *
        if (derivat[x_low - 1] == "1") //1 x sech x sech * * -> x sech x sech * *
        {
            puts("hi 676");
            derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
        }
        else
        {
            derivat.push_back("*");                 //x' x sech ~ x tanh * *
        }
    }
    
    else if (postfix[up] == "sech")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); //x'
        if (derivat.back() == "0") //0 x sech ~ x tanh * * -> 0
        {
//            puts("hi 681");
            return;
        }
        int x_low = derivat.size();
        for (int k = low; k <= up-1; k++) //x' x
        {
            derivat.push_back(postfix[k]);
        }
        derivat.push_back("sech");   //x' x sech
        derivat.push_back("~");      //x' x sech ~
        for (int k = low; k <= up-1; k++) //x' x sech ~ x
        {
            derivat.push_back(postfix[k]);
        }
        derivat.push_back("tanh");   //x' x sech ~ x tanh
        derivat.push_back("*");      //x' x sech ~ x tanh *
        if (derivat[x_low - 1] == "1") //1 x sech ~ x tanh * * -> x sech ~ x tanh *
        {
//            puts("hi 699");
            derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
        }
        else
        {
            derivat.push_back("*");                 //x' x sech ~ x tanh * *
        }
    }
    
    else if (postfix[up] == "exp")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "0") //0 x exp * -> 0
        {
//            puts("hi 663");
            return;
        }
        int x_low = derivat.size();
        for (int k = low; k <= up-1; k++)
        {
            derivat.push_back(postfix[k]);      /* x' x */
        }
        derivat.push_back("exp");               /* x' x exp */
        if (derivat[x_low - 1] == "1") //1 x exp * -> x exp
        {
//            puts("hi 674");
            derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
        }
        else
        {
            derivat.push_back("*");               /* x' x exp * */
        }
    }
    
    else if (postfix[up] == "~")
    {
        derivePostfixHelper(low, up-1, dx, postfix, grasp, true, trace_derivat); /* x' */
        if (derivat.back() == "~")
        {
//            puts("hi 561");
            derivat.pop_back(); //two unary minuses cancel each-other
        }
        else
        {
            derivat.push_back(postfix[up]); /* x' ~ */
        }
    }
    
    else
    {
        if (postfix[up] == dx)
        {
            derivat.push_back("1");
        }
        else
        {
            derivat.push_back("0");
        }
    }
}

void derivePostfix(int low, int up, const std::string& dx, const std::vector<std::string>& postfix, std::vector<int>& grasp, bool trace_derivat = false)
{
    derivePostfixHelper(low, up, dx, postfix, grasp, false, trace_derivat);
}


int main()
{
    std::vector<std::string> postfix, temp; //array of postfix expression elements read from left to right
    std::vector<int> grasp;
    
    postfix = {"x","x","+"}; // x+x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 1 + (postfix) -> 1+1 = 2 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x","x","x","-","+"}; // (x-x)+x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp, true); //1 1 1 - + (postfix) -> 1+(1-1) = 1+0 = 1 ✅
                                                             //1 (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x","x","-","x","-","y","+"}; // (x-x)-x+y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp);
        //1 1 - 1 - 0 +  (postfix) -> (1-1)-1+0 = -1 ✅
        //1 1 - 1 - (postfix) -> (1-1)-1 = -1 ✅
        //1 ~ (postfix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"y","y","x","/","*","cos", "y", "+"}; // cos((y*y)/x) + y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp);
        //y y x / * sin ~ y 0 x * y 1 * - x x * / * 0 y x / * + * 0 + (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
        //y y x / * sin ~ y 0 x * y 1 * - x x * / * 0 y x / * + * (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
        //y y x / * sin ~ y 0 x * y 1 * - x x * / * 0 + * (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
        //y y x / * sin ~ y 0 x * y 1 * - x x * / * * (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
        //y y x / * sin ~ y 0 y 1 * - x x * / * * (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
        //y y x / * sin ~ y 0 y - x x * / * * (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
        //y y x / * sin ~ y y ~ x x * / * * (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
        //y y ~ x x * / * y y x / * sin ~ * (postfix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"y","y","x","*","*","cos", "y", "+"}; // cos((y*x)*y) + y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp);
    //y y x * * sin ~ y y 1 * 0 x * + * 0 y x * * + * 0 + (postfix) -> -sin((y*x)*y)*y*y (infix) ✅
    //y y x * * sin ~ y y 1 * 0 x * + * 0 y x * * + * (postfix) -> -sin((y*x)*y)*y*y (infix) ✅
    //y y x * * sin ~ y y 0 x * + * 0 y x * * + * (postfix) -> -sin((y*x)*y)*y*y (infix) ✅
    //y y x * * sin ~ y y 0 + * 0 + * (postfix) -> -sin((y*x)*y)*y*y (infix) ✅
    //y y x * * sin ~ y y * * (postfix) -> -sin((y*x)*y)*y*y (infix) ✅
    //y y * y y x * * sin ~ * (postfix) -> -sin((y*x)*y)*y*y (infix) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"y","x","x","*","+"}; //x*x + y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 x 1 * 1 x * + + (postfix) -> x + x ✅
                                                             //x 1 * 1 x * +  (postfix) -> x + x ✅
                                                             //x 1 x * + (postfix) -> x + x ✅
                                                             //x x + (postfix) -> x + x ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"y","x","x","+","+"}; //x + x + y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 1 1 + + (postfix) -> 1 + 1 ✅
                                                             //1 1 + (postfix) -> 1 + 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"y","x","cos","x","+","+"}; //cos(x) + x + y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 x sin ~ 1 * 1 + + (postfix) -> -sin(x) + 1 ✅
                                                             //x sin ~ 1 * 1 + (postfix) -> -sin(x) + 1 ✅
                                                             //x sin ~ 1 + (postfix) -> -sin(x) + 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"y","x","cos","x","+","-"}; //y - (cos(x) + x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //1 x sin ~ 0 * 0 + - (postfix) -> 1 ✅
                                                             //1 x sin ~ 0 * - (postfix) -> 1 ✅
                                                             //1 (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"y","x","-"}; //y - x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //1 0 - (postfix) -> 1 ✅
                                                             //1 (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "y","x","-", "cos", "cos", "*"}; //x * cos(cos(y-x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp);
    //x y x - cos sin ~ y x - sin ~ 0 1 - * * * 1 y x - cos cos * + (postfix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
    //x y x - cos sin ~ y x - sin ~ 1 ~ * * * 1 y x - cos cos * + (postfix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
    //x y x - cos sin ~ y x - sin ~ 1 ~ * * * y x - cos cos + (postfix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
    //x 1 ~ y x - sin ~ * y x - cos sin ~ * * y x - cos cos + (postfix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "y", "x", "-", "sin", "/", "+"}; // x + (x/sin(y-x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp);
    //1 1 y x - sin * x y x - cos 0 1 - * * - y x - sin y x - sin * / + (postfix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
    //1 1 y x - sin * x y x - cos 1 ~ * * - y x - sin y x - sin * / + (postfix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
    //1 y x - sin x y x - cos 1 ~ * * - y x - sin y x - sin * / + (postfix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
    //1 y x - sin x 1 ~ y x - cos * * - y x - sin y x - sin * / + (postfix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "y", "y", "sin", "cos", "*", "/", "/"}; // x / (x / (y*cos(sin(y))))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //0 x y y sin cos * / * x 0 y y sin cos * * x y y sin sin ~ y cos 1 * * * 1 y sin cos * + * - y y sin cos * y y sin cos * * / * - x y y sin cos * / x y y sin cos * / * / (postfix) ->  (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                             //0 x y y sin cos * / * x 0 y y sin cos * * x y y sin sin ~ y cos 1 * * * y sin cos + * - y y sin cos * y y sin cos * * / * - x y y sin cos * / x y y sin cos * / * / (postfix) ->  (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                             //0 x 0 x y y sin sin ~ y cos 1 * * * y sin cos + * - y y sin cos * y y sin cos * * / * - x y y sin cos * / x y y sin cos * / * / (postfix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                             //x x y y sin sin ~ y cos 1 * * * y sin cos + * ~ y y sin cos * y y sin cos * * / * ~ x y y sin cos * / x y y sin cos * / * / (postfix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                             //x x y y cos 1 * y sin sin ~ * * y sin cos + * ~ y y sin cos * y y sin cos * * / * ~ x y y sin cos * / x y y sin cos * / * / (postfix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                             //x x y y cos y sin sin ~ * * y sin cos + * ~ y y sin cos * y y sin cos * * / * ~ x y y sin cos * / x y y sin cos * / * / (postfix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "~", "~", "sin", "y", "/"}; // sin(x)/y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x ~ ~ cos 0 ~ ~ * y * x ~ ~ sin 1 * - y y * / (postfix) -> -sin(x)/(y*y) ✅
                                                             //x ~ ~ cos 0 ~ ~ * y * x ~ ~ sin - y y * / (postfix) -> -sin(x)/(y*y) ✅
                                                             //x ~ ~ cos 0 * y * x ~ ~ sin - y y * / (postfix) -> -sin(x)/(y*y) ✅
                                                             //x ~ ~ sin ~ y y * / (postfix) -> -sin(x)/(y*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "sqrt"}; //sqrt(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 2 x sqrt * / (postfix) -> 1/(2*sqrt(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "sqrt", "y", "*"}; //sqrt(x)*y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sqrt 0 * 1 2 x sqrt * / y * + (postfix) -> (1/(2*sqrt(x)))*y ✅
                                                             //0 1 2 x sqrt * / y * + (postfix) -> (1/(2*sqrt(x)))*y ✅
                                                             //1 2 x sqrt * / y * (postfix) -> (1/(2*sqrt(x)))*y
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "ln", "y", "*"}; //ln(x)*y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x ln 0 * 1 x / y * + (postfix) -> (1/x)*y ✅
                                                             //0 1 x / y * + (postfix) -> (1/x)*y ✅
                                                             //1 x / y * (postfix) -> (1/x)*y ✅
    
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "~", "ln", "x", "*"}; //ln(-x) * x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x ~ ln 1 * 1 ~ x ~ / x * + (postfix) -> ln(-x) + 1 ✅
                                                             //x ~ ln 1 ~ x ~ / x * + (postfix) -> ln(-x) + 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "sqrt", "ln", "y", "*"}; //ln(sqrt(x)) * y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sqrt ln 0 * 1 2 x sqrt * / x sqrt / y * + (postfix) -> y/(2*x) ✅
                                                             //0 1 2 x sqrt * / x sqrt / y * + (postfix) -> y/(2*x) ✅
                                                             //1 2 x sqrt * / x sqrt / y * (postfix) -> y/(2*x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "x", "*", "asin"}; //arcsin(x*x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x 1 * 1 x * + 1 x x * x x * * - sqrt / (postfix) -> (2*x)/sqrt(1-x*x*x*x) ✅
                                                             //x 1 x * + 1 x x * x x * * - sqrt / (postfix) -> (2*x)/sqrt(1-x*x*x*x) ✅
                                                             //x x + 1 x x * x x * * - sqrt / (postfix) -> (2*x)/sqrt(1-x*x*x*x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "ln", "y", "*", "asin"}; //arcsin(ln(x)*y)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x ln 1 * 0 x / y * + 1 x ln y * x ln y * * - sqrt / (postfix) -> ln(x) / sqrt(1-ln(x)*y*ln(x)*y) ✅
                                                             //x ln 0 x / y * + 1 x ln y * x ln y * * - sqrt / (postfix) -> ln(x) / sqrt(1-ln(x)*y*ln(x)*y) ✅
                                                             //x ln 0 x / + 1 x ln y * x ln y * * - sqrt / (postfix) -> ln(x) / sqrt(1-ln(x)*y*ln(x)*y) ✅
                                                             //x ln 1 x ln y * x ln y * * - sqrt / (postfix) -> ln(x) / sqrt(1-ln(x)*y*ln(x)*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "ln", "y", "*", "asin"}; //arcsin(ln(x)*y)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x ln 0 * 1 x / y * + 1 x ln y * x ln y * * - sqrt / (postfix) -> (y/x)/sqrt(1-y*ln(x)*y*ln(x)) ✅
                                                             //0 1 x / y * + 1 x ln y * x ln y * * - sqrt / (postfix) -> (y/x)/sqrt(1-y*ln(x)*y*ln(x)) ✅
                                                             //1 x / y * 1 x ln y * x ln y * * - sqrt / (postfix) -> (y/x)/sqrt(1-y*ln(x)*y*ln(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "acos", "y", "/", "asin"}; //arcsin(acos(x)/y)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 1 x x * - sqrt / ~ y * x acos 0 * - y y * / 1 x acos y / x acos y / * - sqrt / (postfix) -> (-y/sqrt(1-x*x))/(y*y) * (1/sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) = -1/(y*sqrt(1-x*x)*sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) ✅
                                                             //1 1 x x * - sqrt / ~ y * 0 - y y * / 1 x acos y / x acos y / * - sqrt / (postfix) -> (-y/sqrt(1-x*x))/(y*y) * (1/sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) = -1/(y*sqrt(1-x*x)*sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) ✅
                                                             //1 1 x x * - sqrt / ~ y * y y * / 1 x acos y / x acos y / * - sqrt / (postfix) -> (-y/sqrt(1-x*x))/(y*y) * (1/sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) = -1/(y*sqrt(1-x*x)*sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "ln", "y", "*", "asin", "y", "acos", "+"}; //arcsin(ln(x)*y)+acos(y)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x ln 1 * 0 x / y * + 1 x ln y * x ln y * * - sqrt / 1 1 y y * - sqrt / ~ + (postfix) -> (ln(x)/sqrt(1-ln(x)*y*ln(x)*y)) + (-1/sqrt(1-y*y)) ✅
                                                             //x ln 0 x / y * + 1 x ln y * x ln y * * - sqrt / 1 1 y y * - sqrt / ~ + (postfix) -> (ln(x)/sqrt(1-ln(x)*y*ln(x)*y)) + (-1/sqrt(1-y*y)) ✅
                                                             //x ln 0 x / + 1 x ln y * x ln y * * - sqrt / 1 1 y y * - sqrt / ~ + (postfix) -> (ln(x)/sqrt(1-ln(x)*y*ln(x)*y)) + (-1/sqrt(1-y*y)) ✅
                                                             //x ln 1 x ln y * x ln y * * - sqrt / 1 1 y y * - sqrt / ~ + (postfix) -> (ln(x)/sqrt(1-ln(x)*y*ln(x)*y)) + (-1/sqrt(1-y*y)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "acos", "acos", "x", "~", "*", "acos"}; //arccos(arccos(arccos(x))*-x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x acos acos 1 ~ * 1 1 x x * - sqrt / ~ 1 x acos x acos * - sqrt / ~ x ~ * + 1 x acos acos x ~ * x acos acos x ~ * * - sqrt / ~ (postfix) -> (arccos(arccos(x)) + (x*((1/sqrt(1-x*x))/sqrt(1-acos(x)*acos(x))))) / sqrt(1-(-x*arccos(arccos(x))*-x*arccos(arccos(x)))) = (arccos(arccos(x)) + x/(sqrt(1-x*x)*sqrt(1-acos(x)*acos(x)))) / sqrt(1-(x*arccos(arccos(x))*x*arccos(arccos(x)))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "exp", "x", "cos", "exp", "/"}; //exp(x) / exp(cos(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 x exp * x cos exp * x exp x sin ~ 1 * x cos exp * * - x cos exp x cos exp * / (postfix) -> (exp(x)*exp(cos(x)) - exp(x)*-sin(x)*exp(cos(x))) / (exp(cos(x))*exp(cos(x))) = (exp(x) - exp(x)*-sin(x)) / (exp(cos(x))) = (exp(x)*(1+sin(x))) / exp(cos(x)) = exp(x-cos(x))*(1+sin(x)) ✅
                                                             //x exp x cos exp * x exp x sin ~ 1 * x cos exp * * - x cos exp x cos exp * / (postfix) -> (exp(x)*exp(cos(x)) - exp(x)*-sin(x)*exp(cos(x))) / (exp(cos(x))*exp(cos(x))) = (exp(x) - exp(x)*-sin(x)) / (exp(cos(x))) = (exp(x)*(1+sin(x))) / exp(cos(x)) = exp(x-cos(x))*(1+sin(x)) ✅
                                                             //x exp x cos exp * x exp x sin ~ x cos exp * * - x cos exp x cos exp * / (postfix) -> (exp(x)*exp(cos(x)) - exp(x)*-sin(x)*exp(cos(x))) / (exp(cos(x))*exp(cos(x))) = (exp(x) - exp(x)*-sin(x)) / (exp(cos(x))) = (exp(x)*(1+sin(x))) / exp(cos(x)) = exp(x-cos(x))*(1+sin(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "~", "exp", "x", "x", "y", "*", "*", "+"}; //exp(-x) + x*y*x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 ~ x ~ exp * x x 0 * 1 y * + * 1 x y * * + + (postfix) -> -exp(-x) + x*y + x*y = -exp(-x) + 2*x*y ✅
                                                             //1 ~ x ~ exp * x 0 1 y * + * 1 x y * * + + (postfix) -> -exp(-x) + 2*x*y ✅
                                                             //1 ~ x ~ exp * x 0 y + * x y * + + (postfix) -> -exp(-x) + 2*x*y ✅
                                                             //1 ~ x ~ exp * x y * x y * + + (postfix) -> -exp(-x) + 2*x*y ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"y", "arcsin", "exp", "x", "~", "*", "acos"}; //arccos(exp(arcsin(y))*-x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //y arcsin exp 0 ~ * 1 1 y y * - sqrt / y arcsin exp * x ~ * + 1 y arcsin exp x ~ * y arcsin exp x ~ * * - sqrt / ~ (postfix) -> ((x/sqrt(1-y*y))*exp(arcsin(y))) / sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y)))) = (x/(sqrt(1-y*y)*sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y))))))*exp(arcsin(y)) = (x*exp(arcsin(y)))/(sqrt(1-y*y)*sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y))))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    postfix = {"x", "y", "^"}; //x ^ y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x y ^ x ln 1 * 0 x / y * + * (postfix) -> (x^y)*ln(x) ✅
                                                             //x y ^ x ln 0 x / y * + * (postfix) -> (x^y)*ln(x) ✅
                                                             //x y ^ x ln 0 x / + * (postfix) -> (x^y)*ln(x) ✅
                                                             //x y ^ x ln * (postfix) -> (x^y)*ln(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "cos", "y", "cos", "^", "x", "*"}; //(cos(x)^(cos(y)))*x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x cos y cos ^ 0 * x cos y cos ^ x cos ln y sin ~ 1 * * x sin ~ 0 * x cos / y cos * + * x * + (postfix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
                                                             //0 x cos y cos ^ x cos ln y sin ~ 1 * * x sin ~ 0 * x cos / y cos * + * x * + (postfix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
                                                             //x cos y cos ^ x cos ln y sin ~ 1 * * x sin ~ 0 * x cos / y cos * + * x * (postfix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
                                                             //x cos y cos ^ x cos ln y sin ~ * * x * (postfix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "cos", "y", "cos", "^", "x", "*"}; //(cos(x)^(cos(y)))*x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x cos y cos ^ 1 * x cos y cos ^ x cos ln y sin ~ 0 * * x sin ~ 1 * x cos / y cos * + * x * + (postfix) -> cos(x)^(cos(y)) + x*cos(x)^(cos(y)) * ((-sin(x)/cos(x))*cos(y)) = cos(x)^(cos(y)) - cos(y)*x*cos(x)^(cos(y)-1)*sin(x) ✅
                                                             //x cos y cos ^ x cos y cos ^ x cos ln y sin ~ 0 * * x sin ~ 1 * x cos / y cos * + * x * + (postfix) -> cos(x)^(cos(y)) + x*cos(x)^(cos(y)) * ((-sin(x)/cos(x))*cos(y)) = cos(x)^(cos(y)) - cos(y)*x*cos(x)^(cos(y)-1)*sin(x) ✅
                                                             //x cos y cos ^ x cos y cos ^ x sin ~ x cos / y cos * * x * + (postfix) -> cos(x)^(cos(y)) + x*cos(x)^(cos(y)) * ((-sin(x)/cos(x))*cos(y)) = cos(x)^(cos(y)) - cos(y)*x*cos(x)^(cos(y)-1)*sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "^", "x", "^", "y", "*"}; //((x^x)^x)*y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x x ^ x ^ 0 * x x ^ x ^ x x ^ ln 1 * x x ^ x ln 1 * 1 x / x * + * x x ^ / x * + * y * + (postfix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
                                                             //0 x x ^ x ^ x x ^ ln 1 * x x ^ x ln 1 * 1 x / x * + * x x ^ / x * + * y * + (postfix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
                                                             //0 x x ^ x ^ x x ^ ln x x ^ x ln 1 x / x * + * x x ^ / x * + * y * + (postfix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
                                                             //x x ^ x ^ x x ^ ln x x ^ x ln 1 x / x * + * x x ^ / x * + * y * (postfix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "^", "x", "^", "y", "*"}; //((x^x)^x)*y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x x ^ x ^ 1 * x x ^ x ^ x x ^ ln 0 * x x ^ x ln 0 * 0 x / x * + * x x ^ / x * + * y * + (postfix) -> ((x^x)^x) ✅
                                                             //x x ^ x ^ 1 * x x ^ x ^ 0 x x ^ 0 0 x / x * + * x x ^ / x * + * y * + (postfix) -> ((x^x)^x) ✅
                                                             //x x ^ x ^ x x ^ x ^ 0 x x ^ 0 0 x / x * + * x x ^ / x * + * y * + (postfix) -> ((x^x)^x) ✅
                                                             //x x ^ x ^ x x ^ x ^ 0 x x ^ 0 0 x / + * x x ^ / x * + * y * + (postfix) -> ((x^x)^x) ✅
                                                             //x x ^ x ^ x x ^ x ^ x x ^ 0 x / x * * x x ^ / x * * y * + (postfix) -> ((x^x)^x) ✅
                                                             //x x ^ x ^ (postfix) -> ((x^x)^x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
        
    postfix = {"x", "sech", "tanh", "x", "^", "y", "*"}; //tanh(sech(x))^x * y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sech tanh x ^ 0 * x sech tanh x ^ x sech tanh ln 1 * x sech sech x sech sech * x sech ~ x tanh * 1 * * x sech tanh / x * + * y * + (postfix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                             //0 x sech tanh x ^ x sech tanh ln 1 * x sech sech x sech sech * x sech ~ x tanh * 1 * * x sech tanh / x * + * y * + (postfix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                             //0 x sech tanh x ^ x sech tanh ln x sech sech x sech sech * x sech ~ x tanh * 1 * * x sech tanh / x * + * y * + (postfix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                             //x sech tanh x ^ x sech tanh ln x sech sech x sech sech * x sech ~ x tanh * 1 * * x sech tanh / x * + * y * (postfix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                             //x sech tanh x ^ x sech tanh ln x sech ~ x tanh * 1 * x sech sech x sech sech * * x sech tanh / x * + * y * (postfix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                             //x sech tanh x ^ x sech tanh ln 1 x sech ~ x tanh * * x sech sech x sech sech * * x sech tanh / x * + * y * (postfix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                             //x sech tanh x ^ x sech tanh ln x sech ~ x tanh * x sech sech x sech sech * * x sech tanh / x * + * y * (postfix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "y", "/", "tanh", "x", "sin", "^", "x", "*"}; //x*tanh(x/y)^(sin(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x y / tanh x sin ^ 1 * x y / tanh x sin ^ x y / tanh ln x cos 1 * * x y / sech x y / sech * 1 y * x 0 * - y y * / * x y / tanh / x sin * + * x * + (postfix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                             //x y / tanh x sin ^ x y / tanh x sin ^ x y / tanh ln x cos 1 * * x y / sech x y / sech * 1 y * x 0 * - y y * / * x y / tanh / x sin * + * x * + (postfix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                             //x y / tanh x sin ^ x y / tanh x sin ^ x y / tanh ln x cos 1 * * x y / sech x y / sech * y x 0 * - y y * / * x y / tanh / x sin * + * x * + (postfix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                             //x y / tanh x sin ^ x y / tanh x sin ^ x y / tanh ln x cos 1 * * x y / sech x y / sech * y 0 - y y * / * x y / tanh / x sin * + * x * + (postfix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                             //x y / tanh x sin ^ x y / tanh x sin ^ x y / tanh ln x cos 1 * * x y / sech x y / sech * y y y * / * x y / tanh / x sin * + * x * + (postfix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                             //x y / tanh x sin ^ x y / tanh x sin ^ x y / tanh ln x cos * x y / sech x y / sech * y y y * / * x y / tanh / x sin * + * x * + (postfix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                             //x y / tanh x sin ^ x y / tanh x sin ^ x y / tanh ln x cos * y y y * / x y / sech x y / sech * * x y / tanh / x sin * + * x * + (postfix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "sin", "sech", "x", "y", "*", "^", "sin", "sin", "sech"}; //sech(sin(sin( sech(sin(x))^(x*y))))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sin sech x y * ^ sin sin sech ~ x sin sech x y * ^ sin sin tanh * x sin sech x y * ^ sin cos x sin sech x y * ^ cos x sin sech x y * ^ x sin sech ln x 0 * 1 y * + * x sin sech ~ x sin tanh * x cos 1 * * x sin sech / x y * * + * * * * (postfix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                             //x sin sech x y * ^ sin sin sech ~ x sin sech x y * ^ sin sin tanh * x sin sech x y * ^ sin cos x sin sech x y * ^ cos x sin sech x y * ^ x sin sech ln 0 1 y * + * x sin sech ~ x sin tanh * x cos 1 * * x sin sech / x y * * + * * * * (postfix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                             //x sin sech x y * ^ sin sin sech ~ x sin sech x y * ^ sin sin tanh * x sin sech x y * ^ sin cos x sin sech x y * ^ cos x sin sech x y * ^ x sin sech ln 0 y + * x sin sech ~ x sin tanh * x cos 1 * * x sin sech / x y * * + * * * * (postfix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                             //x sin sech x y * ^ sin sin sech ~ x sin sech x y * ^ sin sin tanh * x sin sech x y * ^ sin cos x sin sech x y * ^ cos x sin sech x y * ^ x sin sech ln y * x sin sech ~ x sin tanh * x cos 1 * * x sin sech / x y * * + * * * * (postfix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                             //x sin sech x y * ^ sin sin sech ~ x sin sech x y * ^ sin sin tanh * x sin sech x y * ^ x sin sech ln y * x sin sech ~ x sin tanh * x cos * x sin sech / x y * * + * x sin sech x y * ^ cos * x sin sech x y * ^ sin cos * * (postfix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                             //x sin sech x y * ^ x sin sech ln y * x cos x sin sech ~ x sin tanh * * x sin sech / x y * * + * x sin sech x y * ^ cos * x sin sech x y * ^ sin cos * x sin sech x y * ^ sin sin sech ~ x sin sech x y * ^ sin sin tanh * * (postfix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "ln", "arccos", "x", "y", "*", "/", "sech", "~", "sin"}; //sin(-sech(arccos(ln(x))/(x*y)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x ln arccos x y * / sech ~ cos x ln arccos x y * / sech ~ x ln arccos x y * / tanh * 1 x / 1 x ln x ln * - sqrt / ~ x y * * x ln arccos x 0 * 1 y * + * - x y * x y * * / * ~ * (postfix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                             //x ln arccos x y * / sech ~ cos x ln arccos x y * / sech ~ x ln arccos x y * / tanh * 1 x / 1 x ln x ln * - sqrt / ~ x y * * x ln arccos 0 1 y * + * - x y * x y * * / * ~ * (postfix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                             //x ln arccos x y * / sech ~ cos x ln arccos x y * / sech ~ x ln arccos x y * / tanh * 1 x / 1 x ln x ln * - sqrt / ~ x y * * x ln arccos 0 y + * - x y * x y * * / * ~ * (postfix) -> (postfix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                             //x ln arccos x y * / sech ~ cos x ln arccos x y * / sech ~ x ln arccos x y * / tanh * 1 x / 1 x ln x ln * - sqrt / ~ x y * * x ln arccos y * - x y * x y * * / * ~ * (postfix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                              //x ln arccos x y * / sech ~ x ln arccos x y * / tanh * 1 x / 1 x ln x ln * - sqrt / ~ x y * * x ln arccos y * - x y * x y * * / * ~ x ln arccos x y * / sech ~ cos * (postfix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                             //1 x / 1 x ln x ln * - sqrt / ~ x y * * x ln arccos y * - x y * x y * * / x ln arccos x y * / sech ~ x ln arccos x y * / tanh * * ~ x ln arccos x y * / sech ~ cos * (postfix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
    
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
        
    postfix = {"0", "x", "*"}; //0*x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 0 x * + (postfix) -> 0 ✅
                                                             //0 0 + (postfix) -> 0 ✅
                                                             //0
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"0", "x", "*", "x", "x", "sin", "+", "-"}; //0*x - (x+sin(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 x cos 1 * + ~ (postfix) -> -1-cos(x) ✅
                                                             //1 x cos 1 * + ~ (postfix) -> -1-cos(x) ✅
                                                             //0 0 + 1 x cos 1 * + - (postfix) -> -1-cos(x) ✅
                                                             //1 x cos 1 * + ~ (postfix) -> -1-cos(x) ✅
                                                             //1 x cos + ~ (postfix) -> -1-cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"0", "x", "*", "~", "x", "tanh", "+"}; //-(0*x) + tanh(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sech x sech * 1 * (postfix) -> sech(x)*sech(x) ✅
                                                             //x sech x sech * 1 * (postfix) -> sech(x)*sech(x) ✅
                                                             //0 0 + ~ x sech x sech * 1 * + (postfix) -> sech(x)*sech(x) ✅
                                                             //0 ~ x sech x sech * 1 * + (postfix) -> sech(x)*sech(x) ✅
                                                             //0 ~ 1 x sech x sech * * + (postfix) -> sech(x)*sech(x) ✅
                                                             //0 ~ x sech x sech * + (postfix) -> sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"1", "x", "*"}; //1*x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 0 x * + (postfix) -> 1 ✅
                                                             //1 0 + (postfix) -> 1 ✅
                                                             //1 (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"1", "x", "*", "x", "x", "sin", "+", "-"}; //1*x - (x+sin(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 0 x * + 1 x cos 1 * + - (postfix) -> 1 - (1+cos(x)) ✅
                                                             //1 0 + 1 x cos 1 * + - (postfix) -> 1 - (1+cos(x)) ✅
                                                             //1 1 x cos 1 * + - (postfix) -> 1 - (1+cos(x)) ✅
                                                             //1 1 x cos + - (postfix) -> 1 - (1+cos(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"1", "x", "*", "~", "x", "tanh", "+"}; //-(1*x) + tanh(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 0 x * + ~ x sech x sech * 1 * + (postfix) -> -1 + sech(x)*sech(x) ✅
                                                             //1 0 + ~ x sech x sech * 1 * + (postfix) -> -1 + sech(x)*sech(x) ✅
                                                             //1 ~ x sech x sech * 1 * + (postfix) -> -1 + sech(x)*sech(x) ✅
                                                             //1 ~ 1 x sech x sech * * + (postfix) -> -1 + sech(x)*sech(x) ✅
                                                             //1 ~ x sech x sech * + (postfix) -> -1 + sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
        
    postfix = {"x", "0", "*"}; //x*0
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 0 + (postfix) -> 0 ✅
                                                             //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "0", "*", "x", "x", "sin", "+", "-"}; //x*0 - (x+sin(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 x cos 1 * + ~ (postfix) -> -1-cos(x) ✅
                                                             //0 0 + 1 x cos 1 * + - (postfix) -> -1-cos(x) ✅
                                                             //1 x cos 1 * + ~ (postfix) -> -1-cos(x) ✅
                                                             //1 x cos + ~ (postfix) -> -1-cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "0", "*", "~", "x", "tanh", "+"}; //-(x*0) + tanh(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sech x sech * 1 * (postfix) -> sech(x)*sech(x) ✅
                                                             //0 0 + ~ x sech x sech * 1 * + (postfix) -> sech(x)*sech(x) ✅
                                                             //0 ~ x sech x sech * 1 * + (postfix) -> sech(x)*sech(x) ✅
                                                             //0 ~ 1 x sech x sech * * + (postfix) -> sech(x)*sech(x) ✅
                                                             //0 ~ x sech x sech * + (postfix) -> sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "*", "1", "*"}; //x*x*1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x x +  (postfix) -> 2*x ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "sin", "+", "1", "*"}; //(sin(x)+x)*1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 x cos 1 * + (postfix) -> cos(x)*1 + 1 ✅
                                                             //1 x cos + (postfix) -> cos(x) + 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "1", "*", "tanh", "~", "1", "*", "1", "+"}; //-tanh(x*1)*1 + 1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x 1 * sech x 1 * sech * 1 * ~ (postfix) -> -sech(x)*sech(x) ✅
                                                             //1 x 1 * sech x 1 * sech * * ~ (postfix) -> -sech(x)*sech(x) ✅
                                                             //x 1 * sech x 1 * sech * ~ (postfix) -> -sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "sin", "x", "sin", "-", "x", "sin", "+"}; //sin(x) - sin(x) + sin(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp);  //x cos 1 * (postfix) -> cos(x) ✅
                                                              //x cos (postfix) -> cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
        
    postfix = {"x", "1", "/"}; //x/1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 x 0 * - 1 1 * / (postfix) -> 1/1 ✅
                                                             //1 0 - 1 1 * / (postfix) -> 1/1 ✅
                                                             //1 0 - (postfix) -> 1/1 ✅
                                                             //1 (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "*", "1", "/"}; //(x*x)/1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x x + x x * 0 * - 1 1 * / (postfix) -> (2*x)/1 ✅
                                                             //x x + 0 - 1 1 * / (postfix) -> (2*x)/1 ✅
                                                             //x x + 0 - (postfix) -> x+x ✅
                                                             //x x + (postfix) -> x+x ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "cos", "*", "1", "/"}; //(x*cos(x))/1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x x sin ~ 1 * * x cos + x x cos * 0 * - 1 1 * / (postfix) -> (-sin(x)*x + cos(x))/1 ✅
                                                             //x x sin ~ 1 * * x cos + 0 - 1 1 * / (postfix) -> (-sin(x)*x + cos(x))/1 ✅
                                                             //x x sin ~ 1 * * x cos + 0 - (postfix) -> (-sin(x)*x + cos(x)) ✅
                                                             //x x sin ~ 1 * * x cos + (postfix) -> (-sin(x)*x + cos(x)) ✅
                                                             //x x sin ~ * x cos + (postfix) -> (-sin(x)*x + cos(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"0", "x", "x", "*", "/"}; //0/(x*x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 0 - x x * x x * * / (postfix) -> 0/(x*x*x*x) ✅
                                                             //0 x x * x x * * / (postfix) -> 0/(x*x*x*x) ✅
                                                             //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"0", "x", "x", "cos", "*", "/"}; //0/(x*cos(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 0 - x x cos * x x cos * * / (postfix) -> 0/(x*cos(x)*x*cos(x)) ✅
                                                             //0 x x cos * x x cos * * / (postfix) -> 0/(x*cos(x)*x*cos(x)) ✅
                                                             //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"0", "x", "sin", "x", "sech", "*", "/"}; //0/(sin(x)*sech(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 0 - x sin x sech * x sin x sech * * / (postfix) -> 0/(sin(x)*sech(x)*sin(x)*sech(x)) ✅
                                                             //0 x sin x sech * x sin x sech * * / (postfix) -> 0/(sin(x)*sech(x)*sin(x)*sech(x)) ✅
                                                             //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"1", "x", "x", "*", "/"}; //1/(x*x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 x x + - x x * x x * * / (postfix) -> -2/(x*x*x) ✅
                                                             //x x + ~ x x * x x * * / (postfix) -> -2/(x*x*x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"1", "x", "cos", "/"}; //1/cos(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 x sin ~ 1 * - x cos x cos * / (postfix) -> sin(x)/(cos(x)*cos(x)) ✅
                                                             //x sin ~ 1 * ~ x cos x cos * / (postfix) -> sin(x)/(cos(x)*cos(x)) ✅
                                                             //x sin ~ ~ x cos x cos * / (postfix) -> sin(x)/(cos(x)*cos(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"1", "x", "cos", "x", "sin", "*", "/"}; //1/(cos(x)*sin(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp);  //0 x cos x cos 1 * * x sin ~ 1 * x sin * + - x cos x sin * x cos x sin * * / (postfix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                              //x cos x cos 1 * * x sin ~ 1 * x sin * + ~ x cos x sin * x cos x sin * * / (postfix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                              //x cos x cos 1 * * x sin ~ x sin * + ~ x cos x sin * x cos x sin * * / (postfix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                              //x cos x cos * x sin ~ x sin * + ~ x cos x sin * x cos x sin * * / (postfix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "~", "~", "sin", "+"}; //x + sin(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 x ~ ~ cos 1 * + (postfix) -> 1 + 1*cos(x) ✅
                                                             //1 x ~ ~ cos + (postfix) -> 1 + cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "~", "~", "tanh", "x", "-"}; //tanh(x) - x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x ~ ~ sech x ~ ~ sech * 1 * 1 - (postfix) -> sech(x)*sech(x) - 1 ✅
                                                             //1 x ~ ~ sech x ~ ~ sech * * 1 - (postfix) -> sech(x)*sech(x) - 1 ✅
                                                             //x ~ ~ sech x ~ ~ sech * 1 - (postfix) -> sech(x)*sech(x) - 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "0", "x", "^", "+"}; //x + 0^x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 0 0 ln 0 0 / x * + * + (postfix) -> 1 ✅
                                                             //1 (postfix) -> 1 ✅

    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"0", "x", "^", "x", "-"}; //0^x - x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x ~ ~ sech x ~ ~ sech * 1 * 1 - (postfix) -> -1 ✅
                                                             //1 ~ (postfix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"x", "cos", "0", "x", "^", "-"}; //cos(x) - 0^x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sin ~ 1 * 0 0 ln 0 0 / x * + * - (postfix) -> -sin(x) ✅
                                                             //x sin ~ 1 * (postfix) -> -sin(x) ✅
                                                             //x sin ~ (postfix) -> -sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "0", "^", "+"}; //x + x^0
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 1 0 * + (postfix) -> 1 ✅
                                                             //1 (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"x", "0", "^", "x", "-"}; //x^0 - x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 0 * 1 - (postfix) -> -1 ✅
                                                             //1 ~ (postfix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"x", "cos", "x", "0", "^", "-"}; //cos(x) - x^0
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sin ~ 1 * 1 0 * - (postfix) -> -sin(x) ✅
                                                             //x sin ~ 1 * (postfix) -> -sin(x) ✅
                                                             //x sin ~ (postfix) -> -sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "1", "x", "^", "+"}; //x + 1^x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 0 0 ln 0 0 / x * + * + (postfix) -> 1 ✅
                                                             //1 (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"1", "x", "^", "x", "-"}; //1^x - x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x ~ ~ sech x ~ ~ sech * 1 * 1 - (postfix) -> -1 ✅
                                                             //1 ~ (postfix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"x", "cos", "1", "x", "^", "-"}; //cos(x) - 1^x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sin ~ 1 * 0 0 ln 0 0 / x * + * - (postfix) -> -sin(x) ✅
                                                             //x sin ~ (postfix) -> -sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "1", "^", "+"}; //x + x^1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 x 1 x / * + (postfix) -> 2 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"x", "1", "^", "x", "-"}; //x^1 - x
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x 1 x / * 1 - (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");

    postfix = {"x", "cos", "x", "1", "^", "-"}; //cos(x) - x^1
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x sin ~ 1 * x 1 x / * - (postfix) -> -sin(x) - 1 ✅
                                                             //x sin ~ x 1 x / * - (postfix) -> -sin(x) - 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
        
    postfix = {"1", "x", "exp", "*", "ln"}; //ln(1*exp(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //1 (postfix) -> 1 ✅
                                                             //x exp 1 x exp * / (postfix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "1", "x", "exp", "*", "ln", "-"}; //x - ln(1*exp(x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
                                                             //1 x exp 1 x exp * / - (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "1", "x", "exp", "*", "ln", "-", "cos"}; //cos(x - ln(1*exp(x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x 1 x exp * ln - sin ~ 0 * (postfix) -> 0 ✅
                                                             //x 1 x exp * ln - sin ~ 1 x exp 1 x exp * / - * (postfix) -> 0 ✅
                                                             //1 x exp 1 x exp * / - x 1 x exp * ln - sin ~ * (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"y", "y", "*", "exp", "ln"}; //ln(exp(y*y))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x", "x", "*", "exp", "y", "ln", "*"}; //exp(x*x)*ln(y)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x x * exp 1 y / * (postfix) -> exp(x*x)*(1/y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"y", "y", "-", "exp", "exp", "x", "sin", "+"}; //exp(exp(y-y))+sin(x)
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x cos 1 * (postfix) -> cos(x) ✅
                                                             //x cos (postfix) -> cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
        
    postfix = {"x", "x", "*", "sin", "y", "/"}; // sin(x*x)/y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x x * sin ~ y y * / (postfix) -> -sin(x*x)/(y*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "cos", "sin", "y", "/"}; // sin(cos(x))/y
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "y", postfix, grasp); //x cos sin ~ y y * / (postfix) -> -sin(cos(x))/(y*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "-", "sqrt", "cos"}; // cos(sqrt(x-x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "*", "x", "x", "*", "-", "sqrt", "tanh", "sin"}; // sin(tanh(sqrt(x*x - x*x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x x * x x * - sqrt sech x x * x x * - sqrt sech * 0 * x x * x x * - sqrt tanh cos * (postfix) -> 0 ✅
                                                             //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "cos", "*", "x", "x", "cos", "*", "-", "sqrt", "sqrt"}; // sqrt(sqrt(x*cos(x) - x*cos(x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
        
    postfix = {"x", "x", "-", "arcsin", "cos"}; // cos(arcsin(x-x))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "^", "x", "x", "^", "-", "asin", "tanh", "sin"}; // sin(tanh(arcsin(x^x - x^x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x x ^ x x ^ - asin sech x x ^ x x ^ - asin sech * 0 * x x ^ x x ^ - asin tanh cos * (postfix) -> 0 ✅
                                                             //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "sin", "*", "x", "x", "sin", "*", "-", "arcsin", "asin"}; // arcsin(arcsin(x*sin(x) - x*sin(x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
        
    postfix = {"x", "tanh", "x", "tanh", "-", "acos", "exp"}; // exp(acos(tanh(x)-tanh(x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "/", "x", "x", "/", "-", "arccos", "sech", "sech"}; // sech(sech(arccos(x/x - x/x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //x x / x x / - arccos sech sech ~ x x / x x / - arccos sech tanh * x x / x x / - arccos sech ~ x x / x x / - arccos tanh * 0 * * (postfix) -> 0 ✅
                                                             //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "sech", "-", "x", "x", "sech", "-", "-", "arccos", "acos"}; // acos(arccos((x-sech(x)) - (x-sech(x))))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "exp", "*", "x", "x", "exp", "*", "-", "tanh", "acos"}; // acos(tanh(x*exp(x) - x*exp(x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
        
    postfix = {"x", "x", "exp", "*", "x", "x", "exp", "*", "-", "sech", "asin"}; // asin(sech(x*exp(x) - x*exp(x)))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x", "x", "sech", "-", "x", "x", "sech", "-", "-", "sech", "acos"}; // acos(sech((x-sech(x)) - (x-sech(x))))
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x", postfix, grasp); //0 (postfix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    postfix = {"x0", "x0", "cos", "/", "tanh", "acos", "cos"};
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x0", postfix, grasp);
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = derivat;
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x0", postfix, grasp);
    std::cout << grasp << '\n';
    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    postfix = {"x1", "x2", "+", "cos", "x1", "x2", "+", "+"};
    std::cout << "postfix = " << postfix << '\n'; derivePostfix(0, postfix.size()-1, "x1", postfix, grasp);
    std::cout << "postfix = " << postfix << '\n';
    std::cout << grasp << '\n';
    std::cout << "LGBs = " << getLGBs(postfix) << '\n';

    sout << derivat; std::cout << "derivat = {" << derivat << "}\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PostfixDifferentiationSymbolic.cpp")); } sout.str("");
    
    return 0;
}

//g++ -std=c++20 -o PostfixDifferentiationSymbolic PostfixDifferentiationSymbolic.cpp
