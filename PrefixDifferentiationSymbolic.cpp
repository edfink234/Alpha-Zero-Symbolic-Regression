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
// ✅     ✅      ✅     ✅     ✅     ✅     ✅     ✅    ✅     ✅
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

//Function to compute the RGB, from https://www.jstor.org/stable/43998756 (top of pg. 165), modified for prefix
void RGB(int z, int& ind, const std::vector<std::string>& prefix)
{
    do
    {
        ++ind;
        if (is_unary(prefix[ind]))
        {
            RGB(1, ind, prefix);
        }
        else if (is_binary(prefix[ind]))
        {
            RGB(2, ind, prefix);
        }
        --z;
    } while (z);
}

std::vector<int> getRGBs(const std::vector<std::string>& prefix)
{
    std::vector<int> temp(prefix.size());
    for (size_t k = 0; k < prefix.size(); ++k)
    {
        int start = k;
        int& ptr_lgb = start;
        if (is_unary(prefix[k]))
        {
            RGB(1, ptr_lgb, prefix);
        }
        else if (is_binary(prefix[k]))
        {
            RGB(2, ptr_lgb, prefix);
        }
        temp[k]=(ptr_lgb);
    }
    return temp;
}

//Computes the grasp of an arbitrary element prefix[i], from https://www.jstor.org/stable/43998756 (bottom of pg. 165)
int GR(int i, const std::vector<std::string>& prefix)
{
    int start = i;
    int& ptr_lgb = start;
    if (is_unary(prefix[i]))
    {
        RGB(1, ptr_lgb, prefix);
    }
    else if (is_binary(prefix[i]))
    {
        RGB(2, ptr_lgb, prefix);
    }
    return (ptr_lgb - i);
}


void setGR(const std::vector<std::string>& prefix, std::vector<int>& grasp)
{
    grasp.reserve(prefix.size());
    for (size_t k = 0; k < prefix.size(); ++k)
    {
        grasp.push_back(GR(k, prefix));
        // printf("%d ",grasp[k]);
    }
    // puts("");
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
low and up: lower and upper Index bounds, respectively, for the piece of the array prefix which is to be the subject of the processing.
dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
*/
void derivePrefixHelper(int low, int up, const std::string& dx, const std::vector<std::string>& prefix, std::vector<int>& grasp, bool setGRvar = false, bool trace_derivat = false)
{
    if (!setGRvar)
    {
        grasp.clear();
        derivat.clear();
        // std::cout << derivat.size();
        derivat.reserve(100);
//        Index = 0;
        setGR(prefix, grasp);
    }
    
    //allowed ops: +, -, *, /, ^, unary +, unary -, sin(), cos(), tan(), ctg(), log(), sqrt(), const, x0, x1, ..., x_numFeatures
    //Define `grasp` of prefix[i], i.e., the number of elements forming operands of prefix[i] (grasp(operand) = 0)
    //The grasped elements of prefix[i] are the elements forming operands of prefix[i]
    //The left-grasp-bound (LGB) of prefix[i] is the Index of the left-most grasped element of prefix[i] in the array prefix
    //For the expression formed by prefix[i] and its grasped elements, the element prefix[i] is termed the `head` or `main element`
    //ANY element prefix[i] is an N-arity operator acting on operands arg1, ..., argN, the heads of which are op1, ..., opN, where op1 is the left-neighest operator of prefix[i] in the array prefix (so operands are 0-arity operators)
    //For an N-arity operator prefix[i]:
        //The `grasp` of prefix[i]  is equal to i - LGB(prefix[i])
        //grasp(prefix[i]) = N + sum(grasp(op1), ..., grasp(op_k), ..., grasp(opN))
        //grasp(prefix[i]) = N + sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //LGB(prefix[i]) = i - N - sum( (1 - grasp(op1)), ..., (k - grasp(op_k)), ..., (N - grasp(opN)))
        //op_(N-j) = prefix[i - sum(grasp(op_(N-1)), ..., grasp(op_(N-j-1))) - j - 1], where j = 0, ..., N-1
    //If the grasp of an arbitrary prefix[i] is greater than N, then at least one of its argument heads is also an operator.
        //Example: If the grasp of any binary operator prefix[i] is greater than 2, then at least one of the two preceding elements in the RPN of the expression (prefix[i-1] and prefix[i-2]) is also an operator (unary or binary).
    //prefix[numElements] is certainly an operator (unary or binary)
    
    //if prefix[up] is a binary operator, then:
        //the head of its second argument (let's call it op2) is equal to prefix[up-1]
        //then the grasped elements of op2 are the elements from prefix[up-1-grasp[up-1]] to prefix[up-1]
            //e.g. prefix = {"x", "x", "*", "x", "cos", "x", "*", "+"}, up = 7 -> prefix[up] = "+" is binary
            //so prefix[up-1] = "*" is the head of the second argument of "+" and so the grasped elements
            //of prefix[up-1] are the elements [(prefix[up-1-grasp[up-1]] = prefix[6-3] = prefix[3]), prefix[up-1] = prefix[6]]
            //i.e., the elements {"x", "cos", "x", "*"}
        //the head of its first argument (lets call it op1) is equal to prefix[up-grasp(op2)-2] which is equal to prefix[up-2-grasp[up-1]].
        //then the grasped elements of op1 are the elements from prefix[low = 0] to prefix[up-2-grasp[up-1]]
            //e.g. prefix = {"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}, up = 9 ->prefix[up] = "+" is binary
            //so prefix[up-grasp(op2)-2] = prefix[9-5-2] = prefix[2] = "*" is the head of the first argument of "+" and so the grasped elements
            //of prefix[up-grasp(op2)-2] are the elements [(prefix[low] = prefix[0], prefix[up-grasp(op2)-2] = prefix[9-5-2] = prefix[2]]
            //i.e., the elements {"x", "x", "*"}

    if (trace_derivat)
    {
        std::cout << "derivat = {" << derivat << "}, low = " << low << ", up = " << up
        << ", prefix[up] = " << prefix[up] << ", prefix[low] = " << prefix[low] << ", prefix = " << prefix << '\n';
    }
    
    if (prefix[low] == "+" || prefix[low] == "-")
    {
        
        int op_idx = derivat.size();
        derivat.push_back(prefix[low]); //+/-

        int temp = low+1+grasp[low+1];
        int x_prime_low = derivat.size();
        derivePrefixHelper(low+1, temp, dx, prefix, grasp, true, trace_derivat);  /* +/- x' */
        int x_prime_high = derivat.size();
        derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true, trace_derivat); /* +/- x' y' */
        int y_prime_high = derivat.size();
        int step;
        
        /*
         Simplification cases:
            
          1.) y' == 0, +/-, x'
          2.) x' == 0,   +, y'
          3.) x' == 0,   -, ~ y'
         
         */
        
        if (derivat[x_prime_high] == "0") //1.) +/- x' 0 -> x'
        {
            //remove y'
            if (x_prime_high == static_cast<int>(derivat.size() - 1))
            {
                derivat.pop_back();
            }
            else
            {
                derivat.erase(derivat.begin() + x_prime_high, derivat.end());
            }
            derivat.erase(derivat.begin() + op_idx); //remove +/- operator at beginning
        }
        
        else if (derivat[x_prime_low] == "0") //2.) and 3.)
        {
//            puts("hi 180");
            if (prefix[low] == "+") //2.) + 0 y' -> y'
            {
                derivat.erase(derivat.begin() + op_idx, derivat.begin() + x_prime_high); //remove "+" and "x'"
            }
            else //3.) prefix[low] == "-", - 0 y' -> ~ y'
            {
//                puts("hi 187");
                derivat[op_idx] = "~"; //change binary minus to unary minus
                derivat.erase(derivat.begin() + x_prime_low); //remove x'
            }
        }
        else if ((prefix[low] == "-") && ((step = (y_prime_high - x_prime_high)) == (x_prime_high - x_prime_low)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
        {
//            puts("hi 194");
            assert(derivat[op_idx] == prefix[low]);
            derivat[op_idx] = "0"; //change "-" to "0";
            derivat.erase(derivat.begin() + op_idx + 1, derivat.begin() + y_prime_high);
        }
    }
    else if (prefix[low] == "*")
    {
        derivat.push_back("+"); /* +  */
        derivat.push_back("*"); /* + * */
        int x_low = derivat.size();
        int temp = low+1+grasp[low+1];
        for (int k = low+1; k <= temp; k++) /* + * x */
        {
            derivat.push_back(prefix[k]);
        }
        if (derivat[x_low] == "0") //* 0 y' -> 0
        {
//            puts("hi 209");
            derivat[x_low - 1] = "0"; //change "*" to "0"
            derivat.erase(derivat.begin() + x_low); //erase x
        }
        else
        {
            int y_prime_low = derivat.size();
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true, trace_derivat); /* + * x y' */
            
            if (derivat[y_prime_low] == "0") //* x 0 -> 0
            {
//                puts("hi 219");
                derivat[x_low - 1] = "0"; //change "*" to "0"
                derivat.erase(derivat.begin() + x_low, derivat.end()); //erase x and y'
            }
            else if (derivat[x_low] == "1") //* 1 y' -> y'
            {
//                puts("hi 225");
                derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + x_low + 1); //erase "*" and "1"
            }
            else if (derivat[y_prime_low] == "1") //* x 1 -> x
            {
//                puts("hi 230");
                derivat.pop_back(); //remove "1"
                derivat.erase(derivat.begin() + x_low - 1); //remove "*"
            }
        }
        derivat.push_back("*"); /* + * x y' * */
        int x_prime_low = derivat.size();
        derivePrefixHelper(low+1, temp, dx, prefix, grasp, true, trace_derivat); /* + * x y' * x' */
        if (derivat[x_prime_low] == "0") //* 0 y -> 0
        {
//            puts("hi 240");
            derivat[x_prime_low - 1] = "0"; //change "*" to "0"
            derivat.erase(derivat.begin() + x_prime_low); //erase x'
        }
        else
        {
            int y_low = derivat.size();
            for (int k = temp+1; k <= temp+1+grasp[temp+1]; k++)
            {
                derivat.push_back(prefix[k]); /* + * x y' * x' y */
            }
            if (derivat[y_low] == "0") //* x' 0 -> 0
            {
//                puts("hi 253");
                derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //erase x' and y
            }
            else if (derivat[x_prime_low] == "1") //* 1 y -> y
            {
//                puts("hi 259");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
            else if (derivat[y_low] == "1") //* x' 1 -> x'
            {
//                puts("hi 264");
                derivat.pop_back(); //remove "1"
                assert(derivat[x_prime_low - 1] == "*");
                derivat.erase(derivat.begin() + x_prime_low - 1); //remove "*"
            }
        }
        if (derivat[x_low - 1] == "0") //+ 0 * x' y -> * x' y
        {
//            puts("hi 272");
            derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //remove "+" and "0"
        }
        else if (derivat[x_prime_low - 1] == "0") //+ * x y' 0 -> * x y'
        {
//            puts("hi 277");
            assert(static_cast<int>(derivat.size()) == x_prime_low);
            derivat.erase(derivat.begin() + x_low - 2); //erase "+"
            derivat.pop_back(); //remove "0"
        }
    }
    
    else if (prefix[low] == "/")
    {
        int div_idx = derivat.size();
        derivat.push_back("/"); /* / */
        derivat.push_back("-"); /* / - */
        derivat.push_back("*"); /* / - * */
        int temp = low+1+grasp[low+1];
        int x_prime_low = derivat.size();
        int k;
        derivePrefixHelper(low+1, temp, dx, prefix, grasp, true, trace_derivat); /* / - * x' */
        if (derivat[x_prime_low] == "0") //* 0 y -> 0
        {
//            puts("hi 297");
            derivat[x_prime_low - 1] = "0"; //change "*" to "0"
            assert(x_prime_low + 1 == static_cast<int>(derivat.size()));
            derivat.pop_back(); //remove x', which is 0
        }
        else
        {
            int y_low = derivat.size();
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[y_low] == "0") //* x' 0 -> 0
            {
//                puts("hi 312");
                derivat[x_prime_low - 1] = "0"; //change "*" to "0"
                derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //remove x' and 0
            }
            else if (derivat[y_low] == "1") //* x' 1 -> x'
            {
//                puts("hi 318");
                assert(y_low == static_cast<int>(derivat.size() - 1));
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                derivat.pop_back(); //erase the "1"
            }
            else if (derivat[x_prime_low] == "1") //* 1 y -> y
            {
//                puts("hi 358");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        derivat.push_back("*"); /* / - * x' y * */
        int x_low = derivat.size();
        for (k = low+1; k <= temp; k++) /* / - * x' y * x */
        {
            derivat.push_back(prefix[k]);
        }
        if (derivat[x_low] == "0") //* 0 y' -> 0
        {
//            puts("hi 370");
            derivat.erase(derivat.begin() + x_low - 1); //erase "*"
        }
        else
        {
            int y_prime_low = derivat.size();
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true, trace_derivat); /* / - * x' y * x y' */
            if (derivat[y_prime_low] == "0") //* x 0 -> 0
            {
//                puts("hi 379");
                assert(y_prime_low == static_cast<int>(derivat.size() - 1));
                derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and x
            }
            else if (derivat[x_low] == "1") //* 1 y' -> y'
            {
//                puts("hi 385");
                derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and 1
            }
            else if (derivat[y_prime_low] == "1") //* x 1 -> x
            {
//                puts("hi 390");
                assert(y_prime_low == static_cast<int>(derivat.size() - 1));
                derivat.erase(derivat.begin() + x_low - 1); //erase "*"
                derivat.pop_back(); //remove the "1"
            }
        }
        
        if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - (x_low - 1))) && (areDerivatRangesEqual(x_prime_low - 1, x_low - 1, k))) //- thing1 thing1 -> 0
        {
//            puts("hi 399");
            derivat[div_idx] = "0";
            derivat.erase(derivat.begin() + div_idx + 1, derivat.end()); //erase everything else
        }
        else
        {
            if (derivat[x_prime_low - 1] == "0") //- 0 * x y' -> ~ * x y'
            {
                //puts("hi 407");
                derivat[x_prime_low - 2] = "~"; //change "-" to "~"
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "0"
            }
            else if (derivat[x_low - 1] == "0") //- * x' y 0 -> * x' y
            {
//                puts("hi 413");
                assert(static_cast<int>(derivat.size()) == x_low);
                derivat.erase(derivat.begin() + x_prime_low - 2); //erase the "-"
                derivat.pop_back(); //erase the "0"
            }
            derivat.push_back("*"); /* / - * x' y * x y' * */
            int y_low = derivat.size();
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[y_low] == "1") // / - * x' y * x y' * 1 1 ->  - * x' y * x y'
            {
//                puts("hi 426");
                assert(y_low == static_cast<int>(derivat.size() - 1));
                derivat.erase(derivat.begin() + y_low - 1); //erase "*"
                derivat.erase(derivat.begin() + div_idx); //erase "/"
                derivat.pop_back(); //erase "1"
            }
            else
            {
                for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y y */
                {
                    derivat.push_back(prefix[k]);
                }
            }
        }
    }
    
    else if (prefix[low] == "^")
    {
        derivat.push_back("*"); /* * */
        derivat.push_back("^"); /* * ^ */
        int temp = low+1+grasp[low+1];
        int k;
        int x_low = derivat.size();
        for (k = low+1; k <= temp; k++) /* * ^ x */
        {
            derivat.push_back(prefix[k]);
        }
        if (derivat[x_low] == "0") //* ^ 0 y (* ln 0 y)' -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
        {
//            puts("hi 454");
            assert(x_low == static_cast<int>(derivat.size() - 1));
            derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
            return;
        }
        else if (derivat[x_low] == "1") //* ^ 1 y (* ln 1 y)' -> 0 (because ln(1) is 0)
        {
//            puts("hi 461");
            assert(x_low == static_cast<int>(derivat.size() - 1));
            derivat[x_low] = "0"; //change "1" to "0"
            derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
            return;
        }
        int y_low = derivat.size();
        for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ^ x y */
        {
            derivat.push_back(prefix[k]);
        }
        if (derivat[y_low] == "0") //* ^ x 0 (* ln x 0)' -> 0
        {
            assert(y_low == static_cast<int>(derivat.size() - 1));
//            puts("hi 474");
            derivat[x_low - 2] = "0"; //change "*" to "0)
            derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
            return;
        }
        else if (derivat[y_low] == "1") //^ x 1 -> x
        {
            assert(y_low == static_cast<int>(derivat.size() - 1));
            derivat.pop_back(); //erase the "1"
            derivat.erase(derivat.begin() + x_low - 1); //erase the "*"
//            puts("hi 485");
        }
        std::vector<std::string> prefix_temp;
        std::vector<int> grasp_temp;
        size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> * ln x y
        prefix_temp.reserve(reserve_amount);
        grasp_temp.reserve(reserve_amount);
        prefix_temp.push_back("*"); /* * */
        prefix_temp.push_back("ln"); /* * ln */
        int x_temp_low = prefix_temp.size();
        for (k = low+1; k <= temp; k++) /* * ln x */
        {
            prefix_temp.push_back(prefix[k]);
        }
        y_low = prefix_temp.size();
        for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ln x y */
        {
            prefix_temp.push_back(prefix[k]);
        }
        if (prefix_temp[y_low] == "1") //* ln x 1 -> ln x
        {
//            puts("hi 506");
            assert(y_low == static_cast<int>(prefix_temp.size()) - 1);
            prefix_temp.pop_back(); //remove the "1"
            prefix_temp.erase(prefix_temp.begin() + x_temp_low - 2); //erase the "*"
        }
        setGR(prefix_temp, grasp_temp);
        int temp_term_low = derivat.size();
        derivePrefixHelper(0, prefix_temp.size() - 1, dx, prefix_temp, grasp_temp, true, trace_derivat); /* * ^ x y (* ln x y)' */

        if (derivat[temp_term_low] == "0") //* ^ x y 0 -> 0
        {
//            puts("hi 516");
            derivat[x_low - 2] = "0"; //changing "*" to "0"
            derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
        }
        else if (derivat[temp_term_low] == "1") //* ^ x y 1 -> ^ x y
        {
//            puts("hi 522");
            assert(temp_term_low == static_cast<int>(derivat.size() - 1));
            derivat.erase(derivat.begin() + x_low - 2); //erasing "*"
            derivat.pop_back(); //erasing the "1"
        }
    }

    else if (prefix[low] == "cos")
    {
        derivat.push_back("*"); /* * */
        int x_prime_low = derivat.size();
        int temp = low+1;
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); /* * x' */
        if (derivat[x_prime_low] == "0") //* 0 ~ sin x -> 0
        {
//            puts("hi 538");
            assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
            derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
            return;
        }
        derivat.push_back("~"); /* * x' ~ */
        derivat.push_back("sin"); /* * x' ~ sin */
        for (int k = temp; k <= temp+grasp[temp]; k++)
        {
            derivat.push_back(prefix[k]); /* * x' ~ sin x */
        }
        if (derivat[x_prime_low] == "1") //* 1 ~ sin x -> ~ sin x
        {
//            puts("hi 551");
            derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
        }
    }
    
    else if (prefix[low] == "sin")
    {
        derivat.push_back("*"); /* * */
        int x_prime_low = derivat.size();
        int temp = low+1;
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); /* * x' */
        if (derivat[x_prime_low] == "0") //* 0 cos x -> 0
        {
//            puts("hi 565");
            assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
            derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
            return;
        }
        derivat.push_back("cos"); /* * x' cos */
        for (int k = temp; k <= temp+grasp[temp]; k++)
        {
            derivat.push_back(prefix[k]); /* * x' cos x */
        }
        if (derivat[x_prime_low] == "1") //* 1 cos x -> cos x
        {
//            puts("hi 577");
            derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
        }
    }
    
    else if (prefix[low] == "sqrt")
    {
        derivat.push_back("/");         /* / */
        int temp = low+1;
        int x_prime_low = derivat.size();
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); /* / x' */
        if (derivat[x_prime_low] == "0")
        {
//            puts("hi 590");
            assert(x_prime_low == static_cast<int>(derivat.size() - 1));
            derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "/"
            return;
        }
        derivat.push_back("*");         /* / x' * */
        derivat.push_back("2");         /* / x' * 2 */
        derivat.push_back("sqrt");      /* / x' * 2 sqrt */
        for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' * 2 sqrt x */
        {
            derivat.push_back(prefix[k]);
        }
    }
    
    else if (prefix[low] == "log" || prefix[low] == "ln")
    {
        derivat.push_back("/");               /* / */
        int temp = low+1;
        int x_prime_low = derivat.size();
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); /* / x' */
        if (derivat[x_prime_low] == "0") // / 0 x -> 0
        {
//            puts("hi 578");
            assert(static_cast<int>(derivat.size()) - 1 == x_prime_low);
            derivat[x_prime_low - 1] = "0"; //change "/" to 0
            derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //delete the rest
            return;
        }
        int x_low = derivat.size();
        for (int k = temp; k <= temp+grasp[temp]; k++)
        {
            derivat.push_back(prefix[k]);      /* / x' x */
        }
        int step = derivat.size() - x_low;
        if ((step == (x_low - x_prime_low)) && areDerivatRangesEqual(x_prime_low, x_low, step)) // / something something -> 1
        {
//            puts("hi 591");
            derivat[x_prime_low - 1] = "1"; //change "/" to 0
            derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //delete the rest
        }
    }
    
    else if (prefix[low] == "asin" || prefix[low] == "arcsin")
    {
        derivat.push_back("/");   /* / */
        int temp = low+1;
        int x_prime_low = derivat.size();
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); /* / x' */
        if (derivat[x_prime_low] == "0")
        {
//            puts("hi 640");
            assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
            derivat.erase(derivat.begin() + x_prime_low - 1); //erase "/"
            return;
        }
        derivat.push_back("sqrt"); /* / x' sqrt */
        derivat.push_back("-");    /* / x' sqrt - */
        derivat.push_back("1");    /* / x' sqrt - 1 */
        derivat.push_back("*");    /* / x' sqrt - 1 * */
        for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x */
        {
            derivat.push_back(prefix[k]);
        }
        for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x x */
        {
            derivat.push_back(prefix[k]);
        }
    }
    
    else if (prefix[low] == "acos" || prefix[low] == "arccos")
    {
        derivat.push_back("~");   /* ~ */
        derivat.push_back("/");   /* ~ / */
        int temp = low+1;
        int x_prime_low = derivat.size();
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); /* ~ / x' */
        if (derivat[x_prime_low] == "0")
        {
//            puts("hi 668");
            assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
            derivat.erase(derivat.begin() + x_prime_low - 2, derivat.begin() + x_prime_low); //erase "~" and "/"
            return;
        }
        derivat.push_back("sqrt"); /* ~ / x' sqrt */
        derivat.push_back("-");    /* ~ / x' sqrt - */
        derivat.push_back("1");    /* ~ / x' sqrt - 1 */
        derivat.push_back("*");    /* ~ / x' sqrt - 1 * */
        for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x */
        {
            derivat.push_back(prefix[k]);
        }
        for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x x */
        {
            derivat.push_back(prefix[k]);
        }
    }
    
    else if (prefix[low] == "tanh")
    {
        derivat.push_back("*");      //*
        int x_prime_low = derivat.size();
        int temp = low+1;
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); //* x'
        if (derivat[x_prime_low] == "0")
        {
//            puts("hi 696");
            assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
            derivat.erase(derivat.begin() + x_prime_low - 1); //delete the "*"
            return;
        }
        derivat.push_back("*");      //* x' *
        derivat.push_back("sech");   //* x' * sech
        for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x
        {
            derivat.push_back(prefix[k]);
        }
        derivat.push_back("sech");   //* x' * sech x sech
        for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x sech x
        {
            derivat.push_back(prefix[k]);
        }
        if (derivat[x_prime_low] == "1") //* 1 * sech x sech x -> * sech x sech x
        {
//            puts("hi 715");
            derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
        }
    }
    
    else if (prefix[low] == "sech")
    {
        derivat.push_back("*"); //*
        int x_prime_low = derivat.size();
        int temp = low+1;
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); //* x'
        if (derivat[x_prime_low] == "0") //* 0 * ~ sech x tanh x -> 0
        {
//            puts("hi 722");
            assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
            derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "*"
            return;
        }
        derivat.push_back("*");      //* x' *
        derivat.push_back("~");      //* x' * ~
        derivat.push_back("sech");   //* x' * ~ sech
        for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x
        {
            derivat.push_back(prefix[k]);
        }
        derivat.push_back("tanh");   //* x' * ~ sech x tanh
        for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x tanh x
        {
            derivat.push_back(prefix[k]);
        }
        if (derivat[x_prime_low] == "1") //* 1 exp x -> exp x
        {
//            puts("hi 742");
            derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
        }
    }
    
    else if (prefix[low] == "exp")
    {
        derivat.push_back("*");               //*
        int temp = low+1;
        int x_prime_low = derivat.size();
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); //* x'
        if (derivat[x_prime_low] == "0") //* 0 exp x -> 0
        {
//            puts("hi 682");
            assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
            derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
            return;
        }
        derivat.push_back("exp");           //* x' exp
        for (int k = temp; k <= temp+grasp[temp]; k++)
        {
            derivat.push_back(prefix[k]);      //* x' exp x
        }
        if (derivat[x_prime_low] == "1") //* 1 exp x -> exp x
        {
//            puts("hi 694");
            derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
        }
    }
    
    else if (prefix[low] == "~")
    {
        int temp = low+1;
        int un_minus_idx = derivat.size();
        derivat.push_back(prefix[low]); /* ~ */
        int x_prime_low = derivat.size();
        derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true, trace_derivat); /* ~ x' */
        if (derivat[x_prime_low] == "~")
        {
//            puts("hi 590");
            derivat.erase(derivat.begin() + un_minus_idx, derivat.begin() + x_prime_low + 1); //erase the two "~"
        }
    }
    
    else
    {
        if (prefix[low] == dx)
        {
            derivat.push_back("1");
        }
        else
        {
            derivat.push_back("0");
        }
    }
}

void derivePrefix(int low, int up, const std::string& dx, const std::vector<std::string>& prefix, std::vector<int>& grasp, bool trace_derivat = false)
{
    derivePrefixHelper(low, up, dx, prefix, grasp, false, trace_derivat);
}


int main()
{
    std::vector<std::string> prefix; //array of prefix expression elements read from left to right
    std::vector<int> grasp;
    
    prefix = {"+", "-", "+", "x", "y", "z", "+", "-", "+", "x", "y", "z", "x"};
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp, true); //+ 1 + 1 1 (prefix) -> 1+1+1 = 3 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+","x","x"}; // x+x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ 1 1 (prefix) -> 1+1 = 2 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+","-","x","x","x"}; // (x-x)+x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ - 1 1 1 (prefix) -> (1-1)+1 = 0+1 = 1 ✅
                                                          //1 (prefix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+","-","-","x","x","x","y"}; // (x-x)-x+y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ - - 1 1 1 0 (prefix) -> (1-1)-1+0 = -1 ✅
                                                          //- - 1 1 1 (prefix) -> (1-1)-1 = -1 ✅
                                                          //~ 1 (prefix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+", "+", "x", "x", "y"}; //x + x + y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ + 1 1 0 (prefix) -> 1 + 1 ✅
                                                          //+ 1 1 (prefix) -> 1 + 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"-","y","x"}; //y - x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); //- 1 0 (prefix) -> 1 ✅
                                                          //1 (prefix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+","*","x","x","y"}; //x*x + y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ + * x 1 * 1 x 0 (prefix) -> x + x ✅
                                                          //+ * x 1 * 1 x (prefix) -> x + x ✅
                                                          //+ x * 1 x (prefix) -> x + x ✅
                                                          //+ x x (prefix) -> x + x ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "0", "x"}; //0*x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * 0 1 * 0 x (prefix) -> 0 ✅
                                                          //+ 0 * 0 x (prefix) -> 0 ✅
                                                          //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "*", "x", "x", "1"}; //x*x*1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ x x (prefix) -> x+x ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "x", "y"}; //x*y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //y ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"*", "x", "y"}; //x*y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); //x ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "x", "y"}; //x*y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "z", prefix, grasp); //0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "x", "0"}; //x+0
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"+", "x", "x"}; //x+x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ 1 1 (prefix) -> 1+1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"*", "1", "x"}; //1*x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"*", "x", "1"}; //x*1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"*", "0", "x"}; //0*x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+","cos","/","*","y","y","x", "y"}; // cos((y*y)/x) + y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ~ sin / * y y x / - * + * y 0 * 0 y x * * y y 1 * x x 0 (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
                                                          //* ~ sin / * y y x / - * + * y 0 * 0 y x * * y y 1 * x x (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
                                                          //* ~ sin / * y y x / - * + 0 * 0 y x * * y y 1 * x x (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
                                                          //* ~ sin / * y y x / - * 0 x * * y y 1 * x x (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
                                                          //* ~ sin / * y y x / - 0 * * y y 1 * x x (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
                                                          //* ~ sin / * y y x / - 0 * y y * x x (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
                                                          //* ~ sin / * y y x / ~ * y y * x x (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
                                                          //* / ~ * y y * x x ~ sin / * y y x (prefix) -> -sin(y*(y/x))*y*(-y)/(x*x)) (infix) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+", "cos", "*", "*","y","x","y","y"}; // cos((y*x)*y) + y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ~ sin * * y x y + * * y x 0 * + * y 1 * 0 x y 0  (prefix) -> -sin((y*x)*y)*y*y (infix) ✅
                                                          //* ~ sin * * y x y + * * y x 0 * + * y 1 * 0 x y (prefix) -> -sin((y*x)*y)*y*y (infix) ✅
                                                          //* ~ sin * * y x y + 0 * + * y 1 * 0 x y (prefix) -> -sin((y*x)*y)*y*y (infix) ✅
                                                          //* ~ sin * * y x y + 0 * + y * 0 x y (prefix) -> -sin((y*x)*y)*y*y (infix) ✅
                                                          //* ~ sin * * y x y * y y (prefix) -> -sin((y*x)*y)*y*y (infix) ✅
                                                          //* * y y ~ sin * * y x y (prefix) -> -sin((y*x)*y)*y*y (infix) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    

    prefix = {"+","+","cos","x","x","y"}; //cos(x) + x + y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ + * ~ sin x 1 1 0 (prefix) -> 1-sin(x) ✅
                                                          //+ * ~ sin x 1 1 (prefix) -> 1-sin(x) ✅
                                                          //+ ~ sin x 1 (prefix) -> 1-sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"-", "y", "+","cos","x","x"}; //y - (cos(x) + x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp);
    std::cout << grasp << '\n'; //- 1 + * ~ sin x 0 0 (prefix) -> 1 ✅
                                //- 1 * ~ sin x 0 (prefix) -> 1 ✅
                                //1 (prefix) -> 1 ✅
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "x", "cos", "cos", "-","y","x"}; //x * cos(cos(y-x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * x * ~ sin cos - y x * ~ sin - y x - 0 1 * 1 cos cos - y x (prefix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
                                                          //+ * x * ~ sin cos - y x * ~ sin - y x ~ 1 * 1 cos cos - y x (prefix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
                                                          //+ * x * ~ sin cos - y x * ~ sin - y x ~ 1 cos cos - y x (prefix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
                                                          //+ * x * * ~ 1 ~ sin - y x ~ sin cos - y x cos cos - y x (prefix) -> x*(-sin(cos(y-x))*sin(y-x))+cos(cos(y-x)) = x*sin(cos(x-y))*sin(x-y)+cos(cos(x-y)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+", "x", "/", "x", "sin", "-", "y", "x"}; // x + (x/sin(y-x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ 1 / - * 1 sin - y x * x * cos - y x - 0 1 * sin - y x sin - y x (prefix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
                                                          //+ 1 / - * 1 sin - y x * x * cos - y x ~ 1 * sin - y x sin - y x (prefix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
                                                          //+ 1 / - sin - y x * x * cos - y x ~ 1 * sin - y x sin - y x (prefix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
                                                          //+ 1 / - sin - y x * x * ~ 1 cos - y x * sin - y x sin - y x (prefix) -> 1 + (sin(y-x) + x*cos(y-x))/(sin(x-y)*sin(x-y)) = 1 - (1/sin(x-y)) + x*cos(y-x)/(sin(x-y)*sin(x-y)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"/", "x", "/", "x", "*", "y", "cos", "sin", "y"}; // x / (x / (y*cos(sin(y))))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); // / - * 0 / x * y cos sin y * x / - * 0 * y cos sin y * x + * y * ~ sin sin y * cos y 1 * 1 cos sin y * * y cos sin y * y cos sin y * / x * y cos sin y / x * y cos sin y (prefix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                          // / - * 0 / x * y cos sin y * x / - * 0 * y cos sin y * x + * y * ~ sin sin y * cos y 1 cos sin y * * y cos sin y * y cos sin y * / x * y cos sin y / x * y cos sin y (prefix) -> ((x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y)))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                          // / - 0 * x / - 0 * x + * y * ~ sin sin y * cos y 1 cos sin y * * y cos sin y * y cos sin y * / x * y cos sin y / x * y cos sin y (prefix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                          // / ~ * x / ~ * x + * y * ~ sin sin y * cos y 1 cos sin y * * y cos sin y * y cos sin y * / x * y cos sin y / x * y cos sin y (prefix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                          // / ~ * x / ~ * x + * y * * cos y 1 ~ sin sin y cos sin y * * y cos sin y * y cos sin y * / x * y cos sin y / x * y cos sin y (prefix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
                                                          // / ~ * x / ~ * x + * y * cos y ~ sin sin y cos sin y * * y cos sin y * y cos sin y * / x * y cos sin y / x * y cos sin y (prefix) -> (x*x*(y*(-sin(sin(y))*cos(y))+cos(sin(y))))/(y*cos(sin(y))*y*cos(sin(y))) / ((x/(y*cos(sin(y))))*(x/(y*cos(sin(y))))) = (y*(-sin(sin(y))*cos(y))+cos(sin(y))) = cos(sin(y)) - y*sin(sin(y))*cos(y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"/", "sin", "~", "~", "x", "y"}; // sin(x)/y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); // / - * * cos ~ ~ x ~ ~ 0 y * sin ~ ~ x 1 * y y (prefix) -> -sin(x)/(y*y) ✅
                                                          // / - * * cos ~ ~ x ~ ~ 0 y sin ~ ~ x * y y (prefix) -> -sin(x)/(y*y) ✅
                                                          // / - * * cos ~ ~ x 0 y sin ~ ~ x * y y (prefix) -> -sin(x)/(y*y) ✅
                                                          // / ~ sin ~ ~ x * y y (prefix) -> -sin(x)/(y*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"sqrt", "x"}; //sqrt(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / 1 * 2 sqrt x  (prefix) -> 1/(2*sqrt(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "sqrt", "x", "y"}; //sqrt(x)*y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * sqrt x 0 * / 1 * 2 sqrt x y (prefix) -> (1/(2*sqrt(x)))*y ✅
                                                          //+ 0 * / 1 * 2 sqrt x y (prefix) -> (1/(2*sqrt(x)))*y ✅
                                                          //* / 1 * 2 sqrt x y (prefix) -> (1/(2*sqrt(x)))*y ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "ln", "x", "y"}; //ln(x)*y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ln x 0 * / 1 x y (prefix) -> (1/x)*y ✅
                                                          //+ 0 * / 1 x y (prefix) -> (1/x)*y ✅
                                                          //* / 1 x y (1/x)*y ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "ln", "~", "x", "x"}; //ln(-x) * x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ln ~ x 1 * / ~ 1 ~ x x (prefix) -> ln(-x) + 1 ✅
                                                          //+ ln ~ x * / ~ 1 ~ x x (prefix) -> ln(-x) + 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"*", "ln", "sqrt", "x", "y"}; //ln(sqrt(x)) * y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ln sqrt x 0 * / / 1 * 2 sqrt x sqrt x y (prefix) -> y/(2*x) ✅
                                                          //+ 0 * / / 1 * 2 sqrt x sqrt x y (prefix) -> y/(2*x) ✅
                                                          //* / / 1 * 2 sqrt x sqrt x y (prefix) -> y/(2*x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"asin", "*", "x", "x"}; //arcsin(x*x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / + * x 1 * 1 x sqrt - 1 * * x x * x x (prefix) -> (2*x)/sqrt(1-x*x*x*x) ✅
                                                          // / + x * 1 x sqrt - 1 * * x x * x x (prefix) -> (2*x)/sqrt(1-x*x*x*x) ✅
                                                          // / + x x sqrt - 1 * * x x * x x (prefix) -> (2*x)/sqrt(1-x*x*x*x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"arcsin", "*", "ln", "x", "y"}; //arcsin(ln(x)*y)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); // / + * ln x 1 * / 0 x y sqrt - 1 * * ln x y * ln x y (prefix) -> ln(x) / sqrt(1-ln(x)*y*ln(x)*y) ✅
                                                          // / + ln x * / 0 x y sqrt - 1 * * ln x y * ln x y (prefix) -> ln(x) / sqrt(1-ln(x)*y*ln(x)*y) ✅
                                                          // / ln x sqrt - 1 * * ln x y * ln x y (prefix) -> ln(x) / sqrt(1-ln(x)*y*ln(x)*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"arcsin", "*", "ln", "x", "y"}; //arcsin(ln(x)*y)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp, true); // / + * ln x 0 * / 1 x y sqrt - 1 * * ln x y * ln x y (prefix) -> (y/x)/sqrt(1-y*ln(x)*y*ln(x)) ✅
                                                          // / + 0 * / 1 x y sqrt - 1 * * ln x y * ln x y (prefix) -> (y/x)/sqrt(1-y*ln(x)*y*ln(x)) ✅
                                                          // / * / 1 x y sqrt - 1 * * ln x y * ln x y (prefix) -> (y/x)/sqrt(1-y*ln(x)*y*ln(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"arcsin", "/", "acos", "x", "y"}; //arcsin(acos(x)/y)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / / - * ~ / 1 sqrt - 1 * x x y * acos x 0 * y y sqrt - 1 * / acos x y / acos x y (prefix) -> (-y/sqrt(1-x*x))/(y*y) * (1/sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) = -1/(y*sqrt(1-x*x)*sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) ✅
                                                          // / / - * ~ / 1 sqrt - 1 * x x y 0 * y y sqrt - 1 * / acos x y / acos x y (prefix) -> (-y/sqrt(1-x*x))/(y*y) * (1/sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) = -1/(y*sqrt(1-x*x)*sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) ✅
                                                          // / / * ~ / 1 sqrt - 1 * x x y * y y sqrt - 1 * / acos x y / acos x y (prefix) -> (-y/sqrt(1-x*x))/(y*y) * (1/sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) = -1/(y*sqrt(1-x*x)*sqrt(1-((arcos(x)/y)*(arcos(x)/y)))) ✅

    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"+", "arcsin", "*", "ln", "x", "y", "acos", "y"}; //arcsin(ln(x)*y)+acos(y)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); //+ / + * ln x 1 * / 0 x y sqrt - 1 * * ln x y * ln x y ~ / 1 sqrt - 1 * y y (prefix) -> (ln(x)/sqrt(1-ln(x)*y*ln(x)*y)) + (-1/sqrt(1-y*y)) ✅
                                                          //+ / + ln x * / 0 x y sqrt - 1 * * ln x y * ln x y ~ / 1 sqrt - 1 * y y (prefix) -> (ln(x)/sqrt(1-ln(x)*y*ln(x)*y)) + (-1/sqrt(1-y*y)) ✅
                                                          //+ / ln x sqrt - 1 * * ln x y * ln x y ~ / 1 sqrt - 1 * y y 
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"acos", "*", "acos", "acos", "x", "~", "x"}; //arccos(arccos(arccos(x))*-x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //~ / + * acos acos x ~ 1 * ~ / ~ / 1 sqrt - 1 * x x sqrt - 1 * acos x acos x ~ x sqrt - 1 * * acos acos x ~ x * acos acos x ~ x (prefix) -> (arccos(arccos(x)) + (x*((1/sqrt(1-x*x))/sqrt(1-acos(x)*acos(x))))) / sqrt(1-(-x*arccos(arccos(x))*-x*arccos(arccos(x)))) = (arccos(arccos(x)) + x/(sqrt(1-x*x)*sqrt(1-acos(x)*acos(x)))) / sqrt(1-(x*arccos(arccos(x))*x*arccos(arccos(x)))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"/", "exp", "x", "exp", "cos", "x"}; //exp(x) / exp(cos(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * * exp x 1 exp cos x * exp x * exp cos x * ~ sin x 1 * exp cos x exp cos x (prefix) -> (exp(x)*exp(cos(x)) - exp(x)*-sin(x)*exp(cos(x))) / (exp(cos(x))*exp(cos(x))) = (exp(x) - exp(x)*-sin(x)) / (exp(cos(x))) = (exp(x)*(1+sin(x))) / exp(cos(x)) = exp(x-cos(x))*(1+sin(x)) ✅
                                                          // / - * * 1 exp x exp cos x * exp x * * ~ sin x 1 exp cos x * exp cos x exp cos x (prefix) -> (exp(x)*exp(cos(x)) - exp(x)*-sin(x)*exp(cos(x))) / (exp(cos(x))*exp(cos(x))) = (exp(x) - exp(x)*-sin(x)) / (exp(cos(x))) = (exp(x)*(1+sin(x))) / exp(cos(x)) = exp(x-cos(x))*(1+sin(x)) ✅
                                                          // / - * exp x exp cos x * exp x * * ~ sin x 1 exp cos x * exp cos x exp cos x (prefix) -> (exp(x)*exp(cos(x)) - exp(x)*-sin(x)*exp(cos(x))) / (exp(cos(x))*exp(cos(x))) = (exp(x) - exp(x)*-sin(x)) / (exp(cos(x))) = (exp(x)*(1+sin(x))) / exp(cos(x)) = exp(x-cos(x))*(1+sin(x)) ✅
                                                          // / - * exp x exp cos x * exp x * ~ sin x exp cos x * exp cos x exp cos x (prefix) -> (exp(x)*exp(cos(x)) - exp(x)*-sin(x)*exp(cos(x))) / (exp(cos(x))*exp(cos(x))) = (exp(x) - exp(x)*-sin(x)) / (exp(cos(x))) = (exp(x)*(1+sin(x))) / exp(cos(x)) = exp(x-cos(x))*(1+sin(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"+", "exp", "~", "x", "*", "*", "x", "y", "x"}; //exp(-x) + x*y*x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * exp ~ x ~ 1 + * * x y 1 * + * x 0 * 1 y x (prefix) -> -exp(-x) + x*y + x*y = -exp(-x) + 2*x*y ✅
                                                          //+ * exp ~ x ~ 1 + * * x y 1 * + 0 * 1 y x (prefix) -> -exp(-x) + x*y + x*y = -exp(-x) + 2*x*y ✅
                                                          //+ * exp ~ x ~ 1 + * x y * + 0 * 1 y x (prefix) -> -exp(-x) + x*y + x*y = -exp(-x) + 2*x*y ✅
                                                          //+ * exp ~ x ~ 1 + * x y * y x (prefix) -> -exp(-x) + x*y + x*y = -exp(-x) + 2*x*y ✅
                                                          //+ * ~ 1 exp ~ x + * x y * y x (prefix) -> -exp(-x) + x*y + x*y = -exp(-x) + 2*x*y ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"arccos", "*", "exp", "arcsin", "y", "~", "x"}; //arccos(exp(arcsin(y))*-x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); //~ / + * exp arcsin y ~ 0 * * exp arcsin y / 1 sqrt - 1 * y y ~ x sqrt - 1 * * exp arcsin y ~ x * exp arcsin y ~ x (prefix) -> ((x/sqrt(1-y*y))*exp(arcsin(y))) / sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y)))) = (x/(sqrt(1-y*y)*sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y))))))*exp(arcsin(y)) = (x*exp(arcsin(y)))/(sqrt(1-y*y)*sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y))))) ✅
                                                          //~ / + * exp arcsin y ~ 0 * * / 1 sqrt - 1 * y y exp arcsin y ~ x sqrt - 1 * * exp arcsin y ~ x * exp arcsin y ~ x  (prefix) -> ((x/sqrt(1-y*y))*exp(arcsin(y))) / sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y)))) = (x/(sqrt(1-y*y)*sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y))))))*exp(arcsin(y)) = (x*exp(arcsin(y)))/(sqrt(1-y*y)*sqrt(1-(x*exp(arcsin(y))*x*exp(arcsin(y))))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';

    prefix = {"^", "x", "y"}; //x ^ y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); //* ^ x y + * ln x 1 * / 0 x y (prefix) -> (x^y)*ln(x) ✅
                                                          //* ^ x y + ln x * / 0 x y (prefix) -> (x^y)*ln(x) ✅
                                                          //* ^ x y ln x (prefix) -> (x^y)*ln(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "^", "cos", "x", "cos", "y", "x"}; //(cos(x)^(cos(y)))*x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp, true); //+ * ^ cos x cos y 0 * * ^ cos x cos y + * ln cos x * ~ sin y 1 * / * ~ sin x 0 cos x cos y x (prefix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
                                                          //+ 0 * * ^ cos x cos y + * ln cos x * ~ sin y 1 * / * ~ sin x 0 cos x cos y x (prefix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
                                                          //* * ^ cos x cos y + * ln cos x * ~ sin y 1 * / * ~ sin x 0 cos x cos y x (prefix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
                                                          //* * ^ cos x cos y * ln cos x ~ sin y x (prefix) -> -x*(cos(x)^(cos(y)))*ln(cos(x))*sin(y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "^", "cos", "x", "cos", "y", "x"}; //(cos(x)^(cos(y)))*x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ^ cos x cos y 1 * * ^ cos x cos y + * ln cos x * ~ sin y 0 * / * ~ sin x 1 cos x cos y x (prefix) -> cos(x)^(cos(y)) + x*cos(x)^(cos(y)) * ((-sin(x)/cos(x))*cos(y)) = cos(x)^(cos(y)) - cos(y)*x*cos(x)^(cos(y)-1)*sin(x) ✅
                                                          //+ ^ cos x cos y * * ^ cos x cos y + * ln cos x * ~ sin y 0 * / * ~ sin x 1 cos x cos y x (prefix) -> cos(x)^(cos(y)) + x*cos(x)^(cos(y)) * ((-sin(x)/cos(x))*cos(y)) = cos(x)^(cos(y)) - cos(y)*x*cos(x)^(cos(y)-1)*sin(x) ✅
                                                          //+ ^ cos x cos y * * ^ cos x cos y * / ~ sin x cos x cos y x (prefix) -> cos(x)^(cos(y)) + x*cos(x)^(cos(y)) * ((-sin(x)/cos(x))*cos(y)) = cos(x)^(cos(y)) - cos(y)*x*cos(x)^(cos(y)-1)*sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"*", "^", "^", "x", "x", "x", "y"}; //((x^x)^x)*y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ^ ^ x x x 0 * * ^ ^ x x x + * ln ^ x x 1 * / * ^ x x + * ln x 1 * / 1 x x ^ x x x y (prefix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
                                                         //+ 0 * * ^ ^ x x x + * ln ^ x x 1 * / * ^ x x + * ln x 1 * / 1 x x ^ x x x y (prefix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
                                                         //+ 0 * * ^ ^ x x x + ln ^ x x * / * ^ x x + ln x * / 1 x x ^ x x x y (prefix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
                                                         //* * ^ ^ x x x + ln ^ x x * / * ^ x x + ln x * / 1 x x ^ x x x y (prefix) -> y*((x^x)^x)*(ln(x^x) + (((x^x)*(ln(x)+1))/(x^x))*x) = y*((x^x)^x)*(ln(x^x) + x*(ln(x)+1)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "^", "^", "x", "x", "x", "y"}; //((x^x)^x)*y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); //+ * ^ ^ x x x 1 * * ^ ^ x x x + * ln ^ x x 0 * / * ^ x x + * ln x 0 * / 0 x x ^ x x x y (prefix) -> ((x^x)^x) ✅
                                                          //+ * ^ ^ x x x 1 * * ^ ^ x x x + 0 * / * ^ x x + 0 * / 0 x x ^ x x x y (prefix) -> ((x^x)^x) ✅
                                                          //+ ^ ^ x x x * * ^ ^ x x x + 0 * / * ^ x x + 0 * / 0 x x ^ x x x y (prefix) -> ((x^x)^x) ✅
                                                          //+ ^ ^ x x x * * ^ ^ x x x * / * ^ x x * / 0 x x ^ x x x y (prefix) -> ((x^x)^x) ✅
                                                          //^ ^ x x x (prefix) -> ((x^x)^x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
        
    prefix = {"*", "^", "tanh", "sech", "x", "x", "y"}; //tanh(sech(x))^x * y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * ^ tanh sech x x 0 * * ^ tanh sech x x + * ln tanh sech x 1 * / * * sech sech x sech sech x ~ * * sech x tanh x 1 tanh sech x x y (prefix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                          //+ 0 * * ^ tanh sech x x + * ln tanh sech x 1 * / * * sech sech x sech sech x ~ * * sech x tanh x 1 tanh sech x x y (prefix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                         //+ 0 * * ^ tanh sech x x + ln tanh sech x * / * * sech sech x sech sech x ~ * * sech x tanh x 1 tanh sech x x y (prefix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                         //* * ^ tanh sech x x + ln tanh sech x * / * * sech sech x sech sech x ~ * * sech x tanh x 1 tanh sech x x y (prefix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                         //* * ^ tanh sech x x + ln tanh sech x * / * * 1 * ~ sech x tanh x * sech sech x sech sech x tanh sech x x y (prefix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
                                                         //* * ^ tanh sech x x + ln tanh sech x * / * * ~ sech x tanh x * sech sech x sech sech x tanh sech x x y (prefix) -> y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))+(x*((sech(sech(x))*sech(sech(x))*-sech(x)*tanh(x))/tanh(sech(x))))) = y*(tanh(sech(x))^x)*(ln(tanh(sech(x)))-((x*sech(sech(x))*sech(sech(x))*sech(x)*tanh(x))/tanh(sech(x)))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "x", "^", "tanh", "/", "x", "y", "sin", "x"}; //x*tanh(x/y)^(sin(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * x * ^ tanh / x y sin x + * ln tanh / x y * cos x 1 * / * * sech / x y sech / x y / - * 1 y * x 0 * y y tanh / x y sin x * 1 ^ tanh / x y sin x (prefix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                          //+ * x * ^ tanh / x y sin x + * ln tanh / x y * cos x 1 * / * * sech / x y sech / x y / - * 1 y * x 0 * y y tanh / x y sin x ^ tanh / x y sin x (prefix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                          //+ * x * ^ tanh / x y sin x + * ln tanh / x y * cos x 1 * / * * sech / x y sech / x y / - y * x 0 * y y tanh / x y sin x ^ tanh / x y sin x (prefix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                          //+ * x * ^ tanh / x y sin x + * ln tanh / x y * cos x 1 * / * * sech / x y sech / x y / - y 0 * y y tanh / x y sin x ^ tanh / x y sin x (prefix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                          //+ * x * ^ tanh / x y sin x + * ln tanh / x y * cos x 1 * / * * sech / x y sech / x y / y * y y tanh / x y sin x ^ tanh / x y sin x (prefix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                          //+ * x * ^ tanh / x y sin x + * ln tanh / x y cos x * / * * sech / x y sech / x y / y * y y tanh / x y sin x ^ tanh / x y sin x (prefix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                          //+ * x * ^ tanh / x y sin x + * ln tanh / x y cos x * / * / y * y y * sech / x y sech / x y tanh / x y sin x ^ tanh / x y sin x (prefix) -> (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*(1/y)*sin(x))/(tanh(x/y))) = (tanh(x/y)^(sin(x))) + x*(tanh(x/y)^(sin(x)))*(ln(tanh(x/y))*cos(x) + (sech(x/y)*sech(x/y)*sin(x))/(tanh(x/y)*y)) ✅
                                                        
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"sech", "sin", "sin", "^", "sech", "sin", "x", "*", "x", "y"}; //sech(sin(sin( sech(sin(x))^(x*y))))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //~ * * sech sin sin ^ sech sin x * x y tanh sin sin ^ sech sin x * x y * cos sin ^ sech sin x * x y * cos ^ sech sin x * x y * ^ sech sin x * x y + * ln sech sin x + * x 0 * 1 y * / ~ * * sech sin x tanh sin x * cos x 1 sech sin x * x y (prefix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                          //~ * * sech sin sin ^ sech sin x * x y tanh sin sin ^ sech sin x * x y * cos sin ^ sech sin x * x y * cos ^ sech sin x * x y * ^ sech sin x * x y + * ln sech sin x + 0 * 1 y * / ~ * * sech sin x tanh sin x * cos x 1 sech sin x * x y (prefix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                          //~ * * sech sin sin ^ sech sin x * x y tanh sin sin ^ sech sin x * x y * cos sin ^ sech sin x * x y * cos ^ sech sin x * x y * ^ sech sin x * x y + * ln sech sin x y * / ~ * * sech sin x tanh sin x * cos x 1 sech sin x * x y (prefix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                          //~ * * sech sin sin ^ sech sin x * x y tanh sin sin ^ sech sin x * x y * * * ^ sech sin x * x y + * ln sech sin x y * / ~ * * sech sin x tanh sin x cos x sech sin x * x y cos ^ sech sin x * x y cos sin ^ sech sin x * x y (prefix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
                                                          //* * * * ^ sech sin x * x y + * ln sech sin x y * / * cos x * ~ sech sin x tanh sin x sech sin x * x y cos ^ sech sin x * x y cos sin ^ sech sin x * x y * ~ sech sin sin ^ sech sin x * x y tanh sin sin ^ sech sin x * x y (prefix) -> -sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) * cos(sin(sech(sin(x))^(x*y))) * cos(sech(sin(x))^(x*y)) * sech(sin(x))^(x*y) * (ln(sech(sin(x)))*y - x*y*tanh(sin(x))*cos(x)) = -sech(sin(x))^(x*y)*(y*ln(sech(sin(x))) - y*x*tanh(sin(x))*cos(x)) * cos(sech(sin(x))^(x*y)) * cos(sin(sech(sin(x))^(x*y))) * sech(sin(sin(sech(sin(x))^(x*y)))) * tanh(sin(sin(sech(sin(x))^(x*y)))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"sin", "~", "sech", "/", "arccos", "ln", "x", "*", "x", "y"}; //sin(-sech(arccos(ln(x))/(x*y)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* cos ~ sech / arccos ln x * x y ~ ~ * * sech / arccos ln x * x y tanh / arccos ln x * x y / - * ~ / / 1 x sqrt - 1 * ln x ln x * x y * arccos ln x + * x 0 * 1 y * * x y * x y (prefix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                          //* cos ~ sech / arccos ln x * x y ~ ~ * * sech / arccos ln x * x y tanh / arccos ln x * x y / - * ~ / / 1 x sqrt - 1 * ln x ln x * x y * arccos ln x + 0 * 1 y * * x y * x y (prefix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                          //* cos ~ sech / arccos ln x * x y ~ ~ * * sech / arccos ln x * x y tanh / arccos ln x * x y / - * ~ / / 1 x sqrt - 1 * ln x ln x * x y * arccos ln x y * * x y * x y (prefix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                          //* cos ~ sech / arccos ln x * x y * * sech / arccos ln x * x y tanh / arccos ln x * x y / - * ~ / / 1 x sqrt - 1 * ln x ln x * x y * arccos ln x y * * x y * x y (prefix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                          //* * * sech / arccos ln x * x y tanh / arccos ln x * x y / - * ~ / / 1 x sqrt - 1 * ln x ln x * x y * arccos ln x y * * x y * x y cos ~ sech / arccos ln x * x y (prefix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
                                                          //* ~ * / - * ~ / / 1 x sqrt - 1 * ln x ln x * x y * arccos ln x y * * x y * x y * ~ sech / arccos ln x * x y tanh / arccos ln x * x y cos ~ sech / arccos ln x * x y (prefix) -> cos(-sech(arccos(ln(x))/(x*y))) * sech(arccos(ln(x))/(x*y))*tanh(arccos(ln(x))/(x*y))* ((-y/sqrt(1-ln(x)*ln(x))) - arccos(ln(x))*y)/(x*y*x*y) = ((-arccos(ln(x))/(y*x*x)) - (1/(sqrt(1-ln(x)*ln(x))*y*x*x))) * sech(arccos(ln(x))/(x*y)) * tanh(arccos(ln(x))/(x*y)) * cos(sech(arccos(ln(x))/(x*y))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
        
    prefix = {"-", "*", "0", "x", "+", "x", "sin", "x"};  //0*x - (x+sin(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //- + * 0 1 * 0 x + 1 * cos x 1 (prefix) -> -1-cos(x) ✅
                                                          //- + 0 * 0 x + 1 * cos x 1 (prefix) -> -1-cos(x) ✅
                                                          //~ + 1 * cos x 1 (prefix) -> -1-cos(x) ✅
                                                          //~ + 1 cos x (prefix) -> -1-cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "~", "*", "0", "x", "tanh", "x"}; //-(0*x) + tanh(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ ~ + * 0 1 * 0 x * * sech x sech x 1 (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ + 0 * 0 x * * sech x sech x 1 (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ 0 * * sech x sech x 1 (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ 0 * 1 * sech x sech x (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ 0 * sech x sech x (prefix) -> sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "1", "x"}; //1*x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * 1 1 * 0 x (prefix) -> 1 ✅
                                                          //+ 1 * 0 x (prefix) -> 1 ✅
                                                          //1 (prefix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"-", "*", "1", "x", "+", "x", "sin", "x"};  //1*x - (x+sin(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //- + * 1 1 * 0 x + 1 * cos x 1 (prefix) -> 1 - (1+cos(x)) ✅
                                                          //- + 1 * 0 x + 1 * cos x 1 (prefix) -> 1 - (1+cos(x)) ✅
                                                          //- 1 + 1 * cos x 1 (prefix) -> 1 - (1+cos(x)) ✅
                                                          //- 1 + 1 cos x (prefix) -> 1 - (1+cos(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "~", "*", "1", "x", "tanh", "x"}; //-(1*x) + tanh(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ ~ + 1 * 0 x * * sech x sech x 1 (prefix) -> -1 + sech(x)*sech(x) ✅
                                                          //+ ~ 1 * * sech x sech x 1 (prefix) -> -1 + sech(x)*sech(x) ✅
                                                          //+ ~ 1 * 1 * sech x sech x (prefix) -> -1 + sech(x)*sech(x) ✅
                                                          //+ ~ 1 * sech x sech x (prefix) -> -1 + sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "x", "0"}; //x*0
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * 0 1 * 0 x (prefix) -> 0 ✅
                                                          //+ 0 * 0 x (prefix) -> 0 ✅
                                                          //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"-", "*", "x", "0", "+", "x", "sin", "x"}; //x*0 - (x+sin(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //- + * 0 1 * 0 x + 1 * cos x 1 (prefix) -> -1-cos(x) ✅
                                                          //- + 0 * 0 x + 1 * cos x 1 (prefix) -> -1-cos(x) ✅
                                                          //~ + 1 * cos x 1 (prefix) -> -1-cos(x) ✅
                                                          //~ + 1 cos x (prefix) -> -1-cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "~", "*", "x", "0", "tanh", "x"}; //-(x*0) + tanh(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ ~ + * 0 1 * 0 x * * sech x sech x 1 (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ + 0 * 0 x * * sech x sech x 1 (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ 0 * * sech x sech x 1 (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ 0 * 1 * sech x sech x (prefix) -> sech(x)*sech(x) ✅
                                                          //+ ~ 0 * sech x sech x (prefix) -> sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "+", "sin", "x", "x", "1"}; //(sin(x)+x)*1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ * cos x 1 1 (prefix) -> cos(x)*1 + 1 ✅
                                                          //+ cos x 1 (prefix) -> cos(x) + 1 ✅
                                                          
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "~", "tanh", "*", "x", "1", "*", "1", "1"}; //-tanh(x*1)*1 + 1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp);      //~ * * sech * x 1 sech * x 1 1 (prefix) -> -sech(x)*sech(x) ✅
                                                               //~ * 1 * sech * x 1 sech * x 1 (prefix) -> -sech(x)*sech(x) ✅
                                                               //~ * sech * x 1 sech * x 1 (prefix) -> -sech(x)*sech(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
        
    prefix = {"+", "-", "sin", "x", "sin", "x", "sin", "x"}; //sin(x) - sin(x) + sin(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp);    //* cos x 1 (prefix) -> cos(x) ✅
                                                             //cos x (prefix) -> cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "x", "1"}; //x/1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp);  // / - * 1 1 * x 0 * 1 1 (prefix) -> 1 ✅
                                                           // / - 1 * x 0 * 1 1 (prefix) -> 1 ✅
                                                           // / - 1 0 * 1 1 (prefix) -> 1 ✅
                                                           // - 1 0 (prefix) -> 1 ✅
                                                           // 1 (prefix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "*", "x", "x", "1"}; //(x*x)/1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * + x x 1 * * x x 0 * 1 1 (prefix) -> 2*x ✅
                                                          // / - + x x * * x x 0 * 1 1 (prefix) -> 2*x ✅
                                                          // / - + x x 0 * 1 1 (prefix) -> 2*x ✅
                                                          // - + x x 0 (prefix) -> 2*x ✅
                                                          // + x x (prefix) -> 2*x ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "*", "x", "cos", "x", "1"}; //(x*cos(x))/1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * + * x * ~ sin x 1 cos x 1 * * x cos x 0 * 1 1 (prefix) -> (-sin(x)*x + cos(x))/1 ✅
                                                          // / - + * x * ~ sin x 1 cos x * * x cos x 0 * 1 1 (prefix) -> (-sin(x)*x + cos(x))/1 ✅
                                                          // / - + * x * ~ sin x 1 cos x 0 * 1 1 (prefix) -> (-sin(x)*x + cos(x))/1 ✅
                                                          // - + * x * ~ sin x 1 cos x 0 (prefix) -> (-sin(x)*x + cos(x)) ✅
                                                          // + * x * ~ sin x 1 cos x (prefix) -> (-sin(x)*x + cos(x)) ✅
                                                          // + * x ~ sin x cos x (prefix) -> (-sin(x)*x + cos(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "0", "*", "x", "x"}; //0/(x*x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * 0 * x x * 0 + x x * * x x * x x (prefix) -> 0 ✅
                                                          // / - 0 * 0 + x x * * x x * x x (prefix) -> 0 ✅
                                                          // / - 0 0 * * x x * x x (prefix) -> 0 ✅
                                                          // 0 (prefix) -> 0 ✅
                                                        
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "0", "*", "x", "cos", "x"}; //0/(x*cos(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * 0 * x cos x * 0 + * x * ~ sin x 1 cos x * * x cos x * x cos x (prefix) -> 0/(x*cos(x)*x*cos(x)) ✅
                                                          // / - 0 * 0 + * x * ~ sin x 1 cos x * * x cos x * x cos x 0/(x*cos(x)*x*cos(x)) ✅
                                                          // / - 0 0 * * x cos x * x cos x (prefix) -> 0 ✅
                                                          // 0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "0", "*", "sin", "x", "sech", "x"}; //0/(sin(x)*sech(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * 0 * sin x sech x * 0 + * sin x ~ * * sech x tanh x 1 * * cos x 1 sech x * * sin x sech x * sin x sech x (prefix) -> 0 ✅
                                                          // / - 0 * 0 + * sin x ~ * * sech x tanh x 1 * * cos x 1 sech x * * sin x sech x * sin x sech x (prefix) -> 0 ✅
                                                          // / - 0 0 * * sin x sech x * sin x sech x (prefix) -> 0 ✅
                                                          // 0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "1", "*", "x", "x"}; //1/(x*x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * 0 * x x * 1 + x x * * x x * x x (prefix) -> -2/(x*x*x) ✅
                                                          // / - 0 * 1 + x x * * x x * x x (prefix) -> -2/(x*x*x) ✅
                                                          // / - 0 + x x * * x x * x x (prefix) -> -2/(x*x*x) ✅
                                                          // / ~ + x x * * x x * x x (prefix) -> -2/(x*x*x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "1", "cos", "x"}; //1/cos(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * 0 cos x * 1 * ~ sin x 1 * cos x cos x (prefix) -> sin(x)/(cos(x)*cos(x)) ✅
                                                          // / - 0 * 1 * ~ sin x 1 * cos x cos x (prefix) -> sin(x)/(cos(x)*cos(x)) ✅
                                                          // / - 0 * ~ sin x 1 * cos x cos x (prefix) -> sin(x)/(cos(x)*cos(x)) ✅
                                                          // / ~ * ~ sin x 1 * cos x cos x (prefix) -> sin(x)/(cos(x)*cos(x)) ✅
                                                          // / ~ ~ sin x * cos x cos x (prefix) -> sin(x)/(cos(x)*cos(x)) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "1", "*", "cos", "x", "sin", "x"}; //1/(cos(x)*sin(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / - * 0 * cos x sin x * 1 + * cos x * cos x 1 * * ~ sin x 1 sin x * * cos x sin x * cos x sin x (prefix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                          // / - 0 * 1 + * cos x * cos x 1 * * ~ sin x 1 sin x * * cos x sin x * cos x sin x (prefix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                          // / - 0 + * cos x * cos x 1 * * ~ sin x 1 sin x * * cos x sin x * cos x sin x (prefix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                          // / ~ + * cos x * cos x 1 * * ~ sin x 1 sin x * * cos x sin x * cos x sin x (prefix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                          // / ~ + * cos x * cos x 1 * ~ sin x sin x * * cos x sin x * cos x sin x (prefix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
                                                          // / ~ + * cos x cos x * ~ sin x sin x * * cos x sin x * cos x sin x (prefix) -> -(cos(x)*cos(x) - sin(x)*sin(x)) / ((cos(x)*sin(x))*(cos(x)*sin(x))) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
        
    prefix = {"+", "x", "sin", "~", "~", "x"}; //x + sin(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ 1 * cos ~ ~ x 1 (prefix) -> 1 + 1*cos(x) ✅
                                                          //+ 1 cos ~ ~ x (prefix) -> 1 + cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"-", "tanh", "~", "~", "x", "x"}; //tanh(x) - x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //- * * sech ~ ~ x sech ~ ~ x 1 1 (prefix) -> sech(x)*sech(x) - 1 ✅
                                                          //- * 1 * sech ~ ~ x sech ~ ~ x 1 (prefix) -> sech(x)*sech(x) - 1 ✅
                                                          //- * sech ~ ~ x sech ~ ~ x 1 (prefix) -> sech(x)*sech(x) - 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "^", "0", "x", "x"}; //x + 0^x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //1 (prefix) -> 1 ✅

    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "^", "0", "x", "x"}; //0^x - x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //~ 1 (prefix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "cos", "x", "^", "0", "x"}; //cos(x) - 0^x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* ~ sin x 1 (prefix) -> -sin(x) ✅
                                                          //~ sin x (prefix) -> -sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "x", "^", "x", "0"}; //x + x^0
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //1 (prefix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "^", "x", "0", "x"}; //x^0 - x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //~ 1 (prefix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "cos", "x", "^", "x", "0"}; //cos(x) - x^0
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* ~ sin x 1 (prefix) -> -sin(x) ✅
                                                          //~ sin x (prefix) -> -sin(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "x", "^", "1", "x"}; //x + 1^x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //1 (prefix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "^", "1", "x", "x"}; //1^x - x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //~ 1 (prefix) -> -1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "cos", "x", "^", "1", "x"}; //cos(x) - 1^x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* ~ sin x 1 (prefix) -> -sin(x) ✅
                                                          //~ sin x
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "x", "^", "x", "1"}; //x + x^1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //+ 1 * ^ x 1 / 1 x  (prefix) -> 2 ✅
                                                          //+ 1 * x / 1 x (prefix) -> 2 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "^", "x", "1", "x"}; //x^1 - x
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //- * ^ x 1 / 1 x 1 (prefix) -> 0 ✅
                                                          //- * x / 1 x 1 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");

    prefix = {"-", "cos", "x", "^", "x", "1"}; //cos(x) - x^1
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //- * ~ sin x 1 * ^ x 1 / 1 x (prefix) -> -sin(x) - 1 ✅
                                                          //- * ~ sin x 1 * x / 1 x (prefix) -> -sin(x) - 1 ✅
                                                          //- ~ sin x * x / 1 x (prefix) -> -sin(x) - 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
        
    prefix = {"ln", "*", "1", "exp", "x"}; //ln(1*exp(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / * exp x 1 * 1 exp x (prefix) -> 1 ✅
                                                          // 1 (prefix) -> 1 ✅
                                                          // / exp x * 1 exp x (prefix) -> 1 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); }  sout.str("");
    
    prefix = {"-", "x", "ln", "*", "1", "exp", "x"}; //x - ln(1*exp(x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // - 1 / * exp x 1 * 1 exp x (prefix) -> 0 ✅
                                                          // 0 (prefix) -> 0 ✅
                                                          // - 1 / exp x * 1 exp x (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"cos", "-", "x", "ln", "*", "1", "exp", "x"}; //cos(x - ln(1*exp(x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* ~ sin - x ln * 1 exp x - 1 / * exp x 1 * 1 exp x (prefix) -> 0 ✅
                                                          //* ~ sin - x ln * 1 exp x 0 (prefix) -> 0 ✅
                                                          //* ~ sin - x ln * 1 exp x - 1 / exp x * 1 exp x (prefix) -> 0 ✅
                                                          //* - 1 / exp x * 1 exp x ~ sin - x ln * 1 exp x (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); }  sout.str("");
    
    prefix = {"ln", "exp", "*", "y", "y"}; //ln(exp(y*y))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"*", "exp", "*", "x", "x", "ln", "y"}; //exp(x*x)*ln(y)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); //* exp * x x / 1 y (prefix) -> exp(x*x)*(1/y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"+", "exp", "exp", "-", "y", "y", "sin", "x"}; //exp(exp(y-y))+sin(x)
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* cos x 1 (prefix) -> cos(x) ✅
                                                          //cos x (prefix) -> cos(x) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    
    prefix = {"/", "sin", "*", "x", "x", "y"}; // sin(x*x)/y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); // / - * * cos * x x 0 y sin * x x * y y (prefix) -> -sin(x*x)/(y*y) ✅
                                                          // / ~ sin * x x * y y (prefix) -> -sin(x*x)/(y*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"/", "sin", "cos", "x", "y"}; // sin(cos(x))/y
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "y", prefix, grasp); // / - * * cos cos x 0 y sin cos x * y y (prefix) -> -sin(cos(x))/(y*y) ✅
                                                          // / ~ sin cos x * y y (prefix) -> -sin(cos(x))/(y*y) ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"cos", "sqrt", "-", "x", "x"}; // cos(sqrt(x-x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* / 0 * 2 sqrt - x x ~ sin sqrt - x x (prefix) -> 0 ✅
                                                          //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"sin", "tanh", "sqrt", "-", "*", "x", "x", "*", "x", "x"}; // sin(tanh(sqrt(x*x - x*x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* * * sech sqrt - * x x * x x sech sqrt - * x x * x x / 0 * 2 sqrt - * x x * x x cos tanh sqrt - * x x * x x (prefix) -> 0 ✅
                                                          //* * * sech sqrt - * x x * x x sech sqrt - * x x * x x 0 cos tanh sqrt - * x x * x x (prefix) -> 0 ✅
                                                          //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"sqrt", "sqrt", "-", "*", "x", "cos", "x", "*", "x", "cos", "x"}; // sqrt(sqrt(x*cos(x) - x*cos(x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); // / / 0 * 2 sqrt - * x cos x * x cos x * 2 sqrt sqrt - * x cos x * x cos x (prefix) -> 0 ✅
                                                          //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"cos", "arcsin", "-", "x", "x"}; // cos(arcsin(x-x))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"sin", "tanh", "asin", "-", "^", "x", "x", "^", "x", "x"}; // sin(tanh(arcsin(x^x - x^x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //* * * sech asin - ^ x x ^ x x sech asin - ^ x x ^ x x 0 cos tanh asin - ^ x x ^ x x (prefix) -> 0 ✅
                                                          //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"asin", "arcsin", "-", "*", "x", "sin", "x", "*", "x", "sin", "x"}; // arcsin(arcsin(x*sin(x) - x*sin(x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"exp", "acos", "-", "tanh", "x", "tanh", "x"}; // exp(acos(tanh(x)-tanh(x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"sech", "sech", "arccos", "-", "/", "x", "x", "/", "x", "x"}; // sech(sech(arccos(x/x - x/x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //~ * * sech sech arccos - / x x / x x tanh sech arccos - / x x / x x ~ * * sech arccos - / x x / x x tanh arccos - / x x / x x 0 (prefix) -> 0 ✅
                                                          //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"acos", "arccos", "-", "-", "x", "sech", "x", "-", "x", "sech", "x"}; // acos(arccos((x-sech(x)) - (x-sech(x))))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"acos", "tanh", "-", "*", "x", "exp", "x", "*", "x", "exp", "x"}; // acos(tanh(x*exp(x) - x*exp(x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"asin", "sech", "-", "*", "x", "exp", "x", "*", "x", "exp", "x"}; // asin(sech(x*exp(x) - x*exp(x)))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"acos", "sech", "-", "-", "x", "sech", "x", "-", "x", "sech", "x"}; // acos(sech((x-sech(x)) - (x-sech(x))))
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << grasp << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str(""); //std::vector<std::string>(derivat.begin(), derivat.begin() + Index - 1) << '\n';
    
    prefix = {"+", "cos", "+", "x1", "x2", "+", "x1", "x2"};
    std::cout << "prefix = " << prefix << '\n'; derivePrefix(0, prefix.size()-1, "x1", prefix, grasp); //0 (prefix) -> 0 ✅
    std::cout << "prefix = " << prefix << '\n';
    std::cout << grasp << '\n';
    std::cout << "RGBs = " << getRGBs(prefix) << '\n';
    sout << derivat; std::cout << derivat << "\n\n"; if (ASSERT) {assert(string_in_file(sout.str(), "PrefixDifferentiationSymbolic.cpp")); } sout.str("");
    return 0;
}

//g++ -std=c++20 -o PrefixDifferentiationSymbolic PrefixDifferentiationSymbolic.cpp
