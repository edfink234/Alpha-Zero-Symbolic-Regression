//implementation of http://elib.mi.sanu.ac.rs/files/journals/yjor/21/yujorn21p61-75.pdf
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

long numElements = 9;
const std::vector<std::string> postfix = {"y","y","x","*","*","cos", "y", "+"};//{"x", "y", "x", "x", "x", "-", "+", "+", "*"};//{"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}; //array of postfix expression elements read from left to right
std::vector<int> grasp(numElements-1);
std::vector<std::string> derivat(100); //Assuming the derivative isn't larger than 100 elements
int index_count = 0; //global integer variable initially equal to zero, which represents the index of the array derivat

const std::vector<std::string> unary_operators = {"cos"};
const std::vector<std::string> binary_operators = {"+","-","*"};

bool is_unary(const std::string& token)
{
    return (std::find(unary_operators.begin(), unary_operators.end(), token) != unary_operators.end());
}

bool is_binary(const std::string& token)
{
    return (std::find(binary_operators.begin(), binary_operators.end(), token) != binary_operators.end());
}

//Function to compute the LGB, from https://www.jstor.org/stable/43998756 (top of pg. 165)
void LGB(int z, int& ind)
{
    do
    {
        --ind;
        if (is_unary(postfix[ind]))
        {
            LGB(1, ind);
        }
        else if (is_binary(postfix[ind]))
        {
            LGB(2, ind);
        }
        --z;
    } while (z);
}

//Computes the grasp of an arbitrary element postfix[i], from https://www.jstor.org/stable/43998756 (bottom of pg. 165)
int GR(int i)
{
    int start = i;
    int& ptr_lgb = start;
    if (is_unary(postfix[i]))
    {
        LGB(1, ptr_lgb);
    }
    else if (is_binary(postfix[i]))
    {
        LGB(2, ptr_lgb);
    }
    return (i - ptr_lgb);
}


/*
low and up: lower and upper index bounds, respectively, for the piece of the array postfix which is to be the subject of the processing.
dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
*/
void derivePostfix(int low, int up, const std::string& dx)
{
    //allowed ops: +, -, *, /, ^, unary +, unary -, sin(), cos(), tan(), ctg(), log(), sqrt(), const, x0, x1, ..., x_numFeatures
    //Define `grasp` of postfix[i], i.e., the number of elements forming operands of postfix[i] (grasp(operand) = 0)
    //The grasped elements of postfix[i] are the elements forming operands of postfix[i]
    //The left-grasp-bound (LGB) of postfix[i] is the index of the left-most grasped element of postfix[i] in the array postfix
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

    if (postfix[up] == "+" || postfix[up] == "-")
    {
        derivePostfix(low, up-2-grasp[up-1], dx);
        derivePostfix(up-1-grasp[up-1], up-1, dx);
        derivat[index_count++] = postfix[up];
    }
    else if (postfix[up] == "*")
    {
        // printf("up-2-grasp[up-1] = %d\n",up-2-grasp[up-1]);
        for (int k = low; k <= up-2-grasp[up-1]; k++)
        {
            derivat[index_count++] = postfix[k]; /* x */
        }
        // printf("up-1-grasp[up-1] = %d, up-1=%d\n",up-1-grasp[up-1],up-1);
        derivePostfix(up-1-grasp[up-1], up-1, dx); /* x y' */
        derivat[index_count++] = "*"; /* x y' "*" */
        // printf("up-2-grasp[up-1] = %d\n",up-2-grasp[up-1]);
        derivePostfix(low, up-2-grasp[up-1], dx); /* x y' "*" x' */
        for (int k = up-1-grasp[up-1]; k <= up-1; k++)
        {
            derivat[index_count++] = postfix[k]; /* x y' "*" x' y */
        }
        derivat[index_count++] = "*"; /* x y' "*" x' y "*" */
        derivat[index_count++] = "+";
    }
    else if (postfix[up] == "cos")
    {
        for (int k = low; k <= up-1; k++)
        {
            derivat[index_count++]=postfix[k]; /* x */
        }
        derivat[index_count++]="-sin"; /* x "-sin" */
        derivePostfix(low, up-1, dx);
        derivat[index_count++] = "*";
    }
    else
    {
        if (postfix[up] == dx)
        {
            derivat[index_count++] = "1";
        }
        else
        {
            derivat[index_count++] = "0";
        }
    }
    
}

void setGR()
{
    //In the paper they do `k = 1;` instead of `k = 0;`, presumably because GR(postfix[0]) always is 0, but it works
    //if you set k = 0 too.
    for (int k = 0; k < numElements; ++k)
    {
        grasp[k] = GR(k);
        printf("%d ",grasp[k]);
    }
    puts("");
}

int main()
{
    numElements = 8;
    setGR();

    derivePostfix(0, numElements-1, "x");
    for (size_t i = 0; i <= index_count; i++)
    {
        std::cout << derivat[i] << ' ';
    }
    
    return 0;
}




