#include <vector>
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <ctime>
#include <cstdlib>
#include <stack>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>

using Clock = std::chrono::high_resolution_clock;

std::vector<std::vector<float>> generateData(int numRows, int numCols, float minValue, float maxValue, float (*func)(const std::vector<float>&))
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(minValue, maxValue);

    // Create the matrix
    std::vector<std::vector<float>> matrix(numRows, std::vector<float>(numCols));

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols - 1; j++)
        {
            matrix[i][j] = distribution(gen);
        }
        matrix[i][numCols - 1] = func(matrix[i]);
    }

    return matrix;
}

class Data
{
    std::vector<std::vector<float>> data;
    std::unordered_map<std::string, std::vector<float>> features;
    
public:
    
    Data(const std::vector<std::vector<float>>& theData) : data{theData}
    {
        auto temp_sz = data[0].size() - 1;
        for (size_t i = 0; i < temp_sz; i++)
        {
            for (size_t j = 0; j < data.size(); j++)
            {
                features["x"+std::to_string(i)].push_back(data[j][i]);
            }
        }
        
        for (size_t i = 0; i < data.size(); i++)
        {
            features["y"].push_back(data[i][temp_sz]);
        }
        
    }
    const std::vector<float>& operator[] (int i){return data[i];}
    const std::vector<float>& operator[] (const std::string& i)
    {
        return features[i];
    }
    std::size_t size() {return data.size();}
    friend std::ostream& operator<<(std::ostream& os, const Data& matrix)
    {
        for (const auto& row : matrix.data)
        {
            for (const float value : row)
            {
                os << value << ' ';
            }
            os << '\n';
        }
        return os;
    }
};

std::ostream& operator<<(std::ostream& os, const std::vector<float>& row)
{
    for (const float value : row)
    {
        os << value << '\n';
    }
    os << '\n';
    return os;
}

std::vector<float> operator*(const std::vector<float>& v1, const std::vector<float>& v2)
{
    if (v1.size() != v2.size())
    {
        throw std::runtime_error("Vector sizes do not match for element-wise multiplication.");
    }

    std::vector<float> result;
    result.reserve(v1.size());

    for (std::size_t i = 0; i < v1.size(); i++)
    {
        result.push_back(v1[i] * v2[i]);
    }

    return result;
}

std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2)
{
    if (v1.size() != v2.size())
    {
        throw std::runtime_error("Vector sizes do not match for element-wise addition.");
    }

    std::vector<float> result;
    result.reserve(v1.size());

    for (std::size_t i = 0; i < v1.size(); i++)
    {
        result.push_back(v1[i] + v2[i]);
    }

    return result;
}

std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2)
{
    if (v1.size() != v2.size())
    {
        throw std::runtime_error("Vector sizes do not match for element-wise subtraction.");
    }

    std::vector<float> result;
    result.reserve(v1.size());

    for (std::size_t i = 0; i < v1.size(); i++)
    {
        result.push_back(v1[i] - v2[i]);
    }

    return result;
}

std::vector<float> cos(const std::vector<float>& v)
{
    std::vector<float> result;
    result.reserve(v.size());
    for (std::size_t i = 0; i < v.size(); i++)
    {
        result.push_back(cos(v[i]));
    }
    return result;
}

std::pair<int, bool> getRPNdepth(const std::vector<std::string>& expression)
{
    if (expression.empty())
    {
        return std::make_pair(0, false);
    }

    std::unordered_set<std::string> operators = {"cos", "+", "-", "*"};
    std::unordered_set<std::string> unary_operators = {"cos"};
    std::stack<int> stack;
    bool complete = true;

    for (const std::string& token : expression)
    {
        if (unary_operators.count(token) > 0)
        {
            stack.top() += 1;
        }
        else if (operators.count(token) > 0)
        {
            int op2 = stack.top();
            stack.pop();
            int op1 = stack.top();
            stack.pop();
            stack.push(std::max(op1, op2) + 1);
        }
        else
        {
            stack.push(1);
        }
    }

    while (stack.size() > 1)
    {
        int op2 = stack.top();
        stack.pop();
        int op1 = stack.top();
        stack.pop();
        stack.push(std::max(op1, op2) + 1);
        complete = false;
    }

    return std::make_pair(stack.top() - 1, complete);
}

std::pair<int, bool> getPNdepth(const std::vector<std::string>& expression)
{
    if (expression.empty() || (expression.size() == 1 && expression[0] == " "))
    {
        return std::make_pair(0, false);
    }

    std::vector<int> stack;
    int depth = 0, num_binary = 0, num_leaves = 0;
    std::unordered_set<std::string> binary_operators = {"+", "-", "*"};
    std::unordered_set<std::string> unary_operators = {"cos"};

    for (const std::string& val : expression)
    {
        if (binary_operators.count(val) > 0)
        {
            stack.push_back(2);  // Number of operands
            num_binary++;
        }
        else if (unary_operators.count(val) > 0)
        {
            stack.push_back(1);
        }
        else
        {
            num_leaves++;
            while (!stack.empty() && stack.back() == 1)
            {
                stack.pop_back();  // Remove fulfilled operators
            }
            if (!stack.empty())
            {
                stack.back()--;  // Indicate an operand is consumed
            }
        }
        depth = std::max(depth, static_cast<int>(stack.size()) + 1);
    }
    return std::make_pair(depth - 1, num_leaves == num_binary + 1);
}

float MSE(const std::vector<float>& actual, const std::vector<float>& predicted)
{
    if (actual.size() != predicted.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    float mse = 0.0f;
    for (size_t i = 0; i < actual.size(); i++)
    {
        float error = actual[i] - predicted[i];
        mse += error * error;
    }
    
    return mse/actual.size();
}

float loss_func(const std::vector<float>& actual, const std::vector<float>& predicted)
{
    return (1.0f/(1.0f+MSE(actual, predicted)));
}

struct Board
{
    static std::string inline best_expression = "";
    static std::unordered_map<std::string, int> inline expression_dict = {};
    static unsigned int inline expression_dict_len = 0;
    static float inline best_loss = INT_MAX;
    static std::vector<std::string> inline init_expression = {};
    static float inline search_time = 0;
    
    int __num_features;
    size_t data_size;
    std::vector<std::string> __input_vars;
    std::vector<std::string> __unary_operators;
    std::vector<std::string> __binary_operators;
    std::vector<std::string> __operators;
    std::vector<std::string> __other_tokens;
    std::vector<std::string> __tokens;
    std::vector<float> __tokens_float;
    Data data;
    
    std::vector<float> __operators_float;
    std::vector<float> __unary_operators_float;
    std::vector<float> __binary_operators_float;
    std::vector<float> __input_vars_float;
    std::vector<float> __other_tokens_float;
    
    int action_size;
                
    std::unordered_map<float, std::string> __tokens_dict; //Converts number to string
    std::unordered_map<std::string, float> __tokens_inv_dict; //Converts string to number

    int n; //depth of RPN tree
    std::string expression_type;
    // Create the empty expression list.
    std::vector<float> pieces;
    bool visualize_exploration;
    
    Board(const std::vector<std::vector<float>>& theData, int n = 3, const std::string& expression_type = "prefix", bool visualize_exploration = false) : data{theData}
    {
        this->__num_features = data[0].size() - 1;
        this->data_size = data.size();
        this->__input_vars.reserve(this->__num_features);
        for (auto i = 0; i < this->__num_features; i++)
        {
            this->__input_vars.push_back("x"+std::to_string(i));
        }
        this->__unary_operators = {"cos"};
        this->__binary_operators = {"+", "-", "*"};
        this->__operators = {"cos", "+", "-", "*"};
        this->__other_tokens = {"const"};
        this->__tokens = {"cos", "+", "-", "*"};
        for (auto& i: this->__input_vars)
        {
            this->__tokens.push_back(i);
        }
        for (auto& i: this->__other_tokens)
        {
            this->__tokens.push_back(i);
        }

        this->action_size = this->__tokens.size();
        this->__tokens_float.reserve(this->action_size);
        for (int i = 1; i <= this->action_size; ++i)
        {
            this->__tokens_float.push_back(i);
        }

        int num_operators = this->__operators.size();
        int num_unary_operators = this->__unary_operators.size();
        for (int i = 1; i <= num_operators; i++)
        {
            this->__operators_float.push_back(i);
        }
        for (int i = 1; i <= num_unary_operators; i++)
        {
            this->__unary_operators_float.push_back(i);
        }
        for (int i = num_unary_operators + 1; i <= num_operators; i++)
        {
            this->__binary_operators_float.push_back(i);
        }
        int ops_plus_features = num_operators + this->__num_features;
        for (int i = num_operators + 1; i <= ops_plus_features; i++)
        {
            this->__input_vars_float.push_back(i);
        }
        for (int i = ops_plus_features + 1; i <= this->action_size; i++)
        {
            this->__other_tokens_float.push_back(i);
        }
        for (int i = 0; i < this->action_size; i++)
        {
            this->__tokens_dict[this->__tokens_float[i]] = this->__tokens[i];
            this->__tokens_inv_dict[this->__tokens[i]] = this->__tokens_float[i];
        }
        this->n = n;
        this->expression_type = expression_type;
        this->pieces = {};
        this->visualize_exploration = visualize_exploration;
    }
    
    float operator[](size_t index) const
    {
        if (index < this->__tokens_float.size())
        {
            return this->__tokens_float[index];
        }
        throw std::out_of_range("Index out of range");
    }
    
    int __num_binary_ops()
    {
        int count = 0;
        for (float token : pieces)
        {
            if (std::find(__binary_operators_float.begin(), __binary_operators_float.end(), token) != __binary_operators_float.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_unary_ops()
    {
        int count = 0;
        for (float token : pieces)
        {
            if (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), token) != __unary_operators_float.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_leaves()
    {
        int count = 0;
        std::vector<float> leaves = __input_vars_float;
        leaves.insert(leaves.end(), __other_tokens_float.begin(), __other_tokens_float.end());

        for (float token : pieces)
        {
            if (std::find(leaves.begin(), leaves.end(), token) != leaves.end())
            {
                count++;
            }
        }
        return count;
    }
    
    std::vector<float> get_legal_moves()
    {
        //TODO: Reduce calls to size method -> See if multiple of the same calls can be stored in variables.
        //TODO: Probability for stupid moves should be 0 (i.e. binary_op(const, const), -(leaf_i, leaf_i)); this isn't GP!
        if (this->expression_type == "prefix")
        {
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the operators
            {
                return this->__operators_float;
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
            
            //__tokens_dict: converts float to string
            std::vector<std::string> string_pieces;
            string_pieces.reserve(this->pieces.size()+1);
            for (float i: this->pieces)
            {
                string_pieces.push_back(this->__tokens_dict[i]);
            }
            string_pieces.push_back(__binary_operators[0]);
            
            std::vector<float> temp;
            
            if (getPNdepth(string_pieces).first <= this->n)
            {
                temp.insert(temp.end(), this->__binary_operators_float.begin(), this->__binary_operators_float.end());
            }
            string_pieces[string_pieces.size() - 1] = __unary_operators[0];
            if (getPNdepth(string_pieces).first <= this->n)
            {
                temp.insert(temp.end(), this->__unary_operators_float.begin(), this->__unary_operators_float.end());
            }

            string_pieces[string_pieces.size() - 1] = __input_vars[0];
            //The number of leaves can never exceed number of binary + 1 in any RPN expression
            if (!((num_leaves == num_binary + 1) || (getPNdepth(string_pieces).first < this->n && (num_leaves == num_binary))))
            {
                temp.insert(temp.end(), this->__input_vars_float.begin(), this->__input_vars_float.end()); //leaves allowed
                if (std::find(__unary_operators.begin(), __unary_operators.end(), string_pieces[string_pieces.size()-2]) == __unary_operators.end())
                {
                    temp.insert(temp.end(), this->__other_tokens_float.begin(), this->__other_tokens_float.end());
                }
            }
            return temp;
        }
        else //postfix
        {
            std::vector<float> temp;
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the features and const
            {
                temp.insert(temp.end(), this->__input_vars_float.begin(), this->__input_vars_float.end());
                temp.insert(temp.end(), this->__other_tokens_float.begin(), this->__other_tokens_float.end());
                return temp;
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();

            if (num_binary != num_leaves - 1)
            {
                temp.insert(temp.end(), this->__binary_operators_float.begin(), this->__binary_operators_float.end());
            }
            std::vector<std::string> string_pieces;
            string_pieces.reserve(this->pieces.size()+1);
            for (float i: this->pieces)
            {
                string_pieces.push_back(this->__tokens_dict[i]);
            }
            string_pieces.push_back(__unary_operators[0]);
            if ((num_leaves >= 1) && (getRPNdepth(string_pieces).first <= this->n) && (std::find(__other_tokens.begin(), __other_tokens.end(), string_pieces[string_pieces.size()-2]) == __other_tokens.end())) //unary_op(const) is not allowed
            {
                temp.insert(temp.end(), this->__unary_operators_float.begin(), this->__unary_operators_float.end());
            }
            string_pieces[string_pieces.size() - 1] = __input_vars[0];
            if (getRPNdepth(string_pieces).first <= this->n)
            {
                temp.insert(temp.end(), this->__input_vars_float.begin(), this->__input_vars_float.end());
                temp.insert(temp.end(), this->__other_tokens_float.begin(), this->__other_tokens_float.end());
            }
            return temp;
        }
    }

    std::string pn_to_infix()
    {
        if (this->expression_type == "postfix")
        {
            throw std::runtime_error("Can't call pn_to_infix when your expression type is postfix!");
        }
        std::stack<std::string> stack;
        for (int i = pieces.size() - 1; i >= 0; i--) //i has to be int because it becomes -1 at the last step
        {
            std::string token = __tokens_dict[pieces[i]];

            if (std::find(__operators_float.begin(), __operators_float.end(), pieces[i]) == __operators_float.end()) // leaf
            {
                stack.push(token);
            }
            else if (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), pieces[i]) != __unary_operators_float.end()) // Unary operator
            {
                std::string operand = stack.top();
                stack.pop();
                std::string result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = stack.top();
                stack.pop();
                std::string left_operand = stack.top();
                stack.pop();
                std::string result = "(" + right_operand + " " + token + " " + left_operand + ")";
                stack.push(result);
            }
        }
        return stack.top();
    }
    
    std::string expression()
    {
        std::string temp;
        size_t sz = pieces.size() - 1;
        for (size_t i = 0; i <= sz; i++)
        {
            temp += ((i!=sz) ? __tokens_dict[pieces[i]] + " " : __tokens_dict[pieces[i]]);
        }
        return temp;
    }

    std::string rpn_to_infix()
    {
        if (this->expression_type == "prefix")
        {
            throw std::runtime_error("Can't call rpn_to_infix when your expression type is prefix!");
        }
        std::stack<std::string> stack;
        for (size_t i = 0; i < pieces.size(); i++)
        {
            std::string token = __tokens_dict[pieces[i]];

            if (std::find(__operators_float.begin(), __operators_float.end(), pieces[i]) == __operators_float.end()) // leaf
            {
                // std::cout << "leaf " << token << '\n';
                stack.push(token);
            }
            else if (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), pieces[i]) != __unary_operators_float.end()) // Unary operator
            {
                // std::cout << "unary op " << token << '\n';
                std::string operand = stack.top();
                stack.pop();
                std::string result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                // std::cout << "binary op " << token << '\n';
                std::string right_operand = stack.top();
                stack.pop();
                std::string left_operand = stack.top();
                stack.pop();
                std::string result = "(" + left_operand + " " + token + " " + right_operand + ")";
                stack.push(result);
            }
        }
        return stack.top();
    }
    //TODO: constant optimization
    std::vector<float> prefix_expression_evaluator(std::vector<float>& params)
    {
        if (this->expression_type == "postfix")
        {
            throw std::runtime_error("Can't call prefix_expression_evaluator when your expression type is postfix!");
        }
        std::stack<std::vector<float>> stack;

        for (int i = pieces.size() - 1; i >= 0; i--) //i has to be int because it becomes -1 at the last step
        {
            std::string token = __tokens_dict[pieces[i]];

            if (std::find(__operators_float.begin(), __operators_float.end(), pieces[i]) == __operators_float.end()) // leaf
            {
                if (token == "const")
                {
                    stack.push(std::vector<float>(data_size, 1));
                }
                else
                {
                    stack.push(this->data[token]);
                }
            }
            else if (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), pieces[i]) != __unary_operators_float.end()) // Unary operator
            {
                if (token == "cos")
                {
                    std::vector<float> temp = stack.top();
                    stack.pop();
                    stack.push(cos(temp));
                }
            }
            else // binary operator
            {
                std::vector<float> right_operand = stack.top();
                stack.pop();
                std::vector<float> left_operand = stack.top();
                stack.pop();
                
                if (token == "+")
                {
                    stack.push(right_operand + left_operand);
                }
                else if (token == "-")
                {
                    stack.push(right_operand - left_operand);
                }
                else if (token == "*")
                {
                    stack.push(right_operand * left_operand);
                }
            }
        }
        return stack.top();
    }
    //TODO: constant optimization
    std::vector<float> postfix_expression_evaluator()
    {
        if (this->expression_type == "prefix")
        {
            throw std::runtime_error("Can't call postfix_expression_evaluator when your expression type is prefix!");
        }
        std::stack<std::vector<float>> stack;

        for (size_t i = 0; i < pieces.size(); i++)
        {
            std::string token = __tokens_dict[pieces[i]];

            if (std::find(__operators_float.begin(), __operators_float.end(), pieces[i]) == __operators_float.end()) // leaf
            {
                if (token == "const")
                {
                    stack.push(std::vector<float>(data_size, 1));
                }
                else
                {
                    stack.push(this->data[token]);
                }
            }
            else if (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), pieces[i]) != __unary_operators_float.end()) // Unary operator
            {
                if (token == "cos")
                {
                    std::vector<float> temp = stack.top();
                    stack.pop();
                    stack.push(cos(temp));
                }
            }
            else // binary operator
            {
                std::vector<float> left_operand = stack.top();
                stack.pop();
                std::vector<float> right_operand = stack.top();
                stack.pop();

                if (token == "+")
                {
                    stack.push(right_operand + left_operand);
                }
                else if (token == "-")
                {
                    stack.push(right_operand - left_operand);
                }
                else if (token == "*")
                {
                    stack.push(right_operand * left_operand);
                }
            }
        }
        return stack.top();
    }
    
//    void fitFunctionToData(std::vector<float>& params)
//    {
//
//        // Use an optimization algorithm to minimize the least squares error (e.g., gradient descent)
//        // You should replace this with your preferred optimization method.
//
//        // Example: Simple gradient descent (you might need a more advanced optimizer)
//        float learning_rate = 0.01f;
//        int max_iterations = 1000;
//        size_t num_params = params.size();
//        std::vector<float> y_pred;
//        for (int iteration = 0; iteration < max_iterations; ++iteration)
//        {
//            // Calculate the gradient of the error with respect to parameters
//            std::vector<float> gradient(num_params, 0.0f);
//            for (const std::vector<float>& row : data)
//            {
//                y_pred = std::move((expression_type == "prefix") ? prefix_expression_evaluator() : postfix_expression_evaluator());
//                
//                gradient[i] += 2 * (y_pred - data["y"]) * row[i];
//                
//            }
//
//            // Update parameters using the gradient
//            for (size_t i = 0; i < num_params; ++i)
//            {
//                best_params[i] -= learning_rate * gradient[i];
//            }
//        }
//
//        return best_params;
//    }
    
    /*
    Check whether the given player has created a
    complete (depth self.n) expression (again), and
    checks if it is a complete RPN expression.
    Returns the score of the expression if complete,
    where 0 <= score <= 1 and -1 if not complete or if
    the desired depth has not been reached.
    */
    float complete_status()
    {
        std::vector<std::string> expression;
        expression.reserve(pieces.size());
        std::string temp_token;
        unsigned short num_consts = 0;
        
        for (float i: pieces)
        {
            temp_token = __tokens_dict[i];
            if (temp_token == "const"){++num_consts;}
            expression.push_back(__tokens_dict[i]);
        }

        auto [depth, complete] =  ((expression_type == "prefix") ? getPNdepth(expression) : getRPNdepth(expression)); //structured binding :)

        if (!complete || depth < this->n) //Expression not complete
        {
            return -1;
        }
        else
        {
            std::string expression_string = std::accumulate(expression.begin(), expression.end(), std::string(), [](const std::string& a, const std::string& b) { return a + (a.empty() ? "" : " ") + b; });
            Board::expression_dict[expression_string]++;
            Board::expression_dict_len = Board::expression_dict.size();
            if (visualize_exploration)
            {
                //TODO: call some plotting function
            }
            //TODO: Fit routine
            std::vector<float> params(num_consts, 1.0f);
            
            
            
            return ((expression_type == "prefix") ?
                    loss_func(prefix_expression_evaluator(params),data["y"]) :
                    loss_func(postfix_expression_evaluator(),data["y"]));
        }
    }
    const std::vector<float>& operator[] (int i){return data[i];}
    const std::vector<float>& operator[] (const std::string& i)
    {
        return data[i];
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Board& b)
    {
        return (os << b.data);
    }
};

// 2.5382*cos(x_3) + x_0^2 - 1
float exampleFunc(const std::vector<float>& x)
{
    if (x.size() < 3)
    {
        throw std::runtime_error("exampleFunc requires a vector of size 3");
    }

    return cos(x[3]) + (x[0]*x[0]) - 0.5f;
}

void Example()
{
    std::vector<std::vector<float>> data = generateData(10, 6, 0, 10, exampleFunc);
    
    std::vector<std::string> temp = {"cos", "-", "+", "+", "-", "+", "-", "+", "*", "*", "const", "*", "x0", "const", "*", "-", "const", "x0", "x1", "*", "x0", "const", "*", "*", "const", "const", "*", "x0", "const", "*", "*", "x1", "const", "const", "*", "const", "x1", "*", "const", "+", "+", "const", "*", "x0", "const", "*", "x1", "const", "*", "*", "x1", "const", "const", "x0"};
    std::cout << getPNdepth(temp).first << '\n';
    Board x(data, 11, "prefix");
    std::cout << x.data << '\n' << x.data["y"] << '\n';
    std::cout << "x complete = " << x.complete_status() << '\n';
    for (auto i: x.get_legal_moves()){std::cout << i << ' ';}puts("");
    //{"cos" -> 1, "+" -> 1, "-" -> 1, "*" -> 1, "x0" -> 0, "x1" -> 0, "const" -> 0}
    x.pieces.push_back(x.__tokens_inv_dict["-"]);
    std::cout << "x complete = " << x.complete_status() << '\n';
    for (auto i: x.get_legal_moves()){std::cout << i << ' ';}puts("");
    //{"cos" -> 1, "+" -> 1, "-" -> 1, "*" -> 1, "x0" -> 1, "x1" -> 1, "const" -> 1}
    x.pieces.clear();
    for (const std::string& i: temp){ x.pieces.push_back(x.__tokens_inv_dict[i]);}
    for (float i: x.pieces){std::cout << x.__tokens_dict[i] << ' ';}puts("");
    std::cout << x.pn_to_infix() << '\n';
    std::cout << "x complete = " << x.complete_status() << '\n';
    
    std::vector<float> params;
    std::cout << loss_func(x.prefix_expression_evaluator(params), x.data["y"]) << '\n';
    
    Board y(data, 3, "postfix");
    std::cout << y << '\n' << y["y"] << '\n';
    std::cout << "y complete = " << y.complete_status() << '\n';
    temp = {"const", "x3", "cos", "*", "x0", "x0", "*", "const", "-", "+"};
    std::cout << getRPNdepth(temp).first << '\n';
    for (auto i: y.get_legal_moves()){std::cout << i << ' ';}puts("");
    //{"cos" -> 0, "+" -> 0, "-" -> 0, "*" -> 0, "x0" -> 1, "x1" -> 1, "const" -> 1}
    y.pieces.push_back(y.__tokens_inv_dict["x0"]);
    std::cout << "y complete = " << y.complete_status() << '\n';
    for (auto i: y.get_legal_moves()){std::cout << i << ' ';}puts("");
    //{"cos" -> 1, "+" -> 0, "-" -> 0, "*" -> 0, "x0" -> 1, "x1" -> 1, "const" -> 1}
    y.pieces.clear();
    for (const std::string& i: temp){ y.pieces.push_back(y.__tokens_inv_dict[i]);}
    for (float i: y.pieces){std::cout << y.__tokens_dict[i] << ' ';}puts("");
    std::cout << y.rpn_to_infix() << '\n';
    std::cout << "y complete = " << y.complete_status() << '\n';
    
    std::cout << loss_func(y.postfix_expression_evaluator(), y.data["y"]) << '\n';
    
    
}

int main() {
    
//    Example();
    
    std::vector<std::vector<float>> data = generateData(100, 6, 0.0f, 10.0f, exampleFunc);
    Board x(data, 3, "prefix");
    
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
    float score = 0, max_score = 0;
    std::vector<float> temp;
    size_t temp_sz;
    std::string expression, orig_expression, best_expression;
    std::ofstream out("PN_expressions.txt");
    auto start_time = Clock::now();
    for (int i = 0; score <= 0.59; i++)
    {
        while ((score = x.complete_status()) == -1)
        {
            temp = std::move(x.get_legal_moves());
            temp_sz = temp.size();
            std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // Define the range

            x.pieces.push_back(temp[distribution(generator)]);
        }
        
        expression = x.pn_to_infix();
        
        out << "Iteration " << i << ": Original expression = " << x.expression() << ", Infix Expression = " << expression << '\n';

        if (score > max_score)
        {
            std::cout << "Best score = " << max_score << '\n';
            std::cout << "Best expression = " << best_expression << '\n';
            std::cout << "Best expression (original format) = " << orig_expression << '\n';
            max_score = score;
            best_expression = std::move(expression);
            orig_expression = x.expression();
        }
        x.pieces.clear();
    }
    out.close();
    std::cout << "\nUnique expressions = " << Board::expression_dict_len << '\n';
    std::cout << "Best score = " << max_score << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
       
    auto end_time = Clock::now();
    std::cout << "Time difference = "
          << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()/1e9 << " seconds" << '\n';
    
    
    return 0;
}
