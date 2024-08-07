/*
 Symbolic Differential Equation Solver
 =====================================
  - Data of the form (m x N), where m is the number of attributes and N the number of data points
  - Symbolic Regressor creates an equation
  - 
 
 
 */

#include <vector>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <future>         // std::async, std::future
#include <unordered_map>
#include <map>
#include <ctime>
#include <cstdlib>
#include <stack>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cassert>
#include <thread>
#include <mutex>
#include <atomic>
#include <latch>
#include <LBFGS.h>
#include <LBFGSB.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include <boost/unordered/concurrent_flat_map.hpp>

using Clock = std::chrono::high_resolution_clock;

//Returns the number of seconds since `start_time`
template <typename T>
double timeElapsedSince(T start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
}

bool isFloat(const std::string& x)
{
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

// Function to create a matrix with linspace columns. std::vector<float> min and
// std::vector<float> max must have size == cols
Eigen::MatrixXf createLinspaceMatrix(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec) 
{
    assert((cols == min_vec.size()) && (cols == max_vec.size()));
    Eigen::MatrixXf mat(rows, cols);
    for (int col = 0; col < cols; ++col) 
    {
        for (int row = 0; row < rows; ++row) 
        {
            mat(row, col) = min_vec[col] + (max_vec[col] - min_vec[col]) * row / (rows - 1);
        }
    }
    return mat;
}

Eigen::MatrixXf generateData(int numRows, int numCols, float min = -3.0f, float max = 3.0f)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 thread_local gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);

    // Create the matrix
    Eigen::MatrixXf matrix(numRows, numCols);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            matrix(i, j) = distribution(gen);
        }
    }

    return matrix;
}

int trueMod(int N, int M)
{
    return ((N % M) + M) % M;
};

class Data
{
    Eigen::MatrixXf data;
    std::unordered_map<std::string, Eigen::VectorXf> features;
    std::vector<Eigen::VectorXf> rows;
    long num_columns, num_rows;
    
public:
    
    Data() = default; //so we can have a static Data attribute
    
    // Assignment operator
    Data& operator=(const Eigen::MatrixXf& theData)
    {
        this->data = theData;
        this->num_columns = data.cols();
        this->num_rows = data.rows();
        for (size_t i = 0; i < this->num_columns; i++) //for each column
        {
            this->features["x"+std::to_string(i)] = Eigen::VectorXf(this->num_rows);
            for (size_t j = 0; j < this->num_rows; j++)
            {
                this->features["x"+std::to_string(i)](j) = this->data(j,i);
            }
        }
        this->rows.resize(this->num_rows);
        
        for (size_t i = 0; i < num_rows; i++)
        {
            this->rows[i] = data.row(i);
        }
        
        return *this;
    }
    
    bool operator==( Data& other)
    {
        return this->data == other.data;
    }
    
    const Eigen::VectorXf& operator[] (int i){return rows[i];}
    const Eigen::VectorXf& operator[] (const std::string& i)
    {
        return features[i];
    }
    long numRows() const {return num_rows;}
    long numCols() const {return num_columns;}

    friend std::ostream& operator<<(std::ostream& os, const Data& matrix)
    {
        return (os << matrix.data);
    }
};

float MSE(const Eigen::VectorXf& actual)
{
    return actual.squaredNorm() / actual.size();
}

Eigen::AutoDiffScalar<Eigen::VectorXf> MSE(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>& actual)
{
    return actual.squaredNorm() / actual.size();
}

float loss_func(const Eigen::VectorXf& actual)
{
    return (1.0f/(1.0f+MSE(actual)));
}

struct Board
{
    static boost::concurrent_flat_map<std::string, Eigen::VectorXf> inline expression_dict = {};
    static float inline best_loss = FLT_MAX;
    static std::atomic<float> inline fit_time = 0.0;
    
    static constexpr float K = 0.0884956f;
    static constexpr float phi_1 = 2.8f;
    static constexpr float phi_2 = 1.3f;
    static int inline __num_features;
    static std::vector<std::string> inline __input_vars;
    static std::vector<std::string> inline __unary_operators;
    static std::vector<std::string> inline __binary_operators;
    static std::vector<std::string> inline __operators;
    static std::vector<std::string> inline __other_tokens;
    static std::vector<std::string> inline __tokens;
    static std::vector<float> inline __tokens_float;
    Eigen::VectorXf params; //store the parameters of the expression of the current episode after it's completed
    static Data inline data;
    
    static std::vector<float> inline __operators_float;
    static std::vector<float> inline __unary_operators_float;
    static std::vector<float> inline __binary_operators_float;
    static std::vector<float> inline __input_vars_float;
    static std::vector<float> inline __other_tokens_float;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> vel_dist, pos_dist;
    
    static int inline action_size;
    static std::once_flag inline initialization_flag;  // Flag for std::call_once
    
    size_t reserve_amount;
    int num_fit_iter;
    std::string fit_method;
    std::string fit_grad_method;
    
    bool cache;
    std::vector<int> stack;
    int depth = 0, num_binary = 0, num_leaves = 0, idx = 0;
    static std::unordered_map<float, std::string> inline __tokens_dict; //Converts number to string
    static std::unordered_map<std::string, float> inline __tokens_inv_dict; //Converts string to number
    static std::unordered_map<bool, std::unordered_map<bool, std::unordered_map<bool, std::vector<float>>>> inline una_bin_leaf_legal_moves_dict;

    int n; //depth of RPN/PN tree
    std::string expression_type, expression_string;
    static std::mutex inline thread_locker; //static because it needs to protect static members
    std::vector<float> pieces; // Create the empty expression list.
    std::vector<float> derivat;// Vector to store the derivative.
    bool visualize_exploration, is_primary;
    std::vector<float> (*diffeq)(Board&); //differential equation we want to solve
    std::vector<float> diffeq_result;
    
    Board(std::vector<float> (*diffeq)(Board&), bool primary = true, int n = 3, const std::string& expression_type = "prefix", std::string fitMethod = "PSO", int numFitIter = 1, std::string fitGradMethod = "naive_numerical", const Eigen::MatrixXf& theData = {}, bool visualize_exploration = false, bool cache = false) : gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, num_fit_iter{numFitIter}, fit_method{fitMethod}, fit_grad_method{fitGradMethod}, is_primary{primary}
    {
        if (n > 30)
        {
            throw(std::runtime_error("Complexity cannot be larger than 30, sorry!"));
        }
        
        this->n = n;
        this->expression_type = expression_type;
        this->pieces = {};
        this->visualize_exploration = visualize_exploration;
        this->reserve_amount = 2*std::pow(2,this->n)-1;
        this->pieces.reserve(this->reserve_amount);
        this->cache = cache;
        this->diffeq = diffeq;
        
        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                Board::data = theData;
                Board::__num_features = Board::data[0].size();
                Board::__input_vars.clear();
                Board::expression_dict.clear();
                Board::__input_vars.reserve(Board::__num_features);
                for (auto i = 0; i < Board::__num_features; i++)
                {
                    Board::__input_vars.push_back("x"+std::to_string(i));
                }
                Board::__unary_operators = {"cos", "~", "sin", "log", "ln", "asin", "arcsin", "acos", "arccos", "exp", "sech", "tanh", "sqrt"};
                Board::__binary_operators = {"+", "-", "*", "/", "^"};
                Board::__operators.clear();
                for (std::string& i: Board::__unary_operators)
                {
                    Board::__operators.push_back(i);
                }
                for (std::string& i: Board::__binary_operators)
                {
                    Board::__operators.push_back(i);
                }
                Board::__other_tokens = {/*"const",*/ "0", "1", "2"};
                Board::__tokens = Board::__operators;
                
                for (auto& i: this->Board::__input_vars)
                {
                    Board::__tokens.push_back(i);
                }
                for (auto& i: Board::__other_tokens)
                {
                    Board::__tokens.push_back(i);
                }
                Board::action_size = Board::__tokens.size();
                Board::__tokens_float.clear();
                Board::__tokens_float.reserve(Board::action_size);
                for (int i = 1; i <= Board::action_size; ++i)
                {
                    Board::__tokens_float.push_back(i);
                }
                int num_operators = Board::__operators.size();
                Board::__operators_float.clear();
                for (int i = 1; i <= num_operators; i++)
                {
                    Board::__operators_float.push_back(i);
                }
                int num_unary_operators = Board::__unary_operators.size();
                Board::__unary_operators_float.clear();
                for (int i = 1; i <= num_unary_operators; i++)
                {
                    Board::__unary_operators_float.push_back(i);
                }
                Board::__binary_operators_float.clear();
                for (int i = num_unary_operators + 1; i <= num_operators; i++)
                {
                    Board::__binary_operators_float.push_back(i);
                }
                int ops_plus_features = num_operators + Board::__num_features;
                Board::__input_vars_float.clear();
                for (int i = num_operators + 1; i <= ops_plus_features; i++)
                {
                    Board::__input_vars_float.push_back(i);
                }
                Board::__other_tokens_float.clear();
                for (int i = ops_plus_features + 1; i <= Board::action_size; i++)
                {
                    Board::__other_tokens_float.push_back(i);
                }
                for (int i = 0; i < Board::action_size; i++)
                {
                    Board::__tokens_dict[Board::__tokens_float[i]] = Board::__tokens[i];
                    Board::__tokens_inv_dict[Board::__tokens[i]] = Board::__tokens_float[i];
                }
                
                Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__tokens_float;
                Board::una_bin_leaf_legal_moves_dict[true][true][false] = Board::__operators_float;
                Board::una_bin_leaf_legal_moves_dict[true][false][true] = Board::__unary_operators_float; //1
                Board::una_bin_leaf_legal_moves_dict[true][false][false] = Board::__unary_operators_float;
                Board::una_bin_leaf_legal_moves_dict[false][true][true] = Board::__binary_operators_float; //2
                Board::una_bin_leaf_legal_moves_dict[false][true][false] = Board::__binary_operators_float;
                
                for (float i: Board::__input_vars_float)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                    Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                    Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                }
                for (float i: Board::__other_tokens_float)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                    Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                    Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                }
            });
        }
    }
    
    float operator[](size_t index) const
    {
        if (index < Board::__tokens_float.size())
        {
            return Board::__tokens_float[index];
        }
        throw std::out_of_range("Index out of range");
    }
    
    int __num_binary_ops() const
    {
        int count = 0;
        for (float token : pieces)
        {
            if (std::find(Board::__binary_operators_float.begin(), Board::__binary_operators_float.end(), token) != Board::__binary_operators_float.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_unary_ops() const
    {
        int count = 0;
        for (float token : pieces)
        {
            if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), token) != Board::__unary_operators_float.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_leaves() const
    {
        int count = 0;

        for (float token : pieces)
        {
            if (!is_unary(token) && !is_binary(token))
            {
                count++;
            }
        }
        return count;
    }
    
    int __num_consts() const
    {
        int count = 0;

        for (float token : pieces)
        {
            if (__tokens_dict[token] == "const")
            {
                count++;
            }
        }
        return count;
    }
    
    bool is_unary(float token) const
    {
        return (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), token) != __unary_operators_float.end());
    }

    bool is_binary(float token) const
    {
        return (std::find(__binary_operators_float.begin(), __binary_operators_float.end(), token) != __binary_operators_float.end());
    }
    
    bool is_operator(float token) const
    {
        return (is_binary(token) || is_unary(token));
    }
    
    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77180279
     */
    std::pair<int, bool> getPNdepth(const std::vector<float>& expression, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool binary = false, bool unary = false, bool leaf = false)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }
        
        if (stop == 0)
        {
            stop = expression.size();
        }

        if (!cache)
        {
            this->stack.clear();
            this->depth = 0, this->num_binary = 0, this->num_leaves = 0;
            for (size_t i = start; i < stop; i++)
            {
                if (is_binary(expression[i]))
                {
                    this->stack.push_back(2);  // Number of operands
                    this->num_binary++;
                }
                else if (is_unary(expression[i]))
                {
                    this->stack.push_back(1);
                }
                else
                {
                    this->num_leaves++;
                    while (!this->stack.empty() && this->stack.back() == 1) //so the this->stack will shrink one by one from the back until it's empty and/or the last element is NOT 1
                    {
                        this->stack.pop_back();  // Remove fulfilled operators
                    }
                    if (!this->stack.empty())
                    {
                        this->stack.back()--;  // Indicate an operand is consumed
                    }
                }
                this->depth = std::max(this->depth, static_cast<int>(this->stack.size()) + 1);
            }
        }
        else //optimize with caching
        {
            if (not modify) //get_legal_moves()
            {
                if (binary) //Gives the this->depth and completeness of the current PN expression + a binary operator
                {
                    return std::make_pair(std::max(this->depth, static_cast<int>(this->stack.size()) + 2) - 1, this->num_leaves == this->num_binary + 2);
                }
                else if (unary) //Gives the this->depth and completeness of the current PN expression + a unary operator
                {
                    return std::make_pair(std::max(this->depth, static_cast<int>(this->stack.size()) + 2) - 1, this->num_leaves == this->num_binary + 1);
                }
                else if (leaf) //Gives the this->depth and completeness of the current PN expression + a leaf node
                {
                    auto last_filled_op_it = std::find_if(this->stack.rbegin(), this->stack.rend(), [](int i){return i != 1;}); //Find the first element from the back that's not 1
                    return std::make_pair(std::max(this->depth, static_cast<int>(this->stack.rend() - last_filled_op_it) /* this->stack.size() */ + 1) - 1, this->num_leaves == this->num_binary);
                }
            }
            else //modify -> complete_status()
            {
                if (is_binary(expression[this->idx]))
                {
                    this->stack.push_back(2);  // Number of operands
                    this->num_binary++;
                }
                else if (is_unary(expression[this->idx]))
                {
                    this->stack.push_back(1);
                }
                else
                {
                    this->num_leaves++;
                    while (!this->stack.empty() && this->stack.back() == 1) //so the this->stack will shrink one-by-one from the back until it's empty and/or the last element is NOT 1
                    {
                        this->stack.pop_back();  // Remove fulfilled operators
                    }
                    if (!this->stack.empty())
                    {
                        this->stack.back()--;  // Indicate an operand is consumed
                    }
                }
                this->depth = std::max(this->depth, static_cast<int>(this->stack.size()) + 1);
                this->idx++;
            }
        }
        return std::make_pair(this->depth - 1, this->num_leaves == this->num_binary + 1);
    }
    
    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77128902
     */
    std::pair<int, bool> getRPNdepth(const std::vector<float>& expression, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool unary = false, bool leaf = false)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }
        
        if (stop == 0)
        {
            stop = expression.size();
        }

        if (!cache)
        {
            this->stack.clear();
            bool complete = true;
            
            for (size_t i = start; i < stop; i++)
            {
                if (is_unary(expression[i]))
                {
                    this->stack.back() += 1;
                }
                else if (is_binary(expression[i]))
                {
                    int op2 = this->stack.back();
                    this->stack.pop_back();
                    int op1 = this->stack.back();
                    this->stack.pop_back();
                    this->stack.push_back(std::max(op1, op2) + 1);
                }
                else //leaf
                {
                    this->stack.push_back(1);
                }
            }
            
            while (this->stack.size() > 1)
            {
                int op2 = this->stack.back();
                this->stack.pop_back();
                int op1 = this->stack.back();
                this->stack.pop_back();
                this->stack.push_back(std::max(op1, op2) + 1);
                complete = false;
            }
            
            /*
             e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
             {1, 2, 3, 4, 5}
             {1, 2, 3, 6}
             {1, 2, 7}
             {1, 8}
             {9}
             */
            
            return std::make_pair(this->stack.back() - 1, complete);
        }
        else //optimize with caching
        {
            if (not modify)  //get_legal_moves()
            {
                if (unary) //Gives the this->depth and completeness of the current RPN expression + a unary operator
                {
                    if (this->stack.size() == 1)
                    {
                        return std::make_pair(this->stack.back(), true);
                    }
                    else
                    {
                        int curr_max = std::max(this->stack.back()+1, *(this->stack.end()-2))+1;
                        for (int i = this->stack.size() - 2; i >= 1; i--)
                        {
                            curr_max = std::max(curr_max, this->stack[i-1])+1;
                        }
                        
                        /*
                         e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
                         curr_max = max(5, 4)+1 = 6;
                         curr_max = max(6, 3)+1 = 7;
                         curr_max = max(7, 2)+1 = 8;
                         curr_max = max(8, 1)+1 = 9;
                         */
                        
                        return std::make_pair(curr_max - 1, false);
                    }
                }
                else if (leaf) //Gives the this->depth and completeness of the current RPN expression + a leaf node
                {
                    if (this->stack.empty())
                    {
                        return std::make_pair(0, true);
                    }
                    else
                    {
                        int curr_max = std::max(this->stack.back(), 1)+1;
                        for (int i = this->stack.size() - 1; i >= 1; i--)
                        {
                            curr_max = std::max(curr_max, this->stack[i-1])+1;
                        }
                        /*
                         e.g., assume this->stack = {1, 2, 3, 4, 5}, then:
                         curr_max = max(5, 4)+1 = 6;
                         curr_max = max(6, 3)+1 = 7;
                         curr_max = max(7, 2)+1 = 8;
                         curr_max = max(8, 1)+1 = 9;
                         */
                        
                        return std::make_pair(curr_max - 1, false);
                    }
                }
            }
            else //modify -> complete_status()
            {
                if (is_binary(expression[this->idx]))
                {
                    int op2 = this->stack.back();
                    this->stack.pop_back();
                    int op1 = this->stack.back();
                    this->stack.pop_back();
                    this->stack.push_back(std::max(op1, op2) + 1);
                }
                else if (is_unary(expression[this->idx]))
                {
                    this->stack.back() += 1;
                }
                else //leaf
                {
                    this->stack.push_back(1);
                }
                
                this->idx++;
                if (this->stack.size() == 1)
                {
                    return std::make_pair(this->stack.back() - 1, true);
                }
                
                else
                {
                    int curr_max = std::max(this->stack.back(), *(this->stack.end()-2))+1;
                    for (int i = this->stack.size() - 2; i >= 1; i--)
                    {
                        curr_max = std::max(curr_max, this->stack[i-1])+1;
                    }
                    return std::make_pair(curr_max - 1, false);
                }
            }
            
            return std::make_pair(this->stack.back() - 1, true);
        }
    }
    
    std::vector<float> get_legal_moves()
    {
        if (this->expression_type == "prefix")
        {
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the operators...
            {
                if (this->n != 0) // if the depth is not 0
                {
                    return Board::__operators_float;
                }
                else // else it's the leaves
                {
                    return Board::una_bin_leaf_legal_moves_dict[false][false][true];
                }
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
            
            if (this->cache)
            {
                return Board::una_bin_leaf_legal_moves_dict[(getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, true /*unary*/, false /*leaf*/).first <= this->n)][(getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*binary*/, false /*unary*/, false /*leaf*/).first <= this->n)][(!((num_leaves == num_binary + 1) || (getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, false /*unary*/, true /*leaf*/).first < this->n && (num_leaves == num_binary))))];
            }
            
            else
            {
                bool una_allowed = false, bin_allowed = false, leaf_allowed = false;
                if (Board::__binary_operators_float.size() > 0)
                {
                    pieces.push_back(Board::__binary_operators_float[0]);
                    bin_allowed = (getPNdepth(pieces).first <= this->n);
                }
                if (Board::__unary_operators_float.size() > 0)
                {
                    pieces[pieces.size() - 1] = Board::__unary_operators_float[0];
                    una_allowed = (getPNdepth(pieces).first <= this->n);
                }
                pieces[pieces.size() - 1] = Board::__input_vars_float[0];
                leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(pieces).first < this->n && (num_leaves == num_binary))));
                pieces.pop_back();
//                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));
                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }

        else //postfix
        {
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the features and const
            {
                return Board::una_bin_leaf_legal_moves_dict[false][false][true];
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
                 
            if (this->cache)
            {
                return Board::una_bin_leaf_legal_moves_dict[((num_leaves >= 1) && (getRPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*unary*/, false /*leaf*/).first <= this->n))][(num_binary != num_leaves - 1)][(getRPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*unary*/, true /*leaf*/).first <= this->n)];
            }
            
            else
            {
                bool una_allowed = false, bin_allowed = (num_binary != num_leaves - 1), leaf_allowed = false;
                if (Board::__unary_operators_float.size() > 0)
                {
                    pieces.push_back(Board::__unary_operators_float[0]);
                    una_allowed = ((num_leaves >= 1) && (getRPNdepth(pieces).first <= this->n));
                }
                
                pieces[pieces.size() - 1] = Board::__input_vars_float[0];
                leaf_allowed = (getRPNdepth(pieces).first <= this->n);

                pieces.pop_back();
//                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));

                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }

    }
    
    //Returns the `expression_type` string form of the expression stored in the vector<float> attribute pieces
    std::string expression()
    {
        std::string temp;
        temp.reserve(2*pieces.size());
        size_t sz = pieces.size() - 1;
        int const_index = ((expression_type == "postfix") ? 0 : this->params.size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            if (Board::__tokens_dict[pieces[i]] == "const")
            {
                temp += ((i!=sz) ? std::to_string((this->params)(const_index)) + " " : std::to_string((this->params)(const_index)));
                if (expression_type == "postfix")
                {
                    const_index++;
                }
                else
                {
                    const_index--;
                }
            }
            else
            {
                temp += ((i!=sz) ? Board::__tokens_dict[pieces[i]] + " " : Board::__tokens_dict[pieces[i]]);
            }
        }
        return temp;
    }
    
    std::string _to_infix(bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        size_t const_counter = 0;
        std::string result;
                
        for (int i = (is_prefix ? (pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[pieces[i]];

            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
            {
                stack.push( ((!show_consts) || token != "const") ? token : std::to_string((this->params)(const_counter++)));
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
            {
                std::string operand = stack.top();
                stack.pop();
                result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = stack.top();
                stack.pop();
                std::string left_operand = stack.top();
                stack.pop();
                if (expression_type == "prefix")
                {
                    result = "(" + right_operand + " " + token + " " + left_operand + ")";
                }
                else
                {
                    result = "(" + left_operand + " " + token + " " + right_operand + ")";
                }
                stack.push(result);
            }
        }
        
        return stack.top();
    }

    //Returns the `expression_type` string form of the expression stored in the vector<float> parameter pieces
    std::string expression(const std::vector<float>& pieces)
    {
        std::string temp;
        temp.reserve(2*pieces.size());
        size_t sz = pieces.size() - 1;
        int const_index = ((expression_type == "postfix") ? 0 : this->params.size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            if (Board::__tokens_dict[pieces[i]] == "const")
            {
                temp += ((i!=sz) ? std::to_string((this->params)(const_index)) + " " : std::to_string((this->params)(const_index)));
                if (expression_type == "postfix")
                {
                    const_index++;
                }
                else
                {
                    const_index--;
                }
            }
            else
            {
                temp += ((i!=sz) ? Board::__tokens_dict[pieces[i]] + " " : Board::__tokens_dict[pieces[i]]);
            }
        }
        return temp;
    }
    
    std::string _to_infix(const std::vector<float>& pieces, bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        size_t const_counter = 0;
        std::string result;
                
        for (int i = (is_prefix ? (pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[pieces[i]];

            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
            {
                stack.push( ((!show_consts) || token != "const") ? token : std::to_string((this->params)(const_counter++)));
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
            {
                std::string operand = stack.top();
                stack.pop();
                result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = stack.top();
                stack.pop();
                std::string left_operand = stack.top();
                stack.pop();
                if (expression_type == "prefix")
                {
                    result = "(" + right_operand + " " + token + " " + left_operand + ")";
                }
                else
                {
                    result = "(" + left_operand + " " + token + " " + right_operand + ")";
                }
                stack.push(result);
            }
        }
        
        return stack.top();
    }

    Eigen::VectorXf expression_evaluator(const Eigen::VectorXf& params, const std::vector<float>& pieces) const
    {
        std::stack<const Eigen::VectorXf> stack;
        size_t const_count = 0;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[pieces[i]];
            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
            {
                if (token == "const")
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*params(const_count++));
                }
                else if (token == "0")
                {
                    stack.push(Eigen::VectorXf::Zero(Board::data.numRows()));
                }
                else if (token == "1")
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows()));
                }
                else if (token == "2")
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*2.0f);
                }
                else if (isFloat(token))
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*std::stof(token));
                }
                else
                {
                    stack.push(Board::data[token]);
                }
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
            {
                if (token == "cos")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::VectorXf temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
            }
            else // binary operator
            {
                Eigen::VectorXf left_operand = stack.top();
                stack.pop();
                Eigen::VectorXf right_operand = stack.top();
                stack.pop();
                if (token == "+")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() + left_operand.array()) : (left_operand.array() + right_operand.array())));
                }
                else if (token == "-")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() - left_operand.array()) : (left_operand.array() - right_operand.array())));
                }
                else if (token == "*")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() * left_operand.array()) : (left_operand.array() * right_operand.array())));
                }
                else if (token == "/")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() / left_operand.array()) : (left_operand.array() / right_operand.array())));
                }
                else if (token == "^")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array().pow(left_operand.array())) : (left_operand.array().pow(right_operand.array()))));
                }
            }
        }
        return stack.top();
    }
    
    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> expression_evaluator(const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& parameters, const std::vector<float>& pieces) const
    {
        std::stack<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>> stack;
        size_t const_count = 0;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[pieces[i]];

            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
            {
                if (token == "const")
                {
//                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), parameters[const_count++]));
                }
                else if (token == "0")
                {
//                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 0.0f));
                }
                else if (token == "1")
                {
//                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 1.0f));
                }
                else if (token == "2")
                {
//                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), 2.0f));
                }
                else if (isFloat(token))
                {
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), std::stof(token)));
                }
                else
                {
                    stack.push(Board::data[token]);
                }
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
            {
                if (token == "cos")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
            }
            else // binary operator
            {
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> left_operand = stack.top();
                stack.pop();
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> right_operand = stack.top();
                stack.pop();

                if (token == "+")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() + left_operand.array()) : (left_operand.array() + right_operand.array())));
                }
                else if (token == "-")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() - left_operand.array()) : (left_operand.array() - right_operand.array())));
                }
                else if (token == "*")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() * left_operand.array()) : (left_operand.array() * right_operand.array())));
                }
                else if (token == "/")
                {
                    stack.push(((expression_type == "postfix") ? (right_operand.array() / left_operand.array()) : (left_operand.array() / right_operand.array())));
                }
                else if (token == "^")
                {
                    stack.push(((expression_type == "postfix") ? ((left_operand.array()*(right_operand.array().log())).exp()) : ((right_operand.array()*(left_operand.array().log())).exp())));
                }
            }
        }
        return stack.top();
    }
    
    bool AsyncPSO()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::VectorXf particle_positions(this->params.size()), x(this->params.size());
        Eigen::VectorXf v(this->params.size());
        float rp, rg;

        for (size_t i = 0; i < this->params.size(); i++)
        {
            particle_positions(i) = x(i) = pos_dist(gen);
            v(i) = vel_dist(gen);
        }

        float swarm_best_score = loss_func(expression_evaluator(this->params, this->pieces));
        float fpi = loss_func(expression_evaluator(particle_positions, this->pieces));
        float temp, fxi;
        
        if (fpi > swarm_best_score)
        {
            this->params = particle_positions;
            swarm_best_score = fpi;
            improved = true;
        }
        
        auto UpdateParticle = [&](int i)
        {
            for (int j = 0; j < this->num_fit_iter; j++)
            {
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i) = K*(v(i) + phi_1*rp*(particle_positions(i) - x(i)) + phi_2*rg*((this->params)(i) - x(i)));
                x(i) += v(i);
                
                fpi = loss_func(expression_evaluator(particle_positions, this->pieces)); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions, this->pieces)); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
                    (this->params)(i) = particle_positions(i);
                    improved = true;
                    swarm_best_score = fpi;
                }
            }
        };
        
        std::vector<std::future<void>> particles;
        particles.reserve(this->params.size());
        for (int i = 0; i < this->params.size(); i++)
        {
            particles.push_back(std::async(std::launch::async | std::launch::deferred, UpdateParticle, i));
        }
        for (auto& i: particles)
        {
            i.get();
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    bool PSO()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::VectorXf particle_positions(this->params.size()), x(this->params.size());
        Eigen::VectorXf v(this->params.size());
        float rp, rg;

        for (size_t i = 0; i < this->params.size(); i++)
        {
            particle_positions(i) = x(i) = pos_dist(gen);
            v(i) = vel_dist(gen);
        }

        float swarm_best_score = loss_func(expression_evaluator(this->params, this->pieces));
        float fpi = loss_func(expression_evaluator(particle_positions, this->pieces));
        float temp, fxi;
        
        if (fpi > swarm_best_score)
        {
            this->params = particle_positions;
            improved = true;
            swarm_best_score = fpi;
        }
        for (int j = 0; j < this->num_fit_iter; j++)
        {
            for (unsigned short i = 0; i < this->params.size(); i++) //number of particles
            {
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i) = K*(v(i) + phi_1*rp*(particle_positions(i) - x(i)) + phi_2*rg*((this->params)(i) - x(i)));
                x(i) += v(i);
                
                fpi = loss_func(expression_evaluator(particle_positions, this->pieces)); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions, this->pieces)); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
//                    printf("Iteration %d: Changing param %d from %f to %f. Score from %f to %f\n", j, i, (this->params)[i], particle_positions[i], swarm_best_score, fpi);
                    (this->params)(i) = particle_positions(i);
                    improved = true;
                    swarm_best_score = fpi;
                }
            }
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    Eigen::AutoDiffScalar<Eigen::VectorXf> grad_func(std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& inputs)
    {
        return MSE(expression_evaluator(inputs, this->pieces));
    }
    
    /*
     x: parameter vector: (x_0, x_1, ..., x_{x.size()-1})
     g: gradient evaluated at x: (g_0(x_0), g_1(x_1), ..., g_{g.size()-1}(x_{x.size()-1}))
     */
    float operator()(Eigen::VectorXf& x, Eigen::VectorXf& grad)
    {
        if (this->fit_method == "LBFGS" || this->fit_method == "LBFGSB")
        {
            float mse = MSE(expression_evaluator(x, this->pieces));
            if (this->fit_grad_method == "naive_numerical")
            {
                float low_b, temp;
                for (int i = 0; i < x.size(); i++) //finite differences wrt x evaluated at the current values x(i)
                {
                    //https://stackoverflow.com/a/38855586/18255427
                    temp = x(i);
                    x(i) -= 0.00001f;
                    low_b = MSE(expression_evaluator(x, this->pieces));
                    x(i) = temp + 0.00001f;
                    grad(i) = (MSE(expression_evaluator(x, this->pieces)) - low_b) / 0.00002f ;
                    x(i) = temp;
                }
            }

            else if (this->fit_grad_method == "autodiff")
            {
                size_t sz = x.size();
                std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>> inputs(sz);
                inputs.reserve(sz);
                for (size_t i = 0; i < sz; i++)
                {
                    inputs[i].value() = x(i);
                    inputs[i].derivatives() = Eigen::VectorXf::Unit(sz, i);
                }
                grad = grad_func(inputs).derivatives();
            }
            return mse;
        }
        else if (this->fit_method == "LevenbergMarquardt")
        {
            grad = (this->expression_evaluator(x, this->pieces) - Board::data["y"]);
        }
        return 0.f;
    }
    
    bool LBFGS()
    {
        bool improved = false;
        auto start_time = Clock::now();
        LBFGSpp::LBFGSParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSSolver<float, LBFGSpp::LineSearchMoreThuente> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;
        
        Eigen::VectorXf eigenVec = this->params;
        float mse = MSE(expression_evaluator(this->params, this->pieces));
        try
        {
            solver.minimize((*this), eigenVec, fx);
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}
        
//        printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
//            printf("mse = %f -> fx = %f\n", mse, fx);
            this->params = eigenVec;
            improved = true;
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    bool LBFGSB()
    {
        bool improved = false;
        auto start_time = Clock::now();
        LBFGSpp::LBFGSBParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSBSolver<float> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;
        
        Eigen::VectorXf eigenVec = this->params;
        float mse = MSE(expression_evaluator(this->params, this->pieces));
        try
        {
            solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -std::numeric_limits<float>::infinity()), Eigen::VectorXf::Constant(eigenVec.size(), std::numeric_limits<float>::infinity()));
//            solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -10.f), Eigen::VectorXf::Constant(eigenVec.size(), 10.f));
        }
        catch (std::runtime_error& e){}
        catch (std::invalid_argument& e){}
        catch (std::logic_error& e){}
        
//        printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
//            printf("mse = %f -> fx = %f\n", mse, fx);
            this->params = eigenVec;
            improved = true;
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    int values() const
    {
        return Board::data.numRows();
    }
    
    int df(Eigen::VectorXf &x, Eigen::MatrixXf &fjac)
    {
        float epsilon, temp;
        epsilon = 1e-5f;

        for (int i = 0; i < x.size(); i++)
        {
//            Eigen::VectorXf xPlus(x);
//            xPlus(i) += epsilon;
//
//            Eigen::VectorXf xMinus(x);
//            xMinus(i) -= epsilon;
//            x(i) -= epsilon;
            
            temp = x(i);
            
            x(i) = temp + epsilon;
            Eigen::VectorXf fvecPlus(values());
            operator()(x, fvecPlus);

            x(i) = temp - epsilon;
            Eigen::VectorXf fvecMinus(values());
            operator()(x, fvecMinus);

            fjac.block(0, i, values(), 1) = std::move((fvecPlus - fvecMinus) / (2.0f * epsilon));
            
            x(i) = temp;
        }
        return 0;
    }
    
    bool LevenbergMarquardt()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::LevenbergMarquardt<decltype(*this), float> lm(*this);
        float score_before = MSE(expression_evaluator(this->params, this->pieces));
        lm.parameters.maxfev = this->num_fit_iter;
//        std::cout << "ftol (Cost function change) = " << lm.parameters.ftol << '\n';
//        std::cout << "xtol (Parameters change) = " << lm.parameters.xtol << '\n';

        lm.minimize(this->params);
        if (MSE(expression_evaluator(this->params, this->pieces)) < score_before)
        {
            improved = true;
        }
        
//        std::cout << "Iterations = " << lm.nfev << '\n';
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    float fitFunctionToData()
    {
        float loss = 0.0f;
        if (this->params.size())
        {
            bool improved = true;
            if (this->fit_method == "PSO")
            {
                improved = PSO();
            }
            else if (this->fit_method == "AsyncPSO")
            {
                improved = AsyncPSO();
            }
            else if (this->fit_method == "LBFGS")
            {
                improved = LBFGS();
            }
            else if (this->fit_method == "LBFGSB")
            {
                improved = LBFGSB();
            }
            else if (this->fit_method == "LevenbergMarquardt")
            {
                improved = LevenbergMarquardt();
            }
            Eigen::VectorXf temp_vec;
            
            if (improved) //If improved, update the expression_dict with this->params
            {
                if (Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.visit(this->expression_string, [&](auto& x)
                    {
                        x.second = this->params;
                    });
                }
                else
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, this->params);
                }
            }
            Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
            {
                temp_vec = x.second;
            });
                        
            loss = loss_func(expression_evaluator(temp_vec, this->pieces));
        }
        else
        {
            this->diffeq_result = diffeq(*this);
            loss = loss_func(expression_evaluator(this->params, this->diffeq_result));
        }
        return loss;
    }
    
    /*
    Check whether the given player has created a
    complete (depth self.n) expression (again), and
    checks if it is a complete PN/RPN expression.
    Returns the score of the expression if complete,
    where 0 <= score <= 1 and -1 if not complete or if
    the desired depth has not been reached.
    */
    float complete_status(bool cache = true)
    {
        if (this->pieces.empty())
        {
            this->stack.clear();
            this->idx = 0;
            if (this->expression_type == "prefix")
            {
                this->depth = 0, this->num_binary = 0, this->num_leaves = 0;
            }
        }
        auto [depth, complete] =  ((this->expression_type == "prefix") ? getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/) : getRPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/)); //structured binding :)
        if (!complete || depth < this->n) //Expression not complete
        {
            return -1;
        }
        else
        {
            if (visualize_exploration)
            {
                //whenever. TODO: call some plotting function, e.g. ROOT CERN plotting API, Matplotlib from the Python-C API, Plotly if we want a web application for this, etc. The plotting function could also have the fitted constants (rounded of course), but then this if statement would need to be moved down to below the fitFunctionToData call in this `complete_status` method.
            }
            
            if (is_primary)
            {
                this->expression_string.clear();
                this->expression_string.reserve(8*pieces.size());
                
                for (float i: pieces){this->expression_string += std::to_string(i)+" ";}

                if (!Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, Eigen::VectorXf());
                }
                
                Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
                {
                    this->params = x.second;
                });
                
                if (!this->params.size())
                {
                    this->params.setOnes(this->__num_consts());
                    Board::expression_dict.insert_or_assign(this->expression_string, this->params);
                }

                return fitFunctionToData();
            }
            return 0.0f;
        }
    }
    const Eigen::VectorXf& operator[] (int i)
    {
        return Board::data[i];
    }
    const Eigen::VectorXf& operator[] (const std::string& i)
    {
        return Board::data[i];
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Board& b)
    {
        return (os << b.data);
    }
    
    //Function to compute the LGB or RGB, from https://www.jstor.org/stable/43998756
    //(top of pg. 165)
    void GB(size_t z, size_t& ind, const std::vector<float>& individual)
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
    int GR(size_t i, const std::vector<float>& individual)
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
    
    //Adds pairs containing the starting and stopping indices for each
    //depth-n sub-expression in the expression individual
    void get_indices(std::vector<std::pair<int, int>>& sub_exprs, std::vector<float>& individual)
    {
        size_t temp;
        for (size_t k = 0; k < individual.size(); k++)
        {
            temp = k; //we don't want to change k
            size_t& ptr_GB = temp;
            
            if (is_unary(individual[k]))
            {
                GB(1, ptr_GB, individual);
//                std::cout << k << ' ' << ptr_GB << ' ' << Board::__tokens_dict[individual[k]]
//                << ' ' << Board::__tokens_dict[individual[ptr_GB]] << '\n';
            }
            else if (is_binary(individual[k]))
            {
                GB(2, ptr_GB, individual);
//                std::cout << k << ' ' << ptr_GB << ' ' << Board::__tokens_dict[individual[k]]
//                << ' ' << Board::__tokens_dict[individual[ptr_GB]] << '\n';
            }
            else if (this->n == 0) //depth-0 sub-trees are leaf-nodes
            {
                sub_exprs.push_back(std::make_pair(k, k));
                continue;
            }
            
            auto [start, stop] = std::make_pair( std::min(k, ptr_GB), std::max(k, ptr_GB));
//            std::cout << "start, stop = " << start << " , " << stop << '\n';
            auto [depth, complete] =  ((expression_type == "prefix") ? getPNdepth(individual, start, stop+1, false /*cache*/) : getRPNdepth(individual, start, stop+1));
            
            if (complete && (depth == this->n))
            {
                sub_exprs.push_back(std::make_pair(start, stop));
            }
        }
    }
    
    void setPrefixGR(const std::vector<float>& prefix, std::vector<int>& grasp)
    {
        grasp.reserve(prefix.size());
        for (size_t k = 0; k < prefix.size(); ++k)
        {
            grasp.push_back(GR(k, prefix));
        }
    }

    /*
    low and up: lower and upper Index bounds, respectively, for the piece of the array prefix which is to be the subject of the processing.
    dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
    */
    void derivePrefixHelper(int low, int up, const std::string& dx, const std::vector<float>& prefix, std::vector<int>& grasp, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            this->derivat.clear();
            // std::cout << this->derivat.size();
            this->derivat.reserve(100);
    //        Index = 0;
            setPrefixGR(prefix, grasp);
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
                //of prefix[up-grasp(op2)-2] are the elements [(Board::__tokens_dict[prefix[low]] = prefix[0], prefix[up-grasp(op2)-2] = prefix[9-5-2] = prefix[2]]
                //i.e., the elements {"x", "x", "*"}

        if (Board::__tokens_dict[prefix[low]] == "+" || Board::__tokens_dict[prefix[low]] == "-")
        {
            this->derivat.push_back(prefix[low]); //+/-
            size_t temp = low+1+grasp[low+1];
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true);  /* +/- x' */
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* +/- x' y' */
        }
        else if (Board::__tokens_dict[prefix[low]] == "*")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["+"]); /* +  */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* + * */
            size_t temp = low+1+grasp[low+1];
            for (int k = low+1; k <= temp; k++) /* + * x */
            {
                this->derivat.push_back(prefix[k]);
            }
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* + * x y' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* + * x y' * */
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* + * x y' * x' */
            for (int k = temp+1; k <= temp+1+grasp[temp+1]; k++)
            {
                this->derivat.push_back(prefix[k]); /* + * x y' * x' y */
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "/")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["/"]); /* / */
            this->derivat.push_back(Board::__tokens_inv_dict["-"]); /* / - */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* / - * */
            size_t temp = low+1+grasp[low+1];
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* / - * x' */
            int k;
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y */
            {
                this->derivat.push_back(prefix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* / - * x' y * */
            for (k = low+1; k <= temp; k++) /* / - * x' y * x */
            {
                this->derivat.push_back(prefix[k]);
            }
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* / - * x' y * x y' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* / - * x' y * x y' * */
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y */
            {
                this->derivat.push_back(prefix[k]);
            }
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y y */
            {
                this->derivat.push_back(prefix[k]);
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "^")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* * */
            this->derivat.push_back(Board::__tokens_inv_dict["^"]); /* * ^ */
            size_t temp = low+1+grasp[low+1];
            int k;
            for (k = low+1; k <= temp; k++) /* * ^ x */
            {
                this->derivat.push_back(prefix[k]);
            }
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ^ x y */
            {
                this->derivat.push_back(prefix[k]);
            }
            std::vector<float> prefix_temp;
            std::vector<int> grasp_temp;
            size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> * ln x y
            prefix_temp.reserve(reserve_amount);
            grasp_temp.reserve(reserve_amount);
            prefix_temp.push_back(Board::__tokens_inv_dict["*"]); /* * */
            prefix_temp.push_back(Board::__tokens_inv_dict["ln"]); /* * ln */
            for (k = low+1; k <= temp; k++) /* * ln x */
            {
                prefix_temp.push_back(prefix[k]);
            }
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ln x y */
            {
                prefix_temp.push_back(prefix[k]);
            }
            setPrefixGR(prefix_temp, grasp_temp);
            derivePrefixHelper(0, prefix_temp.size() - 1, dx, prefix_temp, grasp_temp, true); /* * ^ x y (* ln x y)' */
        }

        else if (Board::__tokens_dict[prefix[low]] == "cos")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* * */
            this->derivat.push_back(Board::__tokens_inv_dict["~"]); /* * ~ */
            this->derivat.push_back(Board::__tokens_inv_dict["sin"]); /* * ~ sin */
            size_t temp = low+1;
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                this->derivat.push_back(prefix[k]); /* * ~ sin x */
            }
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * ~ sin x x' */
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "sin")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* * */
            this->derivat.push_back(Board::__tokens_inv_dict["cos"]); /* * cos */
            size_t temp = low+1;
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                this->derivat.push_back(prefix[k]); /* * cos x */
            }
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * cos x x' */
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "sqrt")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);         /* / */
            size_t temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);         /* / x' * */
            this->derivat.push_back(Board::__tokens_inv_dict["2"]);         /* / x' * 2 */
            this->derivat.push_back(Board::__tokens_inv_dict["sqrt"]);      /* / x' * 2 sqrt */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' * 2 sqrt x */
            {
                this->derivat.push_back(prefix[k]);
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "log" || Board::__tokens_dict[prefix[low]] == "ln")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);               /* / */
            size_t temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                this->derivat.push_back(prefix[k]);      /* / x' x */
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "asin" || Board::__tokens_dict[prefix[low]] == "arcsin")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);   /* / */
            size_t temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            this->derivat.push_back(Board::__tokens_inv_dict["sqrt"]); /* / x' sqrt */
            this->derivat.push_back(Board::__tokens_inv_dict["-"]);    /* / x' sqrt - */
            this->derivat.push_back(Board::__tokens_inv_dict["1"]);    /* / x' sqrt - 1 */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);    /* / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x */
            {
                this->derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x x */
            {
                this->derivat.push_back(prefix[k]);
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "acos" || Board::__tokens_dict[prefix[low]] == "arccos")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["~"]);   /* ~ */
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);   /* ~ / */
            size_t temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            this->derivat.push_back(Board::__tokens_inv_dict["sqrt"]); /* / x' sqrt */
            this->derivat.push_back(Board::__tokens_inv_dict["-"]);    /* / x' sqrt - */
            this->derivat.push_back(Board::__tokens_inv_dict["1"]);    /* / x' sqrt - 1 */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);    /* / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x */
            {
                this->derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x x */
            {
                this->derivat.push_back(prefix[k]);
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "tanh")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);      /* * */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);      /* * * */
            this->derivat.push_back(Board::__tokens_inv_dict["sech"]);   /* * * sech */
            size_t temp = low+1;
            for (int k = temp; k <= temp+grasp[temp]; k++) /* * * sech x */
            {
                this->derivat.push_back(prefix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["sech"]);   /* * * sech x sech */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* * * sech x sech x */
            {
                this->derivat.push_back(prefix[k]);
            }
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * * sech x sech x x' */
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "sech")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["~"]);      /* ~ */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);      /* ~ * */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);      /* ~ * * */
            this->derivat.push_back(Board::__tokens_inv_dict["sech"]);   /* ~ * * sech */
            size_t temp = low+1;
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ * * sech x */
            {
                this->derivat.push_back(prefix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["tanh"]);   /* ~ * * sech x tanh */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ * * sech x tanh x */
            {
                this->derivat.push_back(prefix[k]);
            }
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ * * sech x tanh x x' */
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "exp")
        {
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);               /* * */
            this->derivat.push_back(Board::__tokens_inv_dict["exp"]);             /* * exp */
            size_t temp = low+1;
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                this->derivat.push_back(prefix[k]);      /* * exp x */
            }
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * exp x x' */
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "~")
        {
            size_t temp = low+1;
            this->derivat.push_back(prefix[low]); /* ~ */
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ x' */
        }
        
        else
        {
            if (Board::__tokens_dict[prefix[low]] == dx)
            {
                this->derivat.push_back(Board::__tokens_inv_dict["1"]);
            }
            else
            {
                this->derivat.push_back(Board::__tokens_inv_dict["0"]);
            }
        }
    }

    void derivePrefix(int low, int up, const std::string& dx, const std::vector<float>& prefix, std::vector<int>& grasp)
    {
        derivePrefixHelper(low, up, dx, prefix, grasp, false);
    }
    
    void setPostfixGR(const std::vector<float>& postfix, std::vector<int>& grasp)
    {
        grasp.reserve(postfix.size()); //grasp[k] = GR( postfix[k]), k = 1, ... ,i.
        //In the paper they do `k = 1;` instead of `k = 0;`, presumably because GR(postfix[0]) always is 0, but it works
        //if you set k = 0 too.
        for (size_t k = 0; k < postfix.size(); ++k)
        {
            grasp.push_back(GR(k, postfix));
        }
    }
    
    /*
    low and up: lower and upper Index bounds, respectively, for the piece of the array postfix which is to be the subject of the processing.
    dx: string representing the variable by which the derivation is to be made. (The derivative is made wrt dx)
    */
    void derivePostfixHelper(int low, int up, const std::string& dx, const std::vector<float>& postfix, std::vector<int>& grasp, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            this->derivat.clear();
            // std::cout << this->derivat.size();
            this->derivat.reserve(100);
    //        Index = 0;
            setPostfixGR(postfix, grasp);
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
        
        //if Board::__tokens_dict[postfix[up]] is a binary operator, then:
            //the head of its second argument (let's call it op2) is equal to postfix[up-1]
            //then the grasped elements of op2 are the elements from postfix[up-1-grasp[up-1]] to postfix[up-1]
                //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "*", "+"}, up = 7 -> Board::__tokens_dict[postfix[up]] = "+" is binary
                //so postfix[up-1] = "*" is the head of the second argument of "+" and so the grasped elements
                //of postfix[up-1] are the elements [(postfix[up-1-grasp[up-1]] = postfix[6-3] = postfix[3]), postfix[up-1] = postfix[6]]
                //i.e., the elements {"x", "cos", "x", "*"}
            //the head of its first argument (lets call it op1) is equal to postfix[up-grasp(op2)-2] which is equal to postfix[up-2-grasp[up-1]].
            //then the grasped elements of op1 are the elements from postfix[low = 0] to postfix[up-2-grasp[up-1]]
                //e.g. postfix = {"x", "x", "*", "x", "cos", "x", "x", "*", "*", "+"}, up = 9 ->Board::__tokens_dict[postfix[up]] = "+" is binary
                //so postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2] = "*" is the head of the first argument of "+" and so the grasped elements
                //of postfix[up-grasp(op2)-2] are the elements [(postfix[low] = postfix[0], postfix[up-grasp(op2)-2] = postfix[9-5-2] = postfix[2]]
                //i.e., the elements {"x", "x", "*"}

        if (Board::__tokens_dict[postfix[up]] == "+" || Board::__tokens_dict[postfix[up]] == "-")
        {
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true);  /*Putting x'*/
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /*Putting y'*/
            this->derivat.push_back(postfix[up]);
        }
        else if (Board::__tokens_dict[postfix[up]] == "*")
        {
            for (int k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                this->derivat.push_back(postfix[k]);
            }
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x y' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x y' "*" */
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x y' "*" x' */
            for (int k = up-1-grasp[up-1]; k <= up - 1; k++)
            {
                this->derivat.push_back(postfix[k]); /* x y' "*" x' y */
            }
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x y' "*" x' y "*" */
            this->derivat.push_back(Board::__tokens_inv_dict["+"]); /* x y' "*" x' y "*" + */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "/")
        {
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x' */
            int k;
            for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x' y */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' y *  */
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x' y * x */
            {
                this->derivat.push_back(postfix[k]);
            }
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x' y * x y' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' y * x y' * */
            this->derivat.push_back(Board::__tokens_inv_dict["-"]); /* x' y * x y' * - */
            for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y */
            {
                this->derivat.push_back(postfix[k]);
            }
            for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y y */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' y * x y' * - y y * */
            this->derivat.push_back(Board::__tokens_inv_dict["/"]); /* x' y * x y' * - y y * / */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "^")
        {
            int k;
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                this->derivat.push_back(postfix[k]);
            }
            for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x y */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["^"]); /* x y ^ */

            std::vector<float> postfix_temp;
            std::vector<int> grasp_temp;
            size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> x ln y *
            postfix_temp.reserve(reserve_amount);
            grasp_temp.reserve(reserve_amount);
            
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                postfix_temp.push_back(postfix[k]);
            }
            postfix_temp.push_back(Board::__tokens_inv_dict["ln"]); /* x ln  */
            for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x ln y */
            {
                postfix_temp.push_back(postfix[k]);
            }
            postfix_temp.push_back(Board::__tokens_inv_dict["*"]); /* x ln y * */
            setPostfixGR(postfix_temp, grasp_temp);
            
            derivePostfixHelper(0, postfix_temp.size() - 1, dx, postfix_temp, grasp_temp, true); /* x y ^ (x ln y *)' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x y ^ (x ln y *)' * */
        }

        else if (Board::__tokens_dict[postfix[up]] == "cos")
        {
            for (int k = low; k <= up-1; k++)
            {
                this->derivat.push_back(postfix[k]); /* x */
            }
            this->derivat.push_back(Board::__tokens_inv_dict["sin"]); /* x sin */
            this->derivat.push_back(Board::__tokens_inv_dict["~"]); /* x sin ~ */
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x sin ~ x' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x sin ~ x' * */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "sin")
        {
            for (int k = low; k <= up-1; k++)
            {
                this->derivat.push_back(postfix[k]); /* x */
            }
            this->derivat.push_back(Board::__tokens_inv_dict["cos"]); /* x cos */
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x cos x' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]); /* x cos x' * */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "sqrt")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            this->derivat.push_back(Board::__tokens_inv_dict["2"]); /* x' 2 */
            for (int k = low; k <= up-1; k++) /* x' 2 x */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["sqrt"]);    /* x' 2 x sqrt */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);       /* x' 2 x sqrt * */
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);       /* x' 2 x sqrt * / */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "log" || Board::__tokens_dict[postfix[up]] == "ln")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            for (int k = low; k <= up-1; k++)
            {
                this->derivat.push_back(postfix[k]);      /* x' x */
            }
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);               /* x' x / */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "asin" || Board::__tokens_dict[postfix[up]] == "arcsin")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            this->derivat.push_back(Board::__tokens_inv_dict["1"]); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                this->derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);   /* x' 1 x x * */
            this->derivat.push_back(Board::__tokens_inv_dict["-"]);   /* x' 1 x x * - */
            this->derivat.push_back(Board::__tokens_inv_dict["sqrt"]);   /* x' 1 x x * - sqrt */
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);   /* x' 1 x x * - sqrt / */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "acos" || Board::__tokens_dict[postfix[up]] == "arccos")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            this->derivat.push_back(Board::__tokens_inv_dict["1"]); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                this->derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);   /* x' 1 x x * */
            this->derivat.push_back(Board::__tokens_inv_dict["-"]);   /* x' 1 x x * - */
            this->derivat.push_back(Board::__tokens_inv_dict["sqrt"]);   /* x' 1 x x * - sqrt */
            this->derivat.push_back(Board::__tokens_inv_dict["/"]);   /* x' 1 x x * - sqrt / */
            this->derivat.push_back(Board::__tokens_inv_dict["~"]);   /* x' 1 x x * - sqrt / ~ */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "tanh")
        {
            for (int k = low; k <= up-1; k++) /* x */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["sech"]);   /* x sech */
            for (int k = low; k <= up-1; k++) /* x sech x */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["sech"]);   /* x sech x sech */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);      /* x sech x sech * */
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x sech x sech * x' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);   /* x sech x sech * x' * */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "sech")
        {
            for (int k = low; k <= up-1; k++) /* x */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["sech"]);   /* x sech */
            this->derivat.push_back(Board::__tokens_inv_dict["~"]);      /* x sech ~ */
            for (int k = low; k <= up-1; k++) /* x sech ~ x */
            {
                this->derivat.push_back(postfix[k]);
            }
            this->derivat.push_back(Board::__tokens_inv_dict["tanh"]);   /* x sech ~ x tanh */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);      /* x sech ~ x tanh * */
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x sech ~ x tanh * x' */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);   /* x sech ~ x tanh * x' * */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "exp")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            for (int k = low; k <= up-1; k++)
            {
                this->derivat.push_back(postfix[k]);      /* x' x */
            }
            this->derivat.push_back(Board::__tokens_inv_dict["exp"]);               /* x' x exp */
            this->derivat.push_back(Board::__tokens_inv_dict["*"]);               /* x' x exp * */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "~")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            this->derivat.push_back(postfix[up]); /* x' ~ */
        }
        
        else
        {
            if (Board::__tokens_dict[postfix[up]] == dx)
            {
                this->derivat.push_back(Board::__tokens_inv_dict["1"]);
            }
            else
            {
                this->derivat.push_back(Board::__tokens_inv_dict["0"]);
            }
        }
    }

    void derivePostfix(int low, int up, const std::string& dx, const std::vector<float>& postfix, std::vector<int>& grasp)
    {
        derivePostfixHelper(low, up, dx, postfix, grasp, false);
    }
};

std::vector<float> VortexRadialProfile(Board& x)
{
    std::vector<float> result;
    result.reserve(100);
    std::vector<int> grasp;
    std::vector<float> R_prime;
    std::string mu = "1";
    std::string S = "1";
    if (x.expression_type == "prefix")
    {
        //- + + * / 1 2 R'' * / 1 * 2 r R' * - mu / * S S * * 2 r r R * * R R R
        result.push_back(Board::__tokens_inv_dict["-"]);
        result.push_back(Board::__tokens_inv_dict["+"]);
        result.push_back(Board::__tokens_inv_dict["+"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        result.push_back(Board::__tokens_inv_dict["1"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        x.derivePrefix(0, x.pieces.size()-1, "x0", x.pieces, grasp);
        R_prime = x.derivat;
        x.derivePrefix(0, R_prime.size()-1, "x0", R_prime, grasp);
        for (float i: x.derivat) //R''
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        result.push_back(Board::__tokens_inv_dict["1"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        result.push_back(Board::__tokens_inv_dict["x0"]); //r
        for (float i: R_prime) //R'
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["-"]);
        result.push_back(Board::__tokens_inv_dict[mu]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict[S]);
        result.push_back(Board::__tokens_inv_dict[S]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        result.push_back(Board::__tokens_inv_dict["x0"]); //r
        result.push_back(Board::__tokens_inv_dict["x0"]); //r
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
    }
    else if (x.expression_type == "postfix")
    {
        //1 2 / R'' * 1 2 r * / R' * + mu S S * 2 r r * * / - R * + R R * R * -
        result.push_back(Board::__tokens_inv_dict["1"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        x.derivePostfix(0, x.pieces.size()-1, "x0", x.pieces, grasp);
        R_prime = x.derivat;
        x.derivePostfix(0, R_prime.size()-1, "x0", R_prime, grasp);
        for (float i: x.derivat) //R''
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["1"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        result.push_back(Board::__tokens_inv_dict["x0"]); //r
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        for (float i: R_prime) //R'
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["+"]);
        result.push_back(Board::__tokens_inv_dict[mu]);
        result.push_back(Board::__tokens_inv_dict[S]);
        result.push_back(Board::__tokens_inv_dict[S]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        result.push_back(Board::__tokens_inv_dict["x0"]); //r
        result.push_back(Board::__tokens_inv_dict["x0"]); //r
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        result.push_back(Board::__tokens_inv_dict["-"]);
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["+"]);
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        for (float i: x.pieces) //R
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["-"]);

    }
    return result;
}

//https://dl.acm.org/doi/pdf/10.1145/3449639.3459345?casa_token=Np-_TMqxeJEAAAAA:8u-d6UyINV6Ex02kG9LthsQHAXMh2oxx3M4FG8ioP0hGgstIW45X8b709XOuaif5D_DVOm_FwFo
//https://core.ac.uk/download/pdf/6651886.pdf
void SimulatedAnnealing(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0)
{
    std::map<int, std::vector<float>> scores; //map to store the scores
    size_t measure_period = static_cast<size_t>(time/interval);
    
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp-1);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    
    for (int run = 1; run <= num_runs; run++)
    {
        /*
         Outside of thread:
         */
        std::atomic<float> max_score{0.0};
        std::vector<std::pair<int, float>> temp_scores;
        auto start_time = Clock::now();
        std::thread pushBackThread([&]() // Separate thread to push_back the pair every measure_period seconds
        {
            while (timeElapsedSince(start_time) < time)
            {
                std::this_thread::sleep_for(std::chrono::seconds(measure_period));
                temp_scores.push_back(std::make_pair(static_cast<size_t>(timeElapsedSince(start_time)), max_score.load()));
            }
        });
        
        /*
         Inside of thread:
         */
        
        auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point]()
        {
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
            Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
            sync_point.arrive_and_wait();
            Board secondary(diffeq, false, 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache); //For perturbations
            float score = 0.0f, check_point_score = 0.0f;
            
            std::vector<float> current;
            std::vector<std::pair<int, int>> sub_exprs;
            std::vector<float> temp_legal_moves;
            std::uniform_int_distribution<int> rand_depth_dist(0, x.n);
            size_t temp_sz;
    //        std::string expression, orig_expression, best_expression;
            constexpr float T_max = 0.1f;
            constexpr float T_min = 0.012f;
            constexpr float ratio = T_min/T_max;
            float T = T_max;
            
            auto P = [&](float delta)
            {
                return exp(delta/T);
            };
            
            auto updateScore = [&](float r = 1.0f)
            {
    //            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
    //            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
                if ((score > max_score) || (x.pos_dist(generator) < P(score-max_score)))
                {
                    current = x.pieces; //update current expression
                    if (score > max_score)
                    {
    //                    expression = x._to_infix();
    //                    orig_expression = x.expression();
                        max_score = score;
    //                    std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
    //                    std::cout << "Best expression = " << expression << '\n';
    //                    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    //                    best_expression = std::move(expression);
                    }
                }
                else
                {
                    x.pieces = current; //reset perturbed state to current state
                }
                T = r*T;
            };
            
            //Another way to do this might be clustering...
            auto Perturbation = [&](int n, int i)
            {
                //Step 1: Generate a random depth-n sub-expression `secondary_one.pieces`
                secondary.pieces.clear();
                sub_exprs.clear();
                secondary.n = n;
                while (secondary.complete_status() == -1)
                {
                    temp_legal_moves = secondary.get_legal_moves();
                    std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
                    secondary.pieces.push_back(temp_legal_moves[distribution(generator)]);
                }
                
    //            assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces) : secondary.getRPNdepth(secondary.pieces)).first == secondary.n);
    //            assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces) : secondary.getRPNdepth(secondary.pieces)).second);
                
                if (n == x.n)
                {
                    std::swap(secondary.pieces, x.pieces);
                }
                else
                {
                    //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
                    //in `x.pieces` and store them in an std::vector<std::pair<int, int>>
                    //called `sub_exprs`.
                    secondary.get_indices(sub_exprs, x.pieces);
                    
                    //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `pert_ind`

                    std::uniform_int_distribution<int> distribution(0, sub_exprs.size() - 1);
                    int pert_ind = distribution(generator);
                    
                    //Step 4: Substitute sub_exprs_1[pert_ind] in x.pieces with secondary_one.pieces
                    
                    auto start = x.pieces.begin() + sub_exprs[pert_ind].first;
                    auto end = std::min(x.pieces.begin() + sub_exprs[pert_ind].second, x.pieces.end());
                    x.pieces.erase(start, end+1);
                    x.pieces.insert(start, secondary.pieces.begin(), secondary.pieces.end()); //could be a move operation: secondary.pieces doesn't need to be in a defined state after this->params
                }
                
                //Step 5: Evaluate the new mutated `x.pieces` and update score if needed
                score = x.complete_status(false);
                updateScore(pow(ratio, 1.0f/(i+1)));
            };

            //Step 1: generate a random expression
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                x.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                current.push_back(x.pieces.back());
            }
            updateScore();
            
            for (int i = 0; (timeElapsedSince(start_time) < time); i++)
            {
                if (i && (i%50000 == 0))
                {
    //                std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
                    if (check_point_score == max_score)
                    {
                        T = std::min(T*10.0f, T_max);
                    }
                    else
                    {
                        T = std::max(T/10.0f, T_min);
                    }
                    check_point_score = max_score;
                }
                Perturbation(rand_depth_dist(generator), i);
                
            }
        };
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
        }
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i].join();
        }
        
        // Join the separate thread to ensure it has finished before exiting
        pushBackThread.join();
        
        for (auto& i: temp_scores)
        {
            scores[i.first].push_back(i.second);
        }
        
        std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
        std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
        std::cout << "Best score = " << max_score.load() << ", MSE = " << (1/max_score)-1 << '\n';
    }
    std::ofstream out(filename);
    for (auto& i: scores)
    {
        out << i.first << ',';
        for (auto& j: i.second)
        {
            out << j << ((&j == &i.second.back()) ? '\n' : ',');
        }
    }
    out.close();
}

//https://arxiv.org/abs/2310.06609
void GP(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0)
{
    std::map<int, std::vector<float>> scores; //map to store the scores
    size_t measure_period = static_cast<size_t>(time/interval);
    
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp-1);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    
    for (int run = 1; run <= num_runs; run++)
    {
        /*
         Outside of thread:
         */
        std::atomic<float> max_score{0.0};
        std::vector<std::pair<int, float>> temp_scores;
        auto start_time = Clock::now();
        std::thread pushBackThread([&]() // Separate thread to push_back the pair every measure_period seconds
        {
            while (timeElapsedSince(start_time) < time)
            {
                std::this_thread::sleep_for(std::chrono::seconds(measure_period));
                temp_scores.push_back(std::make_pair(static_cast<size_t>(timeElapsedSince(start_time)), max_score.load()));
            }
        });
        
        /*
         Inside of thread:
         */
        
        auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point]()
        {
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
            Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
            sync_point.arrive_and_wait();
            Board secondary_one(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache), secondary_two(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache); //For crossover and mutations
            float score = 0.0f, mut_prob = 0.8f, rand_mut_cross;
            constexpr int init_population = 2000;
            std::vector<std::pair<std::vector<float>, float>> individuals;
            std::pair<std::vector<float>, float> individual_1, individual_2;
            std::vector<std::pair<int, int>> sub_exprs_1, sub_exprs_2;
            individuals.reserve(2*init_population);
            std::vector<float> temp_legal_moves;
            std::uniform_int_distribution<int> rand_depth_dist(0, x.n - 1), selector_dist(0, init_population - 1);
            int rand_depth, rand_individual_idx_1, rand_individual_idx_2;
            std::uniform_real_distribution<float> rand_mut_cross_dist(0.0f, 1.0f);
            size_t temp_sz;
        //    std::string expression, orig_expression, best_expression;
            
            auto updateScore = [&]()
            {
        //        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
        //        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
                if (score > max_score)
                {
        //            expression = x._to_infix();
        //            orig_expression = x.expression();
                    max_score = score;
        //            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        //            std::cout << "Best expression = " << expression << '\n';
        //            std::cout << "Best expression (original format) = " << orig_expression << '\n';
        //            best_expression = std::move(expression);
                }
            };
            
            //Step 1, generate init_population expressions
            for (int i = 0; i < init_population; i++)
            {
                while ((score = x.complete_status()) == -1)
                {
                    temp_legal_moves = x.get_legal_moves(); //the legal moves
                    temp_sz = temp_legal_moves.size(); //the number of legal moves
                    std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                    x.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                }
                
                updateScore();
                individuals.push_back(std::make_pair(x.pieces, score));
                x.pieces.clear();
            }
            
            auto Mutation = [&](int n)
            {
                //Step 1: Generate a random depth-n sub-expression `secondary_one.pieces`
                secondary_one.pieces.clear();
                sub_exprs_1.clear();
                secondary_one.n = n;
                while (secondary_one.complete_status() == -1)
                {
                    temp_legal_moves = secondary_one.get_legal_moves();
                    std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
                    secondary_one.pieces.push_back(temp_legal_moves[distribution(generator)]);
                }
                
    //            assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).first == secondary_one.n);
    //            assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).second);

                
                //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
                //in `x.pieces` and store them in an std::vector<std::pair<int, int>>
                //called `sub_exprs_1`.
                x.pieces = individuals[selector_dist(generator)].first; //A randomly selected individual to be mutated
                secondary_one.get_indices(sub_exprs_1, x.pieces);
                
                //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `mut_ind`
                std::uniform_int_distribution<int> distribution(0, sub_exprs_1.size() - 1);
                int mut_ind = distribution(generator);
                
                //Step 4: Substitute sub_exprs_1[mut_ind] in x.pieces with secondary_one.pieces
                
                auto start = x.pieces.begin() + sub_exprs_1[mut_ind].first;
                auto end = std::min(x.pieces.begin() + sub_exprs_1[mut_ind].second, x.pieces.end()-1);
                x.pieces.erase(start, end+1);
                x.pieces.insert(start, secondary_one.pieces.begin(), secondary_one.pieces.end());
                
                //Step 5: Evaluate the new mutated `x.pieces` and update score if needed
                score = x.complete_status(false);
                updateScore();
                individuals.push_back(std::make_pair(x.pieces, score));
            };
            
            auto Crossover = [&](int n)
            {
                sub_exprs_1.clear();
                sub_exprs_2.clear();
                secondary_one.n = n;
                secondary_two.n = n;
                
                rand_individual_idx_1 = selector_dist(generator);
                individual_1 = individuals[rand_individual_idx_1];
                
                do {
                    rand_individual_idx_2 = selector_dist(generator);
                } while (rand_individual_idx_2 == rand_individual_idx_1);
                individual_2 = individuals[rand_individual_idx_2];
            
                //Step 1: Identify the starting and stopping index pairs of all depth-n sub-expressions
                //in `individual_1.first` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_1`.
                secondary_one.get_indices(sub_exprs_1, individual_1.first);
                
                //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
                //in `individual_2.first` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_2`.
                secondary_two.get_indices(sub_exprs_2, individual_2.first);
                
                //Step 3: Generate a random uniform int from 0 to sub_exprs_1.size() - 1 called `mut_ind_1`
                std::uniform_int_distribution<int> distribution_1(0, sub_exprs_1.size() - 1);
                int mut_ind_1 = distribution_1(generator);
                
                //Step 4: Generate a random uniform int from 0 to sub_exprs_2.size() - 1 called `mut_ind_2`
                std::uniform_int_distribution<int> distribution_2(0, sub_exprs_2.size() - 1);
                int mut_ind_2 = distribution_2(generator);
                
                //Step 5: Swap sub_exprs_1[mut_ind_1] in individual_1.first with sub_exprs_2[mut_ind_2] in individual_2.first
                auto start_1 = individual_1.first.begin() + sub_exprs_1[mut_ind_1].first;
                auto end_1 = std::min(individual_1.first.begin() + sub_exprs_1[mut_ind_1].second, individual_1.first.end());
                
                auto start_2 = individual_2.first.begin() + sub_exprs_2[mut_ind_2].first;
                auto end_2 = std::min(individual_2.first.begin() + sub_exprs_2[mut_ind_2].second, individual_2.first.end());
                
        //        insert the range start_2, end_2+1 into individual_1 and the range start_1, end_1+1 into individual_2.
                
                if ((end_1 - start_1) < (end_2 - start_2))
                {
                    std::swap_ranges(start_1, end_1+1, start_2);
                    //Insert remaining part of sub_individual_2.first into individual_1.first
                    individual_1.first.insert(end_1+1, start_2 + (end_1+1-start_1), end_2+1);
                    //Remove the remaining part of sub_individual_2.first from individual_2.first
                    individual_2.first.erase(start_2 + (end_1+1-start_1), end_2+1);
                }
                else if ((end_2 - start_2) < (end_1 - start_1))
                {
                    std::swap_ranges(start_2, end_2+1, start_1);
                    //Insert remaining part of sub_individual_1.first into individual_2.first
                    individual_2.first.insert(end_2+1, start_1 + (end_2+1-start_2), end_1+1);
                    //Remove the remaining part of sub_individual_1.first from individual_1.first
                    individual_1.first.erase(start_1 + (end_2+1-start_2), end_1+1);
                }
                else
                {
                    std::swap_ranges(start_1, end_1+1, start_2);
                }

                x.pieces = individual_1.first;
                score = x.complete_status(false);
                updateScore();
                
                individuals.push_back(std::make_pair(x.pieces, score));
                
                x.pieces = individual_2.first;
                score = x.complete_status(false);
                updateScore();
                
                individuals.push_back(std::make_pair(x.pieces, score));
            };

            
            for (/*int ngen = 0*/; (timeElapsedSince(start_time) < time); /*ngen++*/)
            {
    //            if (ngen && (ngen%5 == 0))
    //            {
    //                std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
    //            }
                //Produce N additional individuals through crossover and mutation
                for (int n = 0; n < init_population; n++)
                {
                    //Step 1: Generate a random number between 0 and 1 called `rand_mut_cross`
                    rand_mut_cross = rand_mut_cross_dist(generator);
                    
                    //Step 2: Generate a random uniform int from 0 to x.n - 1 called `rand_depth`
                    rand_depth = rand_depth_dist(generator);
                    
                    //Step 4: Call Mutation function if 0 <= rand_mut_cross <= mut_prob, else select Crossover
                    if (rand_mut_cross <= mut_prob)
                    {
                        Mutation(rand_depth);
                    }
                    else
                    {
                        Crossover(rand_depth);
                    }
                }
                std::sort(individuals.begin(), individuals.end(),
                [](std::pair<std::vector<float>, float>& individual_1, std::pair<std::vector<float>, float>& individual_2)
                {
                    return individual_1.second > individual_2.second;
                });
                individuals.resize(init_population);
            }
        };
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
        }
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i].join();
        }
        
        // Join the separate thread to ensure it has finished before exiting
        pushBackThread.join();
        
        for (auto& i: temp_scores)
        {
            scores[i.first].push_back(i.second);
        }
        
        std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
        std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
        std::cout << "Best score = " << max_score.load() << ", MSE = " << (1/max_score)-1 << '\n';
    }
    
    
    std::ofstream out(filename);
    for (auto& i: scores)
    {
        out << i.first << ',';
        for (auto& j: i.second)
        {
            out << j << ((&j == &i.second.back()) ? '\n' : ',');
        }
    }
    out.close();
}

void PSO(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0)
{
    std::map<int, std::vector<float>> scores; //map to store the scores
    size_t measure_period = static_cast<size_t>(time/interval);
    
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp-1);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    
    for (int run = 1; run <= num_runs; run++)
    {
        /*
         Outside of thread:
         */
        
        std::atomic<float> max_score{0.0};
        std::vector<std::pair<int, float>> temp_scores;
        auto start_time = Clock::now();
        std::thread pushBackThread([&]() // Separate thread to push_back the pair every measure_period seconds
        {
            while (timeElapsedSince(start_time) < time)
            {
                std::this_thread::sleep_for(std::chrono::seconds(measure_period));
                temp_scores.push_back(std::make_pair(static_cast<size_t>(timeElapsedSince(start_time)), max_score.load()));
            }
        });
        
        /*
         Inside of thread:
         */
        
        auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point]()
        {
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
            Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
            sync_point.arrive_and_wait();
            float score = 0, check_point_score = 0;
            std::vector<float> temp_legal_moves;
            
            size_t temp_sz;
        //    std::string expression, orig_expression, best_expression;
            
            /*
             For this setup, we don't know a-priori the number of particles, so we generate them and their corresponding velocities as needed
             */
            std::vector<float> particle_positions, best_positions, v, curr_positions;
            particle_positions.reserve(x.reserve_amount); //stores record of all current particle position indices
            best_positions.reserve(x.reserve_amount); //indices corresponding to best pieces
            curr_positions.reserve(x.reserve_amount); //indices corresponding to x.pieces
            v.reserve(x.reserve_amount); //stores record of all current particle velocities
            float rp, rg, new_v, c = 0.0f;
            int c_count = 0;
            std::unordered_map<float, std::unordered_map<int, int>> Nsa;
            std::unordered_map<float, std::unordered_map<int, float>> Psa;
            std::unordered_map<int, float> p_i_vals, p_i;
            
            /*
             In this implementation of PSO:
             
                 The traditional PSO initializes the particle positions to be between 0 and 1. However, in this application,
                 the particle positions are discrete values and any of the legal integer tokens (moves). The
                 velocities are continuous-valued and perturb the postions, which are subsequently constrained by rounding to
                 the nearest whole number then taking the modulo w.r.t. the # of allowed legal moves.
             
             */
            
            for (int iter = 0; (timeElapsedSince(start_time) < time); iter++)
            {
                if (iter && (iter%50000 == 0))
                {
        //            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
        //            std::cout << "check_point_score = " << check_point_score
        //            << ", max_score = " << max_score << ", c = " << c << '\n';
                    if (check_point_score == max_score)
                    {
                        c_count++;
                        std::uniform_real_distribution<float> temp(-c_count, c_count);
        //                std::cout << "c: " << c << " -> ";
                        c = temp(generator);
        //                std::cout << c << '\n';
                    }
                    else
                    {
        //                std::cout << "c: " << c << " -> ";
                        c = 0.0f; //if new best found, reset c and try to exploit the new best
                        c_count = 0;
        //                std::cout << c << '\n';
                    }
                    check_point_score = max_score;
                }
                
                for (int i = 0; (score = x.complete_status()) == -1; i++) //i is the index of the token
                {
                    rp = x.pos_dist(generator), rg = x.pos_dist(generator);
                    temp_legal_moves = x.get_legal_moves(); //the legal moves
                    temp_sz = temp_legal_moves.size(); //the number of legal moves

                    if (i == particle_positions.size()) //Then we need to create a new particle with some initial position and velocity
                    {
                        particle_positions.push_back(x.pos_dist(generator));
                        v.push_back(x.vel_dist(generator));
                    }
                    
                    particle_positions[i] = trueMod(std::round(particle_positions[i]), temp_sz);
                    x.pieces.push_back(temp_legal_moves[particle_positions[i]]); //x.pieces holds the pieces corresponding to the indices
                    curr_positions.push_back(particle_positions[i]);
                    if (i == best_positions.size())
                    {
                        best_positions.push_back(x.pos_dist(generator));
                        best_positions[i] = trueMod(std::round(best_positions[i]), temp_sz);
                    }
                    //https://hal.science/hal-00764996
                    //https://www.researchgate.net/publication/216300408_An_off-the-shelf_PSO
                    new_v = (0.721*v[i] + x.phi_1*rg*(best_positions[i] - particle_positions[i]) + x.phi_2*rp*(p_i[i] - particle_positions[i]) + c);
                    v[i] = copysign(std::min(new_v, FLT_MAX), new_v);
                    particle_positions[i] += v[i];
                    Nsa[curr_positions[i]][i]++;
                }
                
                for (int i = 0; i < curr_positions.size(); i++)
                {
                    Psa[curr_positions[i]][i] = (Psa[curr_positions[i]][i]+score)/Nsa[curr_positions[i]][i];
                    if (Psa[curr_positions[i]][i] > p_i_vals[i])
                    {
                        p_i[i] = curr_positions[i];
                    }
                    p_i_vals[i] = std::max(p_i_vals[i], Psa[curr_positions[i]][i]);
                    
                }
                
                if (score > max_score)
                {
                    for (int idx = 0; idx < curr_positions.size(); idx++)
                    {
                        best_positions[idx] = curr_positions[idx];
                    }
        //            expression = x._to_infix();
        //            orig_expression = x.expression();
                    max_score = score;
        //            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        //            std::cout << "Best expression = " << expression << '\n';
        //            std::cout << "Best expression (original format) = " << orig_expression << '\n';
        //            best_expression = std::move(expression);
                }
                x.pieces.clear();
                curr_positions.clear();
            }
        };
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
        }
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i].join();
        }
        
        // Join the separate thread to ensure it has finished before exiting
        pushBackThread.join();
        
        for (auto& i: temp_scores)
        {
            scores[i.first].push_back(i.second);
        }
        
        std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
        std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
        std::cout << "Best score = " << max_score.load() << ", MSE = " << (1/max_score)-1 << '\n';
    }
    std::ofstream out(filename);
    for (auto& i: scores)
    {
        out << i.first << ',';
        for (auto& j: i.second)
        {
            out << j << ((&j == &i.second.back()) ? '\n' : ',');
        }
    }
    out.close();
}

//https://arxiv.org/abs/2205.13134
void MCTS(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0)
{
    std::map<int, std::vector<float>> scores; //map to store the scores
    size_t measure_period = static_cast<size_t>(time/interval);
    
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp-1);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);
    
    for (int run = 1; run <= num_runs; run++)
    {
        /*
         Outside of thread:
         */
        std::atomic<float> max_score{0.0};
        std::vector<std::pair<size_t, float>> temp_scores;

        auto start_time = Clock::now();
        std::thread pushBackThread([&]()
        {
            while (timeElapsedSince(start_time) < time)
            {
                std::this_thread::sleep_for(std::chrono::seconds(measure_period));
                temp_scores.push_back(std::make_pair(static_cast<size_t>(timeElapsedSince(start_time)), max_score.load()));
            }
        });
        
        /*
         Inside of thread:
         */
        
        auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point]()
        {
            std::random_device rand_dev;
            std::mt19937 thread_local generator(rand_dev());
            Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
            sync_point.arrive_and_wait();
            float score = 0.0f, check_point_score = 0.0f, UCT, best_act, UCT_best;
            
            std::vector<float> temp_legal_moves;
            std::unordered_map<std::string, std::unordered_map<float, float>> Qsa, Nsa;
            std::unordered_map<std::string, float> Ns;
            std::string state;
            
            float c = 1.4f; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
            std::vector<std::pair<std::string, float>> moveTracker;
            moveTracker.reserve(x.reserve_amount);
            temp_legal_moves.reserve(x.reserve_amount);
            state.reserve(2*x.reserve_amount);
            //        double str_convert_time = 0.0;
            auto getString  = [&]()
            {
                if (!x.pieces.empty())
                {
                    state += std::to_string(x.pieces[x.pieces.size()-1]) + " ";
                }
            };
            
            for (int i = 0; (timeElapsedSince(start_time) < time); i++)
            {
                if (i && (i%50000 == 0))
                {
                    //                    std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
                    //                    std::cout << "check_point_score = " << check_point_score
                    //                    << ", max_score = " << max_score << ", c = " << c << '\n';
                    if (check_point_score == max_score)
                    {
                        //                        std::cout << "c: " << c << " -> ";
                        c += 1.4;
                        //                        std::cout << c << '\n';
                    }
                    else
                    {
                        //                        std::cout << "c: " << c << " -> ";
                        c = 1.4; //if new best found, reset c and try to exploit the new best
                        //                        std::cout << c << '\n';
                        check_point_score = max_score;
                    }
                }
                state.clear();
                while ((score = x.complete_status()) == -1)
                {
                    temp_legal_moves = x.get_legal_moves();
//                    auto start_time = Clock::now();
                    getString();
//                    str_convert_time += timeElapsedSince(start_time);
                    UCT = 0.0f;
                    UCT_best = -FLT_MAX;
                    best_act = -1.0f;
                    std::vector<float> best_acts;
                    best_acts.reserve(temp_legal_moves.size());
                    
                    for (float a : temp_legal_moves)
                    {
                        if (Nsa[state].count(a))
                        {
                            UCT = Qsa[state][a] + c*sqrt(log(Ns[state])/Nsa[state][a]);
                        }
                        else
                        {
                            //not explored -> explore it.
                            best_acts.push_back(a);
                            UCT = -FLT_MAX;
                        }
                        
                        if (UCT > UCT_best)
                        {
                            best_act = a;
                            UCT_best = UCT;
                        }
                    }
                    
                    if (best_acts.size())
                    {
                        std::uniform_int_distribution<int> distribution(0, best_acts.size() - 1);
                        best_act = best_acts[distribution(generator)];
                    }
                    x.pieces.push_back(best_act);
                    moveTracker.push_back(make_pair(state, best_act));
                    Ns[state]++;
                    Nsa[state][best_act]++;
                }
                //backprop reward `score`
                for (auto& state_action: moveTracker)
                {
                    Qsa[state_action.first][state_action.second] = std::max(Qsa[state_action.first][state_action.second], score);
                }
                
                if (score > max_score)
                {
                    max_score = score;
                }
                x.pieces.clear();
                moveTracker.clear();
            }
        };
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
        }
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i].join();
        }
        
        // Join the separate thread to ensure it has finished before exiting
        pushBackThread.join();
        
        for (auto& i: temp_scores)
        {
            scores[i.first].push_back(i.second);
        }
        
        std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
        std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
        std::cout << "Best score = " << max_score.load() << ", MSE = " << (1/max_score)-1 << '\n';
    }
    std::ofstream out(filename);
    for (auto& i: scores)
    {
        out << i.first << ',';
        for (auto& j: i.second)
        {
            out << j << ((&j == &i.second.back()) ? '\n' : ',');
        }
    }
    out.close();
}

void RandomSearch(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXf& data, const int depth = 3, const std::string expression_type = "prefix", const std::string method = "LevenbergMarquardt", const int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", const bool cache = true, const double time = 120.0 /*time to run the algorithm in seconds*/, unsigned int num_threads = 0)
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp);
    }
    
    std::vector<std::thread> threads(num_threads);
    std::latch sync_point(num_threads);

    /*
     Outside of thread:
     */
    
    std::atomic<float> max_score{0.0};
    std::string best_expression, orig_expression, best_expr_result, orig_expr_result;
    
    auto start_time = Clock::now();
    
    /*
     Inside of thread:
     */
    
    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result]()
    {
        std::random_device rand_dev;
        std::mt19937 thread_local generator(rand_dev()); // Mersenne Twister random number generator

        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
        sync_point.arrive_and_wait();
        float score = 0.0f;
        std::vector<float> temp_legal_moves;
        size_t temp_sz;
        while (timeElapsedSince(start_time) < time)
        {
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
//                    assert(temp_sz);
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                {
                    x.pieces.emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                }
            }
            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);

            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                
                best_expression = x._to_infix();
                orig_expression = x.expression();
                
                best_expr_result = x._to_infix(x.diffeq_result);
                orig_expr_result = x.expression(x.diffeq_result);
            }
            x.pieces.clear();
        }
    };
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i] = std::thread(func); //TODO: (maybe) provide a depth argument to func to specify if different threads should focus on different depth expressions (and modify the search functions accordingly)?
    }
    
    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads[i].join();
    }
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    std::cout << "Best diff result = " << best_expr_result << '\n';
    std::cout << "Best expression (original format) = " << orig_expr_result << '\n';
}

int main()
{
//    HembergBenchmarks(20 /*numIntervals*/, 120 /*time*/, 50 /*numRuns*/);
//    AIFeynmanBenchmarks(20 /*numIntervals*/, 120 /*time*/, 50 /*numRuns*/);
    
    /*
        Then, move the generated txt files to the directories Hemberg_Benchmarks and
        AIFeynman_Benchmarks and then run PlotData.py
    */
    
//    MCTS(generateData(100000, 7, 1.0f, 5.0f), 8 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, 4 /*time to run the algorithm in seconds*/, 2 /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, 1 /*number of runs*/, 0 /*num threads*/);
    auto data = createLinspaceMatrix(20, 1, {0.1f}, {15.0f});
    
    std::cout << data << '\n';

    RandomSearch(VortexRadialProfile, data, 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, 4 /*time to run the algorithm in seconds*/, 0 /*num threads*/);

 

    return 0;
}
//git push --set-upstream origin PrefixPostfixSymbolicDifferentiator

//g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSR PrefixPostfixMultiThreadDiffSR.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -ffast-math -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSR PrefixPostfixMultiThreadDiffSR.cpp -g -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native
