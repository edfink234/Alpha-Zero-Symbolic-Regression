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
#include "MLP_Vec.h"

using Clock = std::chrono::high_resolution_clock;

//Returns the number of seconds since `start_time`
template <typename T>
double timeElapsedSince(T start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
}

Eigen::MatrixXf generateData(int numRows, int numCols, float (*func)(const Eigen::VectorXf&), float min = -3.0f, float max = 3.0f)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 thread_local gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);

    // Create the matrix
    Eigen::MatrixXf matrix(numRows, numCols);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols - 1; j++)
        {
            matrix(i, j) = distribution(gen);
        }

        Eigen::VectorXf rowVector(numCols - 1);
        for (int j = 0; j < numCols - 1; j++)
        {
            rowVector(j) = matrix(i, j);
        }
        matrix(i, numCols - 1) = func(rowVector);
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
    long num_columns, num_rows;
    
public:
    
    Data() = default; //so we can have a static Data attribute
    std::vector<Eigen::VectorXf> labels;
    std::vector<Eigen::VectorXf> rows;
    // Assignment operator
    Data& operator=(const Eigen::MatrixXf& theData)
    {
        this->data = theData;
        this->num_columns = data.cols();
        this->num_rows = data.rows();

        for (size_t i = 0; i < this->num_columns - 1; i++) //for each column
        {
            this->features["x"+std::to_string(i)] = Eigen::VectorXf(this->num_rows);
            for (size_t j = 0; j < this->num_rows; j++)
            {
                this->features["x"+std::to_string(i)](j) = this->data(j,i);
            }
        }
        
        this->features["y"] = Eigen::VectorXf(this->num_rows);
        this->rows.resize(this->num_rows);
        this->labels.resize(this->num_rows);

        Eigen::VectorXf y_i(1);
        for (size_t i = 0; i < num_rows; i++)
        {
            this->features["y"](i) = this->data(i, this->num_columns - 1);
            this->rows[i] = data.row(i).head(data.row(i).size() - 1);
            y_i << this->features["y"](i);
            this->labels[i] = y_i;
            assert(this->labels[i].size() == 1 && this->rows[i].size() == 2);
        }
//        this->labels.push_back(this->features["y"]);
        
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

float MSE(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
    if (actual.size() != predicted.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    return (actual - predicted).squaredNorm() / actual.size();
}

Eigen::AutoDiffScalar<Eigen::VectorXf> MSE(const Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>& predicted, const Eigen::VectorXf& actual)
{
    if (actual.size() != predicted.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }
        
    return (actual - predicted).squaredNorm() / actual.size();
}

float loss_func(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
    return (1.0f/(1.0f+MSE(actual, predicted)));
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
    static std::mutex inline thread_locker;
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
    bool visualize_exploration, is_primary;
    MultiLayerPerceptron srnn;
    const unsigned long epochs;
    
    Board(bool primary = true, int n = 3, const std::string& expression_type = "prefix", std::string fitMethod = "PSO", int numFitIter = 1, std::string fitGradMethod = "naive_numerical", const Eigen::MatrixXf& theData = {}, bool visualize_exploration = false, bool cache = false, std::vector<int> layers = {}, const unsigned long num_epochs = 1000, float eta = 0.5f, float theta = 0.5f) : gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, num_fit_iter{numFitIter}, fit_method{fitMethod}, fit_grad_method{fitGradMethod}, is_primary{primary}, srnn{layers, 1.0f, eta, theta, "none", "SR", expression_type}, epochs{num_epochs}
    {
        if (n > 30)
        {
            throw(std::runtime_error("Complexity cannot be larger than 30, sorry!"));
        }
        
        this->n = n;
        this->expression_type = expression_type;
        srnn.pieces = {};
        this->visualize_exploration = visualize_exploration;
        this->reserve_amount = 2*std::pow(2,this->n)-1;
        srnn.pieces.reserve(this->reserve_amount);
        this->cache = cache;
        
        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                Board::data = theData;
                
                Board::__num_features = data[0].size() - 1;
                Board::__input_vars.clear();
                Board::expression_dict.clear();
//                Board::__input_vars.reserve(Board::__num_features);
//                for (auto i = 0; i < Board::__num_features; i++)
//                {
//                    Board::__input_vars.push_back("x"+std::to_string(i));
//                }
                Board::__input_vars = {"w_k", "eta", "theta", "gamma", "epsilon", "beta_1", "beta_2", "d_ij", "value", "theta", "d_ij_nest", "velocity_k", "gradient_k", "g_t_k", "expt_grad_squared_k", "delta_w_t_k", "expt_weight_squared_k", "delta_w_t_k_ada_delta", "m_t_k", "v_t_k", "m_t_k_hat", "v_t_k_hat", "prev_w_k", "t"};
                Board::__unary_operators = {"cos", "exp", "sqrt", "sin", "asin", "ln", "tanh", "acos", "~"};
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
                Board::__other_tokens = {/*"const"*/};
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
                MultiLayerPerceptron::__operators_float.clear();
                for (int i = 1; i <= num_operators; i++)
                {
                    Board::__operators_float.push_back(i);
                    MultiLayerPerceptron::__operators_float.push_back(i);
                }
                int num_unary_operators = Board::__unary_operators.size();
                Board::__unary_operators_float.clear();
                MultiLayerPerceptron::__unary_operators_float.clear();
                for (int i = 1; i <= num_unary_operators; i++)
                {
                    Board::__unary_operators_float.push_back(i);
                    MultiLayerPerceptron::__unary_operators_float.push_back(i);
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
                for (int i = ops_plus_features + 1; i <= ops_plus_features + Board::__other_tokens.size(); i++)
                {
                    Board::__other_tokens_float.push_back(i);
                }
                for (int i = 0; i < Board::action_size; i++)
                {
                    Board::__tokens_dict[Board::__tokens_float[i]] = Board::__tokens[i];
                    MultiLayerPerceptron::__tokens_dict[Board::__tokens_float[i]] = Board::__tokens[i];
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
        for (float token : srnn.pieces)
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
        for (float token : srnn.pieces)
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

        for (float token : srnn.pieces)
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

        for (float token : srnn.pieces)
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
            if (srnn.pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the operators...
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
                return Board::una_bin_leaf_legal_moves_dict[(getPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, true /*unary*/, false /*leaf*/).first <= this->n)][(getPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*binary*/, false /*unary*/, false /*leaf*/).first <= this->n)][(!((num_leaves == num_binary + 1) || (getPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, false /*unary*/, true /*leaf*/).first < this->n && (num_leaves == num_binary))))];
            }
            
            else
            {
                bool una_allowed = false, bin_allowed = false, leaf_allowed = false;
                if (Board::__binary_operators_float.size() > 0)
                {
                    srnn.pieces.push_back(Board::__binary_operators_float[0]);
                    bin_allowed = (getPNdepth(srnn.pieces).first <= this->n);
                }
                if (Board::__unary_operators_float.size() > 0)
                {
                    srnn.pieces[srnn.pieces.size() - 1] = Board::__unary_operators_float[0];
                    una_allowed = (getPNdepth(srnn.pieces).first <= this->n);
                }
                srnn.pieces[srnn.pieces.size() - 1] = Board::__input_vars_float[0];
                leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(srnn.pieces).first < this->n && (num_leaves == num_binary))));
                srnn.pieces.pop_back();
//                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));
                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }

        else //postfix
        {
            if (srnn.pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the features and const
            {
                return Board::una_bin_leaf_legal_moves_dict[false][false][true];
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
                 
            if (this->cache)
            {
                return Board::una_bin_leaf_legal_moves_dict[((num_leaves >= 1) && (getRPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*unary*/, false /*leaf*/).first <= this->n))][(num_binary != num_leaves - 1)][(getRPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*unary*/, true /*leaf*/).first <= this->n)];
            }
            
            else
            {
                bool una_allowed = false, bin_allowed = (num_binary != num_leaves - 1), leaf_allowed = false;
                if (Board::__unary_operators_float.size() > 0)
                {
                    srnn.pieces.push_back(Board::__unary_operators_float[0]);
                    una_allowed = ((num_leaves >= 1) && (getRPNdepth(srnn.pieces).first <= this->n));
                }
                
                srnn.pieces[srnn.pieces.size() - 1] = Board::__input_vars_float[0];
                leaf_allowed = (getRPNdepth(srnn.pieces).first <= this->n);

                srnn.pieces.pop_back();
//                assert(!(!una_allowed && !bin_allowed && !leaf_allowed));

                return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
            }
        }

    }
    
    //Returns the `expression_type` string form of the expression stored in the vector<float> attribute pieces
    std::string expression()
    {
        std::string temp;
        temp.reserve(2*srnn.pieces.size());
        size_t sz = srnn.pieces.size() - 1;
        int const_index = ((expression_type == "postfix") ? 0 : this->params.size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            if (std::find(Board::__other_tokens_float.begin(), Board::__other_tokens_float.end(), srnn.pieces[i]) != Board::__other_tokens_float.end())
            {
                puts("true");
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
                temp += ((i!=sz) ? Board::__tokens_dict[srnn.pieces[i]] + " " : Board::__tokens_dict[srnn.pieces[i]]);
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
        
        for (int i = (is_prefix ? (srnn.pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < srnn.pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[srnn.pieces[i]];

            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), srnn.pieces[i]) == Board::__operators_float.end()) // leaf
            {
                stack.push(((!show_consts) || (std::find(Board::__other_tokens.begin(), Board::__other_tokens.end(), token) == Board::__other_tokens.end())) ? token : std::to_string((this->params)(const_counter++)));
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), srnn.pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
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

    Eigen::VectorXf expression_evaluator(const Eigen::VectorXf& params) const
    {
        std::stack<const Eigen::VectorXf> stack;
        size_t const_count = 0;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (srnn.pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < srnn.pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[srnn.pieces[i]];
            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), srnn.pieces[i]) == Board::__operators_float.end()) // leaf
            {
                if (token == "const")
                {
                    stack.push(Eigen::VectorXf::Ones(Board::data.numRows())*params(const_count++));
                }
                else
                {
                    stack.push(Board::data[token]);
                }
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), srnn.pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
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
    
    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> expression_evaluator(const std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& parameters) const
    {
        std::stack<Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>> stack;
        size_t const_count = 0;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (srnn.pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < srnn.pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[srnn.pieces[i]];

            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), srnn.pieces[i]) == Board::__operators_float.end()) // leaf
            {
                if (token == "const")
                {
//                    std::cout << "\nparameters[" << const_count << "] = " << parameters[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(Board::data.numRows(), parameters[const_count++]));
                    
                }
                else
                {
                    stack.push(Board::data[token]);
                }
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), srnn.pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
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

        float swarm_best_score = loss_func(expression_evaluator(this->params),Board::data["y"]);
        float fpi = loss_func(expression_evaluator(particle_positions),Board::data["y"]);
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
                
                fpi = loss_func(expression_evaluator(particle_positions),Board::data["y"]); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions),Board::data["y"]); //calculate the score with the new position
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

        float swarm_best_score = loss_func(expression_evaluator(this->params),Board::data["y"]);
        float fpi = loss_func(expression_evaluator(particle_positions),Board::data["y"]);
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
                
                fpi = loss_func(expression_evaluator(particle_positions),Board::data["y"]); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions),Board::data["y"]); //calculate the score with the new position
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
        return MSE(expression_evaluator(inputs), Board::data["y"]);
    }
    
    /*
     x: parameter vector: (x_0, x_1, ..., x_{x.size()-1})
     g: gradient evaluated at x: (g_0(x_0), g_1(x_1), ..., g_{g.size()-1}(x_{x.size()-1}))
     */
    float operator()(Eigen::VectorXf& x, Eigen::VectorXf& grad)
    {
        if (this->fit_method == "LBFGS" || this->fit_method == "LBFGSB")
        {
            float mse = MSE(expression_evaluator(x), Board::data["y"]);
            if (this->fit_grad_method == "naive_numerical")
            {
                float low_b, temp;
                for (int i = 0; i < x.size(); i++) //finite differences wrt x evaluated at the current values x(i)
                {
                    //https://stackoverflow.com/a/38855586/18255427
                    temp = x(i);
                    x(i) -= 0.00001f;
                    low_b = MSE(expression_evaluator(x), Board::data["y"]); //f(x-h/2)
                    x(i) = temp + 0.00001f;
                    grad(i) = (MSE(expression_evaluator(x), Board::data["y"]) - low_b) / 0.00002f ; //(f(x+h/2) - f(x-h/2))/h
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
            grad = (this->expression_evaluator(x) - Board::data["y"]);
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
        float mse = MSE(expression_evaluator(this->params), Board::data["y"]);
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
        float mse = MSE(expression_evaluator(this->params), Board::data["y"]);
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
        float score_before = MSE(expression_evaluator(this->params), Board::data["y"]);
        lm.parameters.maxfev = this->num_fit_iter;
//        std::cout << "ftol (Cost function change) = " << lm.parameters.ftol << '\n';
//        std::cout << "xtol (Parameters change) = " << lm.parameters.xtol << '\n';

        lm.minimize(this->params);
        if (MSE(expression_evaluator(this->params), Board::data["y"]) < score_before)
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
                        
            loss = loss_func(expression_evaluator(temp_vec),Board::data["y"]);
        }
        else
        {
            loss = 1.0f/(1.0f+this->srnn.train(data.rows, data.labels, this->epochs, false));
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
        if (srnn.pieces.empty())
        {
            this->stack.clear();
            this->idx = 0;
            if (this->expression_type == "prefix")
            {
                this->depth = 0, this->num_binary = 0, this->num_leaves = 0;
            }
        }
        auto [depth, complete] =  ((this->expression_type == "prefix") ? getPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/) : getRPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/)); //structured binding :)
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
                this->expression_string.reserve(8*srnn.pieces.size());
                
                for (float i: srnn.pieces){this->expression_string += std::to_string(i)+" ";}

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
    
    //Computes the grasp of an arbitrary element srnn.pieces[i],
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
};

// 2.5382*cos(x_3) + x_0^2 - 0.5
// postfix = "const x3 cos * x0 x0 * const - +"
// prefix = "+ * const cos x3 - * x0 x0 const"
float exampleFunc(const Eigen::VectorXf& x)
{
    return 2.5382f*cos(x[3]) + (x[0]*x[0]) - 0.5f;
//    return 5.0f*cos(x[1]+x[3])+x[4];
}

float Hemberg_1(const Eigen::VectorXf& x)
{
    return 8.0f / (2.0f + x[0]*x[0] + x[1]*x[1]);
}

float Hemberg_2(const Eigen::VectorXf& x)
{
    return x[0]*x[0]*x[0]*(x[0]-1.0f) + x[1]*(x[1]/2.0f - 1.0f);
}

float Hemberg_3(const Eigen::VectorXf& x)
{
    return x[0]*x[0]*x[0]/5.0f + x[1]*x[1]*x[1]/2.0f - x[1] - x[0];
}

float Hemberg_4(const Eigen::VectorXf& x)
{
    return (30.0f*x[0]*x[0])/((10.0f-x[0])*x[1]*x[1]) + x[0]*x[0]*x[0]*x[0] - x[0]*x[0]*x[0] + x[1]*x[1]/2.0f - x[1] + (8.0f / (2.0f + x[0]*x[0] + x[1]*x[1])) + x[0];
}

float Hemberg_5(const Eigen::VectorXf& x)
{
    return (30.0f*x[0]*x[0])/((10.0f-x[0])*x[1]*x[1]) + x[0]*x[0]*x[0]*x[0] - (4.0f*x[0]*x[0]*x[0])/5.0f + x[1]*x[1]/2.0f - 2.0f*x[1] + (8.0f / (2.0f + x[0]*x[0] + x[1]*x[1])) + (x[1]*x[1]*x[1])/2.0f - x[0];
}

float Feynman_1(const Eigen::VectorXf& x)
{
    return (x[0]*x[1])/(x[2]*(std::pow(x[3],2)-std::pow(x[4],2)));
}

float Feynman_2(const Eigen::VectorXf& x)
{
    return (x[0]*x[1]*x[2])/(std::pow((x[3]-x[4]),2)+std::pow((x[5]-x[6]),2)+std::pow((x[7]-x[8]),2));
}

float Feynman_3(const Eigen::VectorXf& x)
{
    return std::pow((x[0]*x[1]*x[2]*x[3]*x[4])/(4*x[5]*std::pow(sin(x[6]/2),2)),2);
}

float Feynman_4(const Eigen::VectorXf& x)
{
    return (x[0]*x[1]/(x[2]*x[3]))+((x[0]*x[4])/(x[5]*std::pow(x[6],2)*x[2]*x[3]))*x[7];
}

float Feynman_5(const Eigen::VectorXf& x)
{
    return ((x[0]*x[1])/std::pow(x[2],2)) * (1 + (sqrt(1 + ((2*x[3]*std::pow(x[2],2))/(x[0]*std::pow(x[1],2)))) * cos(x[4]-x[5])));
}

//https://dl.acm.org/doi/pdf/10.1145/3449639.3459345?casa_token=Np-_TMqxeJEAAAAA:8u-d6UyINV6Ex02kG9LthsQHAXMh2oxx3M4FG8ioP0hGgstIW45X8b709XOuaif5D_DVOm_FwFo
//https://core.ac.uk/download/pdf/6651886.pdf
void SimulatedAnnealing(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0, std::vector<int> layers = {}, const unsigned long num_epochs = 1000, float eta = 0.5f, float theta = 0.5f)
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
        std::string best_expression, orig_expression;
        
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
        
        auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &layers, &num_epochs, &best_expression, &orig_expression, &eta, &theta]()
        {
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
            Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, layers, num_epochs, eta, theta);
            sync_point.arrive_and_wait();
            Board secondary(false, 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache); //For perturbations
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
    //            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).first == x.n);
    //            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).second);
                if ((score > max_score) || (x.pos_dist(generator) < P(score-max_score)))
                {
                    current = x.srnn.pieces; //update current expression
                    if (score > max_score)
                    {
    //                    expression = x._to_infix();
    //                    orig_expression = x.expression();
                        max_score = score;
    //                    std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
    //                    std::cout << "Best expression = " << expression << '\n';
    //                    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    //                    best_expression = std::move(expression);
                        std::scoped_lock str_lock(Board::thread_locker);
                        best_expression = x._to_infix();
                        orig_expression = x.expression();
                    }
                }
                else
                {
                    x.srnn.pieces = current; //reset perturbed state to current state
                }
                T = r*T;
            };
            
            //Another way to do this might be clustering...
            auto Perturbation = [&](int n, int i)
            {
                //Step 1: Generate a random depth-n sub-expression `secondary_one.srnn.pieces`
                secondary.srnn.pieces.clear();
                sub_exprs.clear();
                secondary.n = n;
                while (secondary.complete_status() == -1)
                {
                    temp_legal_moves = secondary.get_legal_moves();
                    std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
                    secondary.srnn.pieces.push_back(temp_legal_moves[distribution(generator)]);
                }
                
    //            assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.srnn.pieces) : secondary.getRPNdepth(secondary.srnn.pieces)).first == secondary.n);
    //            assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.srnn.pieces) : secondary.getRPNdepth(secondary.srnn.pieces)).second);
                
                if (n == x.n)
                {
                    std::swap(secondary.srnn.pieces, x.srnn.pieces);
                }
                else
                {
                    //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
                    //in `x.srnn.pieces` and store them in an std::vector<std::pair<int, int>>
                    //called `sub_exprs`.
                    secondary.get_indices(sub_exprs, x.srnn.pieces);
                    
                    //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `pert_ind`

                    std::uniform_int_distribution<int> distribution(0, sub_exprs.size() - 1);
                    int pert_ind = distribution(generator);
                    
                    //Step 4: Substitute sub_exprs_1[pert_ind] in x.srnn.pieces with secondary_one.srnn.pieces
                    
                    auto start = x.srnn.pieces.begin() + sub_exprs[pert_ind].first;
                    auto end = std::min(x.srnn.pieces.begin() + sub_exprs[pert_ind].second, x.srnn.pieces.end());
                    x.srnn.pieces.erase(start, end+1);
                    x.srnn.pieces.insert(start, secondary.srnn.pieces.begin(), secondary.srnn.pieces.end()); //could be a move operation: secondary.srnn.pieces doesn't need to be in a defined state after this->params
                }
                
                //Step 5: Evaluate the new mutated `x.srnn.pieces` and update score if needed
                score = x.complete_status(false);
                updateScore(pow(ratio, 1.0f/(i+1)));
            };

            //Step 1: generate a random expression
            while ((score = x.complete_status()) == -1)
            {
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                x.srnn.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                current.push_back(x.srnn.pieces.back());
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
        std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        std::cout << "Best expression = " << best_expression << '\n';
        std::cout << "Best expression (original format) = " << orig_expression << '\n';
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
void GP(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0, std::vector<int> layers = {}, const unsigned long num_epochs = 1000, float eta = 0.5f, float theta = 0.5f)
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
        std::string best_expression, orig_expression;
        
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
        
        auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &layers, &num_epochs, &best_expression, &orig_expression, &eta, &theta]()
        {
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
            Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, layers, num_epochs, eta, theta);
            sync_point.arrive_and_wait();
            Board secondary_one(false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache), secondary_two(false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache); //For crossover and mutations
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
        //        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).first == x.n);
        //        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).second);
                if (score > max_score)
                {
        //            expression = x._to_infix();
        //            orig_expression = x.expression();
                    max_score = score;
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
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
                    x.srnn.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                }
                
                updateScore();
                individuals.push_back(std::make_pair(x.srnn.pieces, score));
                x.srnn.pieces.clear();
            }
            
            auto Mutation = [&](int n)
            {
                //Step 1: Generate a random depth-n sub-expression `secondary_one.srnn.pieces`
                secondary_one.srnn.pieces.clear();
                sub_exprs_1.clear();
                secondary_one.n = n;
                while (secondary_one.complete_status() == -1)
                {
                    temp_legal_moves = secondary_one.get_legal_moves();
                    std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
                    secondary_one.srnn.pieces.push_back(temp_legal_moves[distribution(generator)]);
                }
                
                assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.srnn.pieces) : secondary_one.getRPNdepth(secondary_one.srnn.pieces)).first == secondary_one.n);
                assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.srnn.pieces) : secondary_one.getRPNdepth(secondary_one.srnn.pieces)).second);

                
                //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
                //in `x.srnn.pieces` and store them in an std::vector<std::pair<int, int>>
                //called `sub_exprs_1`.
                x.srnn.pieces = individuals[selector_dist(generator)].first; //A randomly selected individual to be mutated
                secondary_one.get_indices(sub_exprs_1, x.srnn.pieces);
                
                //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `mut_ind`
                std::uniform_int_distribution<int> distribution(0, sub_exprs_1.size() - 1);
                int mut_ind = distribution(generator);
                
                //Step 4: Substitute sub_exprs_1[mut_ind] in x.srnn.pieces with secondary_one.srnn.pieces
                
                auto start = x.srnn.pieces.begin() + sub_exprs_1[mut_ind].first;
                auto end = std::min(x.srnn.pieces.begin() + sub_exprs_1[mut_ind].second, x.srnn.pieces.end()-1);
                x.srnn.pieces.erase(start, end+1);
                x.srnn.pieces.insert(start, secondary_one.srnn.pieces.begin(), secondary_one.srnn.pieces.end());
                
                //Step 5: Evaluate the new mutated `x.srnn.pieces` and update score if needed
                score = x.complete_status(false);
                updateScore();
                individuals.push_back(std::make_pair(x.srnn.pieces, score));
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

                x.srnn.pieces = individual_1.first;
                score = x.complete_status(false);
                updateScore();
                
                individuals.push_back(std::make_pair(x.srnn.pieces, score));
                
                x.srnn.pieces = individual_2.first;
                score = x.complete_status(false);
                updateScore();
                
                individuals.push_back(std::make_pair(x.srnn.pieces, score));
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
        std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        std::cout << "Best expression = " << best_expression << '\n';
        std::cout << "Best expression (original format) = " << orig_expression << '\n';
        
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

void PSO(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0, std::vector<int> layers = {}, const unsigned long num_epochs = 1000, float eta = 0.5f, float theta = 0.5f)
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
        std::string best_expression, orig_expression;
        
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
        
        auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &layers, &num_epochs, &best_expression, &orig_expression, &eta, &theta]()
        {
            std::random_device rand_dev;
            std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
            Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, layers, num_epochs, eta, theta);
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
                    x.srnn.pieces.push_back(temp_legal_moves[particle_positions[i]]); //x.srnn.pieces holds the pieces corresponding to the indices
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
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
        //            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        //            std::cout << "Best expression = " << expression << '\n';
        //            std::cout << "Best expression (original format) = " << orig_expression << '\n';
        //            best_expression = std::move(expression);
                }
                x.srnn.pieces.clear();
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
        std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        std::cout << "Best expression = " << best_expression << '\n';
        std::cout << "Best expression (original format) = " << orig_expression << '\n';
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
void MCTS(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true, double time = 120 /*time to run the algorithm in seconds*/, int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0, std::vector<int> layers = {}, const unsigned long num_epochs = 1000, float eta = 0.5f, float theta = 0.5f)
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
        std::string best_expression, orig_expression;

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
        
        auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &layers, &num_epochs, &best_expression, &orig_expression, &eta, &theta]()
        {
            std::random_device rand_dev;
            std::mt19937 thread_local generator(rand_dev());
            Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, layers, num_epochs, eta, theta);
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
                if (!x.srnn.pieces.empty())
                {
                    state += std::to_string(x.srnn.pieces[x.srnn.pieces.size()-1]) + " ";
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
                    x.srnn.pieces.push_back(best_act);
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
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
                }
                x.srnn.pieces.clear();
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
        std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        std::cout << "Best expression = " << best_expression << '\n';
        std::cout << "Best expression (original format) = " << orig_expression << '\n';
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

void RandomSearch(const Eigen::MatrixXf& data, const int depth = 3, const std::string expression_type = "prefix", const std::string method = "LevenbergMarquardt", const int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", const bool cache = true, const double time = 120.0 /*time to run the algorithm in seconds*/, const int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, const int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0, std::vector<int> layers = {}, const unsigned long num_epochs = 1000, float eta = 0.5f, float theta = 0.5f)
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
        std::string best_expression, orig_expression;
        
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
        
        auto func = [&depth, &expression_type, &method, &num_fit_iter, &fit_grad_method, &data, &cache, &start_time, &time, &max_score, &sync_point, &layers, &num_epochs, &best_expression, &orig_expression, &eta, &theta]()
        {
            std::random_device rand_dev;
            std::mt19937 thread_local generator(rand_dev()); // Mersenne Twister random number generator
            Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache, layers, num_epochs, eta, theta);
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
                        x.srnn.pieces.emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                    }
                }
                assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).first == x.n);
                assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).second);

                if (score > max_score)
                {
                    max_score = score;
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
                }
                x.srnn.pieces.clear();
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
        std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
        std::cout << "Best expression = " << best_expression << '\n';
        std::cout << "Best expression (original format) = " << orig_expression << '\n';
        
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

void HembergBenchmarks(int numIntervals, double time, int numRuns)
{
    RandomSearch(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_1, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_2PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_3, -3.0f, 3.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_3PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_4, -3.0f, 3.0f), 9 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_4PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_5, -3.0f, 3.0f), 10 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_5PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
}

void AIFeynmanBenchmarks(int numIntervals, double time, int numRuns)
{
    RandomSearch(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    RandomSearch(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PostRandomSearchMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PreMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    MCTS(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PostMCTSMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PrePSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    PSO(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PostPSOMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PreGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    GP(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PostGPMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 6, Feynman_1, 1.0f, 5.0f), 4 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_1PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 10, Feynman_2, 1.0f, 5.0f), 5 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_2PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 8, Feynman_3, 1.0f, 5.0f), 6 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_3PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 9, Feynman_4, 1.0f, 5.0f), 7 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_4PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PreSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
    SimulatedAnnealing(generateData(100000, 7, Feynman_5, 1.0f, 5.0f), 8 /*fixed depth*/, "postfix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, time /*time to run the algorithm in seconds*/, numIntervals /*number of equally spaced points in time to sample the best score thus far*/, "Feynman_5PostSimulatedAnnealingMultiThread.txt" /*name of file to save the results to*/, numRuns /*number of runs*/);
}

int main()
{
//    HembergBenchmarks(20 /*numIntervals*/, 120 /*time*/, 50 /*numRuns*/);
//    AIFeynmanBenchmarks(20 /*numIntervals*/, 120 /*time*/, 50 /*numRuns*/);
    
    /*
        Then, move the generated txt files to the directories Hemberg_Benchmarks and
        AIFeynman_Benchmarks and then run PlotData.py
    */
//    f(x_1, x_2, ..., x_{columns-1}) = x_1 + x_2 ...
//
//    x_1       x_2     y
//    1.1123123 3.12312 2.2312312
//    3.12312   4.12431 5.1234123
//    ...
//    3.12312   4.12431 5.1234123
    
//1.1 2.2   Hemberg_2(1.1, 2.2)
//3.0 -0.4  Hemberg_2(3.0, -0.4)
//...
//-1.3 -2.2 Hemberg_2(-1.3, -2.2)
    
    
    RandomSearch(generateData(20 /*rows*/, 3 /*columns*/, Hemberg_2 /*function of two variables to compute the values for the third column*/, -3.0f, 3.0f), 3 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, 1 /*time to run the algorithm in seconds*/, 4 /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, 1 /*number of runs*/, 0 /*num threads*/, {2,10,5,5,1} /*Neural Network*/, 10 /*num_epochs*/, 0.01 /*learning rate*/, 0.1 /*momentum*/);
    MCTS(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 3 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, 1 /*time to run the algorithm in seconds*/, 4 /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, 1 /*number of runs*/, 0 /*num threads*/, {2,10,5,5,1} /*Neural Network*/, 10 /*num_epochs*/, 0.01 /*learning rate*/, 0.1 /*momentum*/);
    PSO(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 3 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, 1 /*time to run the algorithm in seconds*/, 4 /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, 1 /*number of runs*/, 0 /*num threads*/, {2,10,5,5,1} /*Neural Network*/, 10 /*num_epochs*/, 0.01 /*learning rate*/, 0.1 /*momentum*/);
    GP(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 3 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, 1 /*time to run the algorithm in seconds*/, 4 /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, 1 /*number of runs*/, 0 /*num threads*/, {2,10,5,5,1} /*Neural Network*/, 10 /*num_epochs*/, 0.01 /*learning rate*/, 0.1 /*momentum*/);
    SimulatedAnnealing(generateData(20, 3, Hemberg_2, -3.0f, 3.0f), 3 /*fixed depth*/, "prefix", "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/, 1 /*time to run the algorithm in seconds*/, 4 /*number of equally spaced points in time to sample the best score thus far*/, "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/, 1 /*number of runs*/, 0 /*num threads*/, {2,10,5,5,1} /*Neural Network*/, 10 /*num_epochs*/, 0.01 /*learning rate*/, 0.1 /*momentum*/);

    return 0;
}
//git push --set-upstream origin NeuralNetworkWeightUpdate
/*
 Outline for NeuralNetworkWeightUpdate
 
 1. Pick test SR function
 2. Pick a Neural network architecture with N inputs, 1-10 hidden layers with 1 - 10 neurons each, output_type = "none", learning_rate in {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}
    - Fixed at start
 3. Test baseline weight-update rule and set to `best-expression`
 4. While some_condition:
    - Test SR update rule for some number of epochs
        - Get score
            - Fitting -> 
        - Update `best-expression` if needed
    
 
 */

//g++ -Wall -std=c++20 -o NeuralNetworks_VecSR NeuralNetworks_VecSR.cpp MLP_Vec.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -ffast-math -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native
