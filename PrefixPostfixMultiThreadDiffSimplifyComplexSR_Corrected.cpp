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
#include <complex>
#include <LBFGS.h>
#include <LBFGSB.h>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>
#include <boost/unordered/concurrent_flat_map.hpp>

namespace std
{
    std::string to_string(std::complex<float> val)
    {
        return "(" + std::to_string(val.real()) + " + " + std::to_string(val.imag()) + "i)";
    }
}

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
Eigen::MatrixXcf createLinspaceMatrix(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));
    Eigen::MatrixXcf mat(rows, cols);
    for (int col = 0; col < cols; ++col)
    {
        for (int row = 0; row < rows; ++row)
        {
            mat(row, col) = std::complex<float>(min_vec[col] + (max_vec[col] - min_vec[col]) * row / (rows - 1), 0.0f);
        }
    }
    return mat;
}

Eigen::MatrixXcf generateData(int numRows, int numCols, float min = -3.0f, float max = 3.0f)
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 thread_local gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);

    // Create the matrix
    Eigen::MatrixXcf matrix(numRows, numCols);

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            matrix(i, j) = std::complex<float>(distribution(gen), 0.0f);
        }
    }

    return matrix;
}

int trueMod(int N, int M)
{
    return ((N % M) + M) % M;
};

bool isZero(const Eigen::VectorXcf& vec, float tolerance = 1e-5f)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    return (((vec.array()).abs().maxCoeff()) <= tolerance);
}

bool isConstant(const Eigen::VectorXcf& vec, float tolerance = 1e-5f)
{
    if (vec.size() <= 1)
    {
        return true; // A vector with 0 or 1 element is trivially constant
    }
    std::complex<float> firstElement = vec(0);
    return (vec.array() - firstElement).abs().maxCoeff() <= tolerance;
}

class Data
{
    Eigen::MatrixXcf data;
    std::unordered_map<std::string, Eigen::VectorXcf> features;
    std::vector<Eigen::VectorXcf> rows;
    long num_columns, num_rows;
    
public:
    
    Data() = default; //so we can have a static Data attribute
    
    // Assignment operator
    Data& operator=(const Eigen::MatrixXcf& theData)
    {
        this->data = theData;
        this->num_columns = data.cols();
        this->num_rows = data.rows();
        for (long i = 0; i < this->num_columns; i++) //for each column
        {
            this->features["x"+std::to_string(i)] = Eigen::VectorXcf(this->num_rows);
            for (long j = 0; j < this->num_rows; j++)
            {
                this->features["x"+std::to_string(i)](j) = this->data(j,i);
            }
        }
        this->rows.resize(this->num_rows);
        
        for (long i = 0; i < num_rows; i++)
        {
            this->rows[i] = data.row(i);
        }
        
        return *this;
    }
    
    bool operator==(Data& other)
    {
        return this->data == other.data;
    }
    
    const Eigen::VectorXcf& operator[] (int i){return rows[i];}
    const Eigen::VectorXcf& operator[] (const std::string& i)
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

float MSE(const Eigen::VectorXcf& actual)
{
    return actual.squaredNorm() / actual.size();
}

float loss_func(const Eigen::VectorXcf& actual)
{
    return (1.0f/(1.0f+MSE(actual)));
}

struct Board
{
    static boost::concurrent_flat_map<std::string, Eigen::VectorXcf> inline expression_dict = {};
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
    Eigen::VectorXcf params; //store the parameters of the expression of the current episode after it's completed
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
    static std::atomic<float> inline const_val;
    static std::once_flag inline initialization_flag;  // Flag for std::call_once
    
    size_t reserve_amount;
    int num_fit_iter;
    std::string fit_method;
    
    bool cache;
    bool const_token;
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
    float isConstTol;
    bool LBFGS_real;
    float conj;
    
    Board(std::vector<float> (*diffeq)(Board&), bool primary = true, int n = 3, const std::string& expression_type = "prefix", std::string fitMethod = "PSO", int numFitIter = 1, const Eigen::MatrixXcf& theData = {}, bool visualize_exploration = false, bool cache = false, bool const_tokens = false, float isConstTol = 1e-1f, bool const_token = false) : gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, num_fit_iter{numFitIter}, fit_method{fitMethod}, is_primary{primary}
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
        this->isConstTol = isConstTol;
        this->const_token = const_token;
        
        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                Board::data = theData;
                Board::__num_features = Board::data[0].size();
                printf("Number of features = %d\n", Board::__num_features);
                Board::__input_vars.clear();
                Board::expression_dict.clear();
                Board::__input_vars.reserve(Board::__num_features);
                for (auto i = 0; i < Board::__num_features; i++)
                {
                    Board::__input_vars.push_back("x"+std::to_string(i));
                }
                Board::__input_vars.push_back("i");
                Board::__unary_operators = {"~", "log", "ln", "exp", "cos", "sin", "sqrt", "asin", "arcsin", "acos", "arccos", "tanh", "sech", "conj"};
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
                Board::__other_tokens = {"0", "1", "2"};
                if (const_token)
                {
                    Board::__other_tokens.push_back("const");
                }
                Board::__tokens = Board::__operators;
                for (auto& i: this->Board::__input_vars)
                {
                    Board::__tokens.push_back(i);
                }
                for (auto& i: Board::__other_tokens)
                {
                    Board::__tokens.push_back(i);
                }
                if (const_token)
                {
                    assert(Board::__tokens.back() == "const");
                }
                Board::action_size = Board::__tokens.size();
                Board::__tokens_float.clear();
                Board::__tokens_float.reserve(Board::action_size);
                for (int i = 1; i <= Board::action_size; ++i)
                {
                    Board::__tokens_float.push_back(i);
                }
                if (const_token)
                {
                    const_val = Board::__tokens_float.back();
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
                int ops_plus_features_plus_i = num_operators + Board::__num_features + 1;
                Board::__input_vars_float.clear();
                for (int i = num_operators + 1; i <= ops_plus_features_plus_i; i++)
                {
                    Board::__input_vars_float.push_back(i);
                }
                Board::__other_tokens_float.clear();
                for (int i = ops_plus_features_plus_i + 1; i <= Board::action_size; i++)
                {
                    Board::__other_tokens_float.push_back(i);
                }
                for (int i = 0; i < Board::action_size; i++)
                {
                    Board::__tokens_dict[Board::__tokens_float[i]] = Board::__tokens[i];
                    Board::__tokens_inv_dict[Board::__tokens[i]] = Board::__tokens_float[i];
                }
                
                if (const_tokens)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__tokens_float;
                }
                
                else
                {
                    Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__operators_float;
                }

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
                    if (!const_tokens)
                    {
                        Board::una_bin_leaf_legal_moves_dict[true][true][true].push_back(i);
                    }
                }
                if (const_tokens)
                {
                    for (float i: Board::__other_tokens_float)
                    {
                        Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                        Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                        Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                    }
                }
                // Erase 'conj' from the containers containing unary operators
                this->conj = Board::__tokens_inv_dict["conj"];
                auto& v1 = Board::una_bin_leaf_legal_moves_dict[true][true][true];
                v1.erase(std::remove(v1.begin(), v1.end(), conj), v1.end());
                auto& v2 = Board::una_bin_leaf_legal_moves_dict[true][true][false];
                v2.erase(std::remove(v2.begin(), v2.end(), conj), v2.end());
                auto& v3 = Board::una_bin_leaf_legal_moves_dict[true][false][true];
                v3.erase(std::remove(v3.begin(), v3.end(), conj), v3.end());
                auto& v4 = Board::una_bin_leaf_legal_moves_dict[true][false][false];
                v4.erase(std::remove(v4.begin(), v4.end(), conj), v4.end());
                
                // Erase 'i' from the containers containing leaves
                float I = Board::__tokens_inv_dict["i"];
                printf("I = %f\n", I);
//                exit(1);
                auto& v5 = Board::una_bin_leaf_legal_moves_dict[true][true][true];
                v5.erase(std::remove(v5.begin(), v5.end(), I), v5.end());
                auto& v6 = Board::una_bin_leaf_legal_moves_dict[true][false][true];
                v6.erase(std::remove(v6.begin(), v6.end(), I), v6.end());
                auto& v7 = Board::una_bin_leaf_legal_moves_dict[false][true][true];
                v7.erase(std::remove(v7.begin(), v7.end(), I), v7.end());
                auto& v8 = Board::una_bin_leaf_legal_moves_dict[false][false][true];
                v8.erase(std::remove(v8.begin(), v8.end(), I), v8.end());
                
                printf("Board::una_bin_leaf_legal_moves_dict[true][true][true]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[true][true][true]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
                
                printf("Board::una_bin_leaf_legal_moves_dict[true][true][false]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[true][true][false]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
                
                printf("Board::una_bin_leaf_legal_moves_dict[true][false][true]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[true][false][true]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
                
                printf("Board::una_bin_leaf_legal_moves_dict[true][false][false]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[true][false][false]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
                
                printf("Board::una_bin_leaf_legal_moves_dict[false][true][true]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[false][true][true]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
                
                printf("Board::una_bin_leaf_legal_moves_dict[false][true][false]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[false][true][false]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
                
                printf("Board::una_bin_leaf_legal_moves_dict[false][false][true]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[false][false][true]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
               
                printf("Board::una_bin_leaf_legal_moves_dict[false][false][false]: ");
                for (float i: Board::una_bin_leaf_legal_moves_dict[false][false][false]) {std::cout << Board::__tokens_dict[i] << ' ';}puts("");
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
            if ((this->const_token && (token >= const_val)) || (Board::__tokens_dict[token] == "const"))
            {
                count++;
            }
        }
        return count;
    }
    
    int __num_consts_diff() const
    {
        int count = 0;

        for (float token : diffeq_result)
        {
            if ((this->const_token && (token >= const_val)) || (Board::__tokens_dict[token] == "const"))
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
                //basic constraints for depth
                bool una_allowed = (getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, true /*unary*/, false /*leaf*/).first <= this->n);
                bool bin_allowed = (getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, true /*binary*/, false /*unary*/, false /*leaf*/).first <= this->n);
                bool leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, this->cache /*cache*/, false /*modify*/, false /*binary*/, false /*unary*/, true /*leaf*/).first < this->n && (num_leaves == num_binary))));
                std::vector<float> legal_moves = Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
                
                //more complicated constraints for simplification
                if (leaf_allowed)
                {
                    size_t pcs_sz = pieces.size();
                    if (pcs_sz >= 2 && Board::__tokens_dict[pieces[pcs_sz-2]] == "/") // "/ x{i}" should not result in "/ x{i} x{i}" as that is 1
                    {
                        for (float i: Board::__input_vars_float)
                        {
                            if (pieces.back() == i)
                            {
                                if (legal_moves.size() > 1)
                                {
                                    legal_moves.erase(std::remove(legal_moves.begin(), legal_moves.end(), i), legal_moves.end()); //remove "x{i}" from legal_moves
                                }
                                else //if x{i} is the only legal move, then we'll change "/" to another binary operator, like "+", "*", or "^"
                                {
                                    std::vector<float> sub_bin_ops = {Board::__tokens_inv_dict["*"], Board::__tokens_inv_dict["+"], Board::__tokens_inv_dict["^"]};
                                    std::uniform_int_distribution<int> distribution(0, 2);
                                    pieces[pcs_sz-2] = sub_bin_ops[distribution(gen)];
                                }
                                break;
                            }
                        }
                    }
                }
                
                return legal_moves;

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
        std::string temp, token;
        temp.reserve(2*pieces.size());
        size_t const_idx, sz = pieces.size() - 1;
        int const_index = ((expression_type == "postfix") ? 0 : this->params.size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            if (this->const_token && pieces[i] >= this->const_val)
            {
                const_idx = pieces[i] - this->const_val;
                token = "const" + std::to_string(const_idx);
            }
            else
            {
                token = Board::__tokens_dict[pieces[i]];
            }
            
            if (token.substr(0,5) == "const")
            {
                if (!this->const_token)
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
                    temp += ((i!=sz) ? std::to_string((this->params)(const_idx)) + " " : std::to_string((this->params)(const_idx)));
                }
            }
            else
            {
                temp += ((i!=sz) ? token + " " : token);
            }
        }
        return temp;
    }
    
    std::string _to_infix(bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        size_t const_counter = 0, const_idx;
        std::string result, token;
                
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            if (this->const_token && pieces[i] >= this->const_val)
            {
                const_idx = pieces[i] - this->const_val;
                token = "const" + std::to_string(const_idx);
            }
            else
            {
                token = Board::__tokens_dict[pieces[i]];
            }
            std::cout << "token = " << token << '\n';
            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
            {
                if ((!show_consts) || token.substr(0,5) != "const")
                {
                    stack.push(token);
                }
                else if (this->const_token)
                {
                    stack.push(std::to_string((this->params)(const_idx)));
                }
                else
                {
                    stack.push(std::to_string((this->params)(const_counter++)));
                }
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
        std::string temp, token;
        temp.reserve(2*pieces.size());
        size_t const_idx, sz = pieces.size() - 1;
        int const_index = ((expression_type == "postfix") ? 0 : this->params.size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            if (this->const_token && pieces[i] >= this->const_val)
            {
                const_idx = pieces[i] - this->const_val;
                token = "const" + std::to_string(const_idx);
            }
            else
            {
                token = Board::__tokens_dict[pieces[i]];
            }
            
            if (token.substr(0,5) == "const")
            {
                if (!this->const_token)
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
                    temp += ((i!=sz) ? std::to_string((this->params)(const_idx)) + " " : std::to_string((this->params)(const_idx)));
                }
            }
            else
            {
                temp += ((i!=sz) ? token + " " : token);
            }
        }
        return temp;
    }
    
    std::string _to_infix(const std::vector<float>& pieces, bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        size_t const_counter = 0, const_idx;
        std::string result, token;

        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            if (this->const_token && pieces[i] >= this->const_val)
            {
                const_idx = pieces[i] - this->const_val;
                token = "const" + std::to_string(const_idx);
            }
            else
            {
                token = Board::__tokens_dict[pieces[i]];
            }

            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
            {
                if ((!show_consts) || token.substr(0,5) != "const")
                {
                    stack.push(token);
                }
                else if (this->const_token)
                {
                    stack.push(std::to_string((this->params)(const_idx)));
                }
                else
                {
                    stack.push(std::to_string((this->params)(const_counter++)));
                }
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

    Eigen::VectorXcf expression_evaluator(const Eigen::VectorXcf& params, const std::vector<float>& pieces) const
    {
        std::stack<const Eigen::VectorXcf> stack;
        size_t const_count = 0, const_idx;
        std::string token;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (static_cast<int>(pieces.size()) - 1) : 0); (is_prefix ? (i >= 0) : (i < static_cast<int>(pieces.size()))); (is_prefix ? (i--) : (i++)))
        {
            if (this->const_token && (pieces[i] >= const_val))
            {
                const_idx = pieces[i] - this->const_val;
                token = "const" + std::to_string(const_idx);
            }
            else
            {
               token = Board::__tokens_dict[pieces[i]];
            }
            assert(token.size());
            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) //not an operator, i.e., a leaf
            {
                if (token.substr(0,5) == "const")
                {
                    stack.push(Eigen::VectorXcf::Constant(Board::data.numRows(), params((!this->const_token ? (const_count++) : const_idx))));
                }
                else if (token == "0")
                {
                    stack.push(Eigen::VectorXcf::Zero(Board::data.numRows()));
                }
                else if (token == "1")
                {
                    stack.push(Eigen::VectorXcf::Ones(Board::data.numRows()));
                }
                else if (token == "2")
                {
                    stack.push(Eigen::VectorXcf::Constant(Board::data.numRows(), std::complex<float>(2.0f, 0.0f)));
                }
                else if (isFloat(token))
                {
                    stack.push(Eigen::VectorXcf::Constant(Board::data.numRows(), std::complex<float>(std::stof(token), 0.0f)));
                }
                else if (token == "i")
                {
                    stack.push(Eigen::VectorXcf::Constant(Board::data.numRows(), std::complex<float>(0, 1)));
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
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().cos());
                }
                else if (token == "exp")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().exp());
                }
                else if (token == "sqrt")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sqrt());
                }
                else if (token == "sin")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().sin());
                }
                else if (token == "asin" || token == "arcsin")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().asin());
                }
                else if (token == "log" || token == "ln")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().log());
                }
                else if (token == "tanh")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().tanh());
                }
                else if (token == "sech")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(1/temp.array().cosh());
                }
                else if (token == "acos" || token == "arccos")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().acos());
                }
                else if (token == "~") //unary minus
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(-temp.array());
                }
                else if (token == "conj")
                {
                    Eigen::VectorXcf temp = stack.top();
                    stack.pop();
                    stack.push(temp.array().conjugate());
                }
            }
            else // binary operator
            {
                Eigen::VectorXcf left_operand = stack.top();
                stack.pop();
                Eigen::VectorXcf right_operand = stack.top();
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
    
    bool AsyncPSO()
    {
        bool improved = false;
        auto start_time = Clock::now();
        Eigen::VectorXcf particle_positions(this->params.size()), x(this->params.size());
        Eigen::VectorXcf v(this->params.size());
        float rp, rg;

        for (long i = 0; i < this->params.size(); i++)
        {
            particle_positions(i) = x(i) = std::complex<float>(pos_dist(gen), pos_dist(gen));
            v(i) = std::complex<float>(vel_dist(gen), vel_dist(gen));
        }

        float swarm_best_score = loss_func(expression_evaluator(this->params, this->diffeq_result));
        float fpi = loss_func(expression_evaluator(particle_positions, this->diffeq_result));
        float fxi;
        std::complex<float> temp;
        
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
                //real
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i).real(K*(v(i).real() + phi_1*rp*(particle_positions(i).real() - x(i).real()) + phi_2*rg*((this->params)(i).real() - x(i).real())));
                x(i).real(x(i).real() + v(i).real());
                
                fpi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
//                    std::cout << "Iteration " << j << ": Changing param " << i << " from " << (this->params)[i]
//                    << " to " << particle_positions[i] << ". Score from " << swarm_best_score << " to " << fpi
//                    << ".\n";
                    (this->params)(i) = particle_positions(i);
                    improved = true;
                    swarm_best_score = fpi;
                }
                
                //imag
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i).imag(K*(v(i).imag() + phi_1*rp*(particle_positions(i).imag() - x(i).imag()) + phi_2*rg*((this->params)(i).imag() - x(i).imag())));
                x(i).imag(x(i).imag() + v(i).imag());
                
                fpi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
//                    std::cout << "Iteration " << j << ": Changing param " << i << " from " << (this->params)[i]
//                    << " to " << particle_positions[i] << ". Score from " << swarm_best_score << " to " << fpi
//                    << ".\n";
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
        Eigen::VectorXcf particle_positions(this->params.size()), x(this->params.size());
        Eigen::VectorXcf v(this->params.size());
        float rp, rg;

        for (long i = 0; i < this->params.size(); i++)
        {
            particle_positions(i) = x(i) = std::complex<float>(pos_dist(gen), pos_dist(gen));
            v(i) = std::complex<float>(vel_dist(gen), vel_dist(gen));
        }

        float swarm_best_score = loss_func(expression_evaluator(this->params, this->diffeq_result));
        float fpi = loss_func(expression_evaluator(particle_positions, this->diffeq_result));
        float fxi;
        std::complex<float> temp;
        
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
                //real
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i).real(K*(v(i).real() + phi_1*rp*(particle_positions(i).real() - x(i).real()) + phi_2*rg*((this->params)(i).real() - x(i).real())));
                x(i).real(x(i).real() + v(i).real());
                
                fpi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
//                    std::cout << "Iteration " << j << ": Changing param " << i << " from " << (this->params)[i]
//                    << " to " << particle_positions[i] << ". Score from " << swarm_best_score << " to " << fpi
//                    << ".\n";
                    (this->params)(i) = particle_positions(i);
                    improved = true;
                    swarm_best_score = fpi;
                }
                
                //imag
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i).imag(K*(v(i).imag() + phi_1*rp*(particle_positions(i).imag() - x(i).imag()) + phi_2*rg*((this->params)(i).imag() - x(i).imag())));
                x(i).imag(x(i).imag() + v(i).imag());
                
                fpi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions, this->diffeq_result)); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
//                    std::cout << "Iteration " << j << ": Changing param " << i << " from " << (this->params)[i]
//                    << " to " << particle_positions[i] << ". Score from " << swarm_best_score << " to " << fpi
//                    << ".\n";
                    (this->params)(i) = particle_positions(i);
                    improved = true;
                    swarm_best_score = fpi;
                }
            }
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
        
    float fitFunctionToData()
    {
        float score = 0.0f;
        Eigen::VectorXcf expression_eval = expression_evaluator(this->params, this->pieces);
        if ((Board::__num_features == 1) && isConstant(expression_eval, this->isConstTol)) //Ignore the trivial solution (1-d functions)!
        {
            return score;
        }
        else if (Board::__num_features > 1)
        {
            std::vector<int> grasp;
            for (const std::string& i: Board::__input_vars)
            {
                if (this->expression_type == "prefix")
                {
                    this->derivePrefix(0, this->pieces.size() - 1, i, this->pieces, grasp);
                }
                else //postfix
                {
                    this->derivePostfix(0, this->pieces.size() - 1, i, this->pieces, grasp);
                }
                if (isZero(expression_evaluator(this->params, this->derivat), this->isConstTol)) //Ignore the trivial solution (N-d functions)!
                {
                    return score;
                }
            }
        }
        if (this->params.size())
        {
            this->diffeq_result = diffeq(*this);
            if (this->__num_consts_diff())
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
                Eigen::VectorXcf temp_vec;
                
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
                            
                score = loss_func(expression_evaluator(temp_vec, this->diffeq_result));
            }
            else
            {
                score = loss_func(expression_evaluator(this->params, this->diffeq_result));
            }
        }
        else
        {
            this->diffeq_result = diffeq(*this);
            score = loss_func(expression_evaluator(this->params, this->diffeq_result));
        }
        return score;
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
            
            if (is_primary) //this->params is equal to the number of constants in the differential equation this->diffeq (i.e., from VortexRadialProfile)
            {
                this->expression_string.clear();
                this->expression_string.reserve(8*pieces.size());
                size_t const_count = 0;
                
                for (float& i: this->pieces)
                {
                    if (this->const_token && (i == this->const_val))
                    {
                        i += const_count++;
                    }
                    this->expression_string += std::to_string(i)+" ";
                }

                if (!Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, Eigen::VectorXcf());
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
    const Eigen::VectorXcf& operator[] (int i)
    {
        return Board::data[i];
    }
    const Eigen::VectorXcf& operator[] (const std::string& i)
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
            this->derivat.reserve(1000);
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
        
        if (this->const_token && prefix[low] >= Board::const_val)
        {
            this->derivat.push_back(Board::__tokens_inv_dict["0"]);
            return;
        }

        if (Board::__tokens_dict[prefix[low]] == "+" || Board::__tokens_dict[prefix[low]] == "-")
        {
            int op_idx = this->derivat.size();
            this->derivat.push_back(prefix[low]); //+/-
            int temp = low+1+grasp[low+1];
            int x_prime_low = this->derivat.size();
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true);  /* +/- x' */
            int x_prime_high = this->derivat.size();
            derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* +/- x' y' */
            int y_prime_high = derivat.size();
            int step;
            
            /*
             Simplification cases:
                
              1.) y' == 0, +/-, x'
              2.) x' == 0,   +, y'
              3.) x' == 0,   -, ~ y'
             
             */
            
            if (Board::__tokens_dict[derivat[x_prime_high]] == "0") //1.) +/- x' 0 -> x'
            {
    //            puts("hi 147");
                //remove y'
                if (x_prime_high == static_cast<int>(derivat.size()) - 1)
                {
                    derivat.pop_back();
                }
                else
                {
                    derivat.erase(derivat.begin() + x_prime_high, derivat.end());
                }
                derivat.erase(derivat.begin() + op_idx); //remove +/- operator at beginning
            }
            
            else if (Board::__tokens_dict[derivat[x_prime_low]] == "0") //2.) and 3.)
            {
    //            puts("hi 162");
                if (Board::__tokens_dict[prefix[low]] == "+") //2.) + 0 y' -> y'
                {
                    derivat.erase(derivat.begin() + op_idx, derivat.begin() + x_prime_high); //remove "+" and "x'"
                }
                else //3.) prefix[low] == "-", - 0 y' -> ~ y'
                {
    //                puts("hi 170");
                    derivat[op_idx] = Board::__tokens_inv_dict["~"]; //change binary minus to unary minus
                    derivat.erase(derivat.begin() + x_prime_low); //remove x'
                }
            }
            else if ((Board::__tokens_dict[prefix[low]] == "-") && ((step = (y_prime_high - x_prime_high)) == (x_prime_high - x_prime_low)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
            {
//                puts("hi 194");
                assert(derivat[op_idx] == prefix[low]);
                derivat[op_idx] = Board::__tokens_inv_dict["0"]; //change "-" to "0";
                derivat.erase(derivat.begin() + op_idx + 1, derivat.begin() + y_prime_high);
            }
        }
        else if (Board::__tokens_dict[prefix[low]] == "*")
        {
            derivat.push_back(Board::__tokens_inv_dict["+"]); /* +  */
            derivat.push_back(Board::__tokens_inv_dict["*"]); /* + * */
            int x_low = derivat.size();
            int temp = low+1+grasp[low+1];
            for (int k = low+1; k <= temp; k++) /* + * x */
            {
                derivat.push_back(prefix[k]);
            }
            if (Board::__tokens_dict[derivat[x_low]] == "0") //* 0 y' -> 0
            {
    //            puts("hi 187");
                derivat[x_low - 1] = Board::__tokens_inv_dict["0"]; //change "*" to "0"
                derivat.erase(derivat.begin() + x_low); //erase x
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* + * x y' */
                if (Board::__tokens_dict[derivat[y_prime_low]] == "0") //* x 0 -> 0
                {
    //                puts("hi 197");
                    derivat[x_low - 1] = Board::__tokens_inv_dict["0"]; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_low, derivat.end()); //erase x and y'
                }
                else if (Board::__tokens_dict[derivat[x_low]] == "1") //* 1 y' -> y'
                {
    //                puts("hi 203");
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + x_low + 1); //erase "*" and "1"
                }
                else if (Board::__tokens_dict[derivat[y_prime_low]] == "1") //* x 1 -> x
                {
    //                puts("hi 208");
                    derivat.pop_back(); //remove "1"
                    derivat.erase(derivat.begin() + x_low - 1); //remove "*"
                }
            }
            derivat.push_back(Board::__tokens_inv_dict["*"]); /* + * x y' * */
            int x_prime_low = derivat.size();
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* + * x y' * x' */
            if (Board::__tokens_dict[derivat[x_prime_low]] == "0") //* 0 y -> 0
            {
    //            puts("hi 218");
                derivat[x_prime_low - 1] = Board::__tokens_inv_dict["0"]; //change "*" to "0"
                derivat.erase(derivat.begin() + x_prime_low); //erase x'
            }
            else
            {
                int y_low = derivat.size();
                for (int k = temp+1; k <= temp+1+grasp[temp+1]; k++)
                {
                    derivat.push_back(prefix[k]); /* + * x y' * x' y */
                }
                if (Board::__tokens_dict[derivat[y_low]] == "0") //* x' 0 -> 0
                {
    //                puts("hi 231");
                    derivat[x_prime_low - 1] = Board::__tokens_inv_dict["0"]; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //erase x' and y
                }
                else if (Board::__tokens_dict[derivat[x_prime_low]] == "1") //* 1 y -> y
                {
    //                puts("hi 237");
                    derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
                }
                else if (Board::__tokens_dict[derivat[y_low]] == "1") //* x' 1 -> x'
                {
    //                puts("hi 242");
                    derivat.pop_back(); //remove "1"
                    assert(derivat[x_prime_low - 1] == Board::__tokens_inv_dict["*"]);
                    derivat.erase(derivat.begin() + x_prime_low - 1); //remove "*"
                }
            }
            if (Board::__tokens_dict[derivat[x_low - 1]] == "0") //+ 0 * x' y -> * x' y
            {
    //            puts("hi 249");
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //remove "+" and "0"
            }
            else if (Board::__tokens_dict[derivat[x_prime_low - 1]] == "0") //+ * x y' 0 -> * x y'
            {
    //            puts("hi 254");
                assert(static_cast<int>(derivat.size()) == x_prime_low);
                derivat.erase(derivat.begin() + x_low - 2); //erase "+"
                derivat.pop_back(); //remove "0"
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "/")
        {
            int div_idx = derivat.size();
            derivat.push_back(Board::__tokens_inv_dict["/"]); /* / */
            derivat.push_back(Board::__tokens_inv_dict["-"]); /* / - */
            derivat.push_back(Board::__tokens_inv_dict["*"]); /* / - * */
            int temp = low+1+grasp[low+1];
            int x_prime_low = derivat.size();
            int k;
            derivePrefixHelper(low+1, temp, dx, prefix, grasp, true); /* / - * x' */
            if (Board::__tokens_dict[derivat[x_prime_low]] == "0") //* 0 y -> 0
            {
    //            puts("hi 297");
                derivat[x_prime_low - 1] = Board::__tokens_inv_dict["0"]; //change "*" to "0"
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
                if (Board::__tokens_dict[derivat[y_low]] == "0") //* x' 0 -> 0
                {
    //                puts("hi 312");
                    derivat[x_prime_low - 1] = Board::__tokens_inv_dict["0"]; //change "*" to "0"
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //remove x' and 0
                }
                else if (Board::__tokens_dict[derivat[y_low]] == "1") //* x' 1 -> x'
                {
    //                puts("hi 318");
                    assert(y_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                    derivat.pop_back(); //erase the "1"
                }
                else if (Board::__tokens_dict[derivat[x_prime_low]] == "1") //* 1 y -> y
                {
//                    puts("hi 326");
                    derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
                }
            }
            derivat.push_back(Board::__tokens_inv_dict["*"]); /* / - * x' y * */
            int x_low = derivat.size();
            for (k = low+1; k <= temp; k++) /* / - * x' y * x */
            {
                derivat.push_back(prefix[k]);
            }
            if (Board::__tokens_dict[derivat[x_low]] == "0") //* 0 y' -> 0
            {
    //            puts("hi 338");
                derivat.erase(derivat.begin() + x_low - 1); //erase "*"
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePrefixHelper(temp+1, temp+1+grasp[temp+1], dx, prefix, grasp, true); /* / - * x' y * x y' */
                if (Board::__tokens_dict[derivat[y_prime_low]] == "0") //* x 0 -> 0
                {
    //                puts("hi 347");
                    assert(y_prime_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and x
                }
                else if (Board::__tokens_dict[derivat[x_low]] == "1") //* 1 y' -> y'
                {
    //                puts("hi 352");
                    derivat.erase(derivat.begin() + x_low - 1, derivat.begin() + y_prime_low); //erase * and 1
                }
                else if (Board::__tokens_dict[derivat[y_prime_low]] == "1") //* x 1 -> x
                {
    //                puts("hi 357");
                    assert(y_prime_low == static_cast<int>(derivat.size()) - 1);
                    derivat.erase(derivat.begin() + x_low - 1); //erase "*"
                    derivat.pop_back(); //remove the "1"
                }
            }
            
            if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - (x_low - 1))) && (areDerivatRangesEqual(x_prime_low - 1, x_low - 1, k))) //- thing1 thing1 -> 0
            {
    //            puts("hi 367");
                derivat[div_idx] = Board::__tokens_inv_dict["0"];
                derivat.erase(derivat.begin() + div_idx + 1, derivat.end()); //erase everything else
            }
            else
            {
                if (Board::__tokens_dict[derivat[x_prime_low - 1]] == "0") //- 0 * x y' -> ~ * x y'
                {
    //                puts("hi 375");
                    derivat[x_prime_low - 2] = Board::__tokens_inv_dict["~"]; //change "-" to "~"
                    derivat.erase(derivat.begin() + x_prime_low - 1); //erase "0"
                }
                else if (Board::__tokens_dict[derivat[x_low - 1]] == "0") //- * x' y 0 -> * x' y
                {
//                    puts("hi 381");
                    assert(static_cast<int>(derivat.size()) == x_low);
                    derivat.erase(derivat.begin() + x_prime_low - 2); //erase the "-"
                    derivat.pop_back(); //erase the "0"
                }
                derivat.push_back(Board::__tokens_inv_dict["*"]); /* / - * x' y * x y' * */
                int y_low = derivat.size();
                for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* / - * x' y * x y' * y */
                {
                    derivat.push_back(prefix[k]);
                }
                if (Board::__tokens_dict[derivat[y_low]] == "1") // / - * x' y * x y' * 1 1 ->  - * x' y * x y'
                {
    //                puts("hi 381");
                    assert(y_low == static_cast<int>(derivat.size()) - 1);
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
        
        else if (Board::__tokens_dict[prefix[low]] == "^")
        {
            derivat.push_back(Board::__tokens_inv_dict["*"]); /* * */
            derivat.push_back(Board::__tokens_inv_dict["^"]); /* * ^ */
            int temp = low+1+grasp[low+1];
            int k;
            int x_low = derivat.size();
            for (k = low+1; k <= temp; k++) /* * ^ x */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_low] == Board::__tokens_inv_dict["0"]) //* ^ 0 y (* ln 0 y)' -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
            {
    //            puts("hi 454");
                assert(x_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
                return;
            }
            else if (derivat[x_low] == Board::__tokens_inv_dict["1"]) //* ^ 1 y (* ln 1 y)' -> 0 (because ln(1) is 0)
            {
    //            puts("hi 461");
                assert(x_low == static_cast<int>(derivat.size()) - 1);
                derivat[x_low] = Board::__tokens_inv_dict["0"]; //change "1" to "0"
                derivat.erase(derivat.begin() + x_low - 2, derivat.begin() + x_low); //erase "*" and "^"
                return;
            }
            int y_low = derivat.size();
            for (k = temp+1; k <= temp+1+grasp[temp+1]; k++) /* * ^ x y */
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[y_low] == Board::__tokens_inv_dict["0"]) //* ^ x 0 (* ln x 0)' -> 0
            {
                assert(y_low == static_cast<int>(derivat.size()) - 1);
    //            puts("hi 474");
                derivat[x_low - 2] = Board::__tokens_inv_dict["0"]; //change "*" to "0)
                derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
                return;
            }
            else if (derivat[y_low] == Board::__tokens_inv_dict["1"]) //^ x 1 -> x
            {
                assert(y_low == static_cast<int>(derivat.size()) - 1);
                derivat.pop_back(); //erase the "1"
                derivat.erase(derivat.begin() + x_low - 1); //erase the "*"
    //            puts("hi 485");
            }
            std::vector<float> prefix_temp;
            std::vector<int> grasp_temp;
            size_t reserve_amount = up+2-low; //up-low -> x and y, 2 -> ln and *, => up+2-low -> * ln x y
            prefix_temp.reserve(reserve_amount);
            grasp_temp.reserve(reserve_amount);
            prefix_temp.push_back(Board::__tokens_inv_dict["*"]); /* * */
            prefix_temp.push_back(Board::__tokens_inv_dict["ln"]); /* * ln */
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
            if (prefix_temp[y_low] == Board::__tokens_inv_dict["1"]) //* ln x 1 -> ln x
            {
    //            puts("hi 506");
                assert(y_low == static_cast<int>(prefix_temp.size()) - 1);
                prefix_temp.pop_back(); //remove the "1"
                prefix_temp.erase(prefix_temp.begin() + x_temp_low - 2); //erase the "*"
            }
            setPrefixGR(prefix_temp, grasp_temp);
            int temp_term_low = derivat.size();
    //        derivat.push_back("1");
            derivePrefixHelper(0, prefix_temp.size() - 1, dx, prefix_temp, grasp_temp, true); /* * ^ x y (* ln x y)' */
            if (derivat[temp_term_low] == Board::__tokens_inv_dict["0"]) //* ^ x y 0 -> 0
            {
    //            puts("hi 516");
                derivat[x_low - 2] = Board::__tokens_inv_dict["0"]; //changing "*" to "0"
                derivat.erase(derivat.begin() + x_low - 1, derivat.end()); //erase the rest
            }
            else if (derivat[temp_term_low] == Board::__tokens_inv_dict["1"]) //* ^ x y 1 -> ^ x y
            {
    //            puts("hi 522");
                assert(temp_term_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_low - 2); //erasing "*"
                derivat.pop_back(); //erasing the "1"
            }
        }

        else if (Board::__tokens_dict[prefix[low]] == "cos")
        {
            derivat.push_back(Board::__tokens_inv_dict["*"]); /* * */
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * x' */
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"]) //* 0 ~ sin x -> 0
            {
//                puts("hi 538");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["~"]); /* * x' ~ */
            derivat.push_back(Board::__tokens_inv_dict["sin"]); /* * x' ~ sin */
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]); /* * x' ~ sin x */
            }
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["1"]) //* 1 ~ sin x -> ~ sin x
            {
//                puts("hi 551");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "sin")
        {
            derivat.push_back(Board::__tokens_inv_dict["*"]); /* * */
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* * x' */
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"]) //* 0 cos x -> 0
            {
//                puts("hi 565");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["cos"]); /* * x' cos */
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]); /* * x' cos x */
            }
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["1"]) //* 1 cos x -> cos x
            {
//                puts("hi 577");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "sqrt")
        {
            derivat.push_back(Board::__tokens_inv_dict["/"]);         /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"])
            {
    //            puts("hi 590");
                assert(x_prime_low == static_cast<int>(derivat.size() - 1));
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "/"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["*"]);         /* / x' * */
            derivat.push_back(Board::__tokens_inv_dict["2"]);         /* / x' * 2 */
            derivat.push_back(Board::__tokens_inv_dict["sqrt"]);      /* / x' * 2 sqrt */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' * 2 sqrt x */
            {
                derivat.push_back(prefix[k]);
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "log" || Board::__tokens_dict[prefix[low]] == "ln")
        {
            derivat.push_back(Board::__tokens_inv_dict["/"]);               /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"]) // / 0 x -> 0
            {
//                puts("hi 578");
                assert(static_cast<int>(derivat.size()) - 1 == x_prime_low);
                derivat[x_prime_low - 1] = Board::__tokens_inv_dict["0"]; //change "/" to 0
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
//                puts("hi 591");
                derivat[x_prime_low - 1] = Board::__tokens_inv_dict["1"]; //change "/" to 0
                derivat.erase(derivat.begin() + x_prime_low, derivat.end()); //delete the rest
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "asin" || Board::__tokens_dict[prefix[low]] == "arcsin")
        {
            derivat.push_back(Board::__tokens_inv_dict["/"]);   /* / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* / x' */
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"])
            {
//                puts("hi 640");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "/"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["sqrt"]); /* / x' sqrt */
            derivat.push_back(Board::__tokens_inv_dict["-"]);    /* / x' sqrt - */
            derivat.push_back(Board::__tokens_inv_dict["1"]);    /* / x' sqrt - 1 */
            derivat.push_back(Board::__tokens_inv_dict["*"]);    /* / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x */
            {
                derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* / x' sqrt - 1 * x x */
            {
                derivat.push_back(prefix[k]);
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "acos" || Board::__tokens_dict[prefix[low]] == "arccos")
        {
            derivat.push_back(Board::__tokens_inv_dict["~"]);   /* ~ */
            derivat.push_back(Board::__tokens_inv_dict["/"]);   /* ~ / */
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ / x' */
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"])
            {
    //            puts("hi 668");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 2, derivat.begin() + x_prime_low); //erase "~" and "/"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["sqrt"]); /* ~ / x' sqrt */
            derivat.push_back(Board::__tokens_inv_dict["-"]);    /* ~ / x' sqrt - */
            derivat.push_back(Board::__tokens_inv_dict["1"]);    /* ~ / x' sqrt - 1 */
            derivat.push_back(Board::__tokens_inv_dict["*"]);    /* ~ / x' sqrt - 1 * */
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x */
            {
                derivat.push_back(prefix[k]);
            }
            for (int k = temp; k <= temp+grasp[temp]; k++) /* ~ / x' sqrt - 1 * x x */
            {
                derivat.push_back(prefix[k]);
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "tanh")
        {
            derivat.push_back(Board::__tokens_inv_dict["*"]);      //*
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"])
            {
//                puts("hi 696");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //delete the "*"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["*"]);      //* x' *
            derivat.push_back(Board::__tokens_inv_dict["sech"]);   //* x' * sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x
            {
                derivat.push_back(prefix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["sech"]);   //* x' * sech x sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * sech x sech x
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["1"]) //* 1 * sech x sech x -> * sech x sech x
            {
//                puts("hi 715");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "sech")
        {
            derivat.push_back(Board::__tokens_inv_dict["*"]); //*
            int x_prime_low = derivat.size();
            int temp = low+1;
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"]) //* 0 * ~ sech x tanh x -> 0
            {
//                puts("hi 722");
                assert(x_prime_low == static_cast<int>(derivat.size()) - 1);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase the "*"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["*"]);      //* x' *
            derivat.push_back(Board::__tokens_inv_dict["~"]);      //* x' * ~
            derivat.push_back(Board::__tokens_inv_dict["sech"]);   //* x' * ~ sech
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x
            {
                derivat.push_back(prefix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["tanh"]);   //* x' * ~ sech x tanh
            for (int k = temp; k <= temp+grasp[temp]; k++) //* x' * ~ sech x tanh x
            {
                derivat.push_back(prefix[k]);
            }
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["1"]) //* 1 exp x -> exp x
            {
//                puts("hi 742");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "exp")
        {
            derivat.push_back(Board::__tokens_inv_dict["*"]);               //*
            int temp = low+1;
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); //* x'
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["0"]) //* 0 exp x -> 0
            {
    //            puts("hi 682");
                assert(static_cast<int>(derivat.size() - 1) == x_prime_low);
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase "*"
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["exp"]);           //* x' exp
            for (int k = temp; k <= temp+grasp[temp]; k++)
            {
                derivat.push_back(prefix[k]);      //* x' exp x
            }
            if (derivat[x_prime_low] == Board::__tokens_inv_dict["1"]) //* 1 exp x -> exp x
            {
//                puts("hi 694");
                derivat.erase(derivat.begin() + x_prime_low - 1, derivat.begin() + x_prime_low + 1); //erase "*" and "1"
            }
        }
        
        else if (Board::__tokens_dict[prefix[low]] == "~")
        {
            int temp = low+1;
            int un_minus_idx = derivat.size();
            derivat.push_back(prefix[low]); /* ~ */
            int x_prime_low = derivat.size();
            derivePrefixHelper(temp, temp+grasp[temp], dx, prefix, grasp, true); /* ~ x' */
            if (Board::__tokens_dict[derivat[x_prime_low]] == "~")
            {
//                puts("hi 590");
                derivat.erase(derivat.begin() + un_minus_idx, derivat.begin() + x_prime_low + 1); //erase the two "~"
            }
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
            this->derivat.reserve(1000);
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

        if (this->const_token && postfix[up] >= Board::const_val)
        {
            this->derivat.push_back(Board::__tokens_inv_dict["0"]);
            return;
        }
        
        if (Board::__tokens_dict[postfix[up]] == "+" || Board::__tokens_dict[postfix[up]] == "-")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true);  /*Putting x'*/
            int x_prime_high = derivat.size();
            derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /*Putting y'*/
            int y_prime_high = derivat.size();
            int step;
            
            /*
             Simplification cases:
                
              1.) y' == 0, +/-, x'
              2.) x' == 0,   +, y'
              3.) x' == 0,   -, y' ~
             
             */
            
            if (Board::__tokens_dict[derivat.back()] == "0") //1.) x' 0 + -> x'
            {
    //            puts("hi 145");
                derivat.pop_back();
            }
            
            else if (Board::__tokens_dict[derivat[x_prime_high - 1]] == "0")
            {
    //            puts("hi 151");
                //erase elements from derivat[x_prime_low] to derivat[x_prime_high-1] inclusive
                derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + x_prime_high); //0 y + -> y
                if (Board::__tokens_dict[postfix[up]] == "-") //3.)
                {
    //                puts("hi 156");
                    derivat.push_back(Board::__tokens_inv_dict["~"]); //0 y - -> y ~
                }
            }
            
            else if ((Board::__tokens_dict[postfix[up]] == "-") && ((step = (x_prime_high - x_prime_low)) == (y_prime_high - x_prime_high)) && (areDerivatRangesEqual(x_prime_low, x_prime_high, step)))
            {
//                puts("hi 180");
                derivat[x_prime_low] = Board::__tokens_inv_dict["0"]; //change first symbol of x' to 0
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.begin() + y_prime_high); //erase the rest of x' and y'
            }
            
            else
            {
                derivat.push_back(postfix[up]);
            }
        }
        else if (Board::__tokens_dict[postfix[up]] == "*")
        {
            int x_low = derivat.size();
            for (int k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                derivat.push_back(postfix[k]);
            }
            if (Board::__tokens_dict[derivat.back()] == "0") //0 y' * -> 0
            {
    //            puts("hi 176");
            }
            else
            {
                int x_high = derivat.size();
                derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x y' */
                if (Board::__tokens_dict[derivat.back()] == "0") //x 0 * -> 0
                {
    //                puts("hi 184");
                    derivat[x_low] = Board::__tokens_inv_dict["0"]; //change first symbol of x to 0
                    derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase rest of x and y'
                }
                else if (Board::__tokens_dict[derivat[x_high - 1]] == "1") //1 y' * -> y'
                {
    //                puts("hi 190");
                    assert(x_low == x_high - 1);
                    derivat.erase(derivat.begin() + x_low); //erase the x since it's 1
                }
                else if (Board::__tokens_dict[derivat.back()] == "1") //x 1 * -> x
                {
    //                puts("hi 196");
                    derivat.pop_back(); //remove the y' since it's 1
                }
                else
                {
                    derivat.push_back(Board::__tokens_inv_dict["*"]); /* x y' "*" */
                }
            }

            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x y' "*" x' */
            if (Board::__tokens_dict[derivat.back()] == "0") //0 y * -> 0
            {
    //            puts("hi 209");
            }
            else
            {
                int y_low = derivat.size();
                for (int k = up-1-grasp[up-1]; k <= up - 1; k++)
                {
                    derivat.push_back(postfix[k]); /* x y' "*" x' y */
                }
                if (Board::__tokens_dict[derivat.back()] == "0") //x' 0 * -> 0
                {
    //                puts("hi 220");
                    derivat.erase(derivat.begin() + x_prime_low, derivat.begin() + y_low); //erase x'
                }
                else if (Board::__tokens_dict[derivat[y_low - 1]] == "1") //1 y * -> y
                {
    //                puts("hi 225");
                    assert(y_low - 1 == x_prime_low);
                    derivat.erase(derivat.begin() + x_prime_low); //remove the 1
                }
                else if (Board::__tokens_dict[derivat.back()] == "1") //x' 1 * -> x'
                {
    //                puts("hi 231");
                    derivat.pop_back(); //remove the "1"
                }
                else
                {
                    derivat.push_back(Board::__tokens_inv_dict["*"]); /* x y' "*" x' y "*" */
                }
            }
            if (Board::__tokens_dict[derivat[x_prime_low - 1]] == "0") // 0 x' y "*" + -> x' y "*"
            {
//                puts("hi 236");
                derivat.erase(derivat.begin() + x_prime_low - 1); //erase 0
            }
            else if (Board::__tokens_dict[derivat.back()] == "0") //x y' "*" 0 + -> x y' "*"
            {
//                puts("hi 241");
                derivat.pop_back();
            }
            else
            {
                derivat.push_back(Board::__tokens_inv_dict["+"]); /* x y' "*" x' y "*" + */
            }
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "/")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-2-grasp[up-1], dx, postfix, grasp, true); /* x' */
            int k;
            if (Board::__tokens_dict[derivat.back()] == "0") //0 y * -> 0
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
                if (Board::__tokens_dict[derivat.back()] == "0") //x' 0 * -> 0
                {
    //                puts("hi 297");
                    derivat.erase(derivat.begin() + x_prime_low, derivat.end() - 1); //erase x'
                }
                else if (Board::__tokens_dict[derivat.back()] == "1") //x' 1 * -> x'
                {
    //                puts("hi 302");
                    derivat.pop_back(); //remove the "1"
                }
                else if (Board::__tokens_dict[derivat[y_low-1]] == "1") //1 y * -> y
                {
    //                puts("hi 307");
                    derivat.erase(derivat.begin() + y_low - 1); //erase the "1"
                }
                else
                {
                    derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' y *  */
                }
            }
            int x_low = derivat.size();
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x' y * x */
            {
                derivat.push_back(postfix[k]);
            }
            if (Board::__tokens_dict[derivat.back()] == "0") //0 y' * -> 0
            {
    //            puts("hi 322");
            }
            else
            {
                int y_prime_low = derivat.size();
                derivePostfixHelper(up-1-grasp[up-1], up-1, dx, postfix, grasp, true); /* x' y * x y' */
                if (Board::__tokens_dict[derivat.back()] == "0") //x 0 * -> 0
                {
    //                puts("hi 330");
                    derivat.erase(derivat.begin() + x_low, derivat.begin() + y_prime_low); //erase x
                }
                else if (Board::__tokens_dict[derivat.back()] == "1") //x 1 * -> x
                {
    //                puts("hi 335");
                    derivat.pop_back(); //erase the 1
                }
                else if (Board::__tokens_dict[derivat[y_prime_low - 1]] == "1") //1 y' * -> y'
                {
    //                puts("hi 340");
                    derivat.erase(derivat.begin() + y_prime_low - 1); //erase the "1"
                }
                else
                {
                    derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' y * x y' * */
                }
            }
            if (((k = (x_low - x_prime_low)) == (static_cast<int>(derivat.size()) - x_low)) && (areDerivatRangesEqual(x_prime_low, x_low, k))) //thing1 thing1 - -> 0
            {
    //            puts("hi 350");
                derivat[x_prime_low] = Board::__tokens_inv_dict["0"]; //change first symbol of x' to 0
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest of x' y * and x y' *
            }
            else
            {
                if (Board::__tokens_dict[derivat[x_low - 1]] == "0") //0 x y' * - -> x y' * ~
                {
    //                puts("hi 358");
                    derivat.erase(derivat.begin() + x_low - 1); //remove "0"
                    derivat.push_back(Board::__tokens_inv_dict["~"]); //add "~" at the end
                }
                else if (Board::__tokens_dict[derivat.back()] == "0") //x' y * 0 - -> x' y *
                {
    //                puts("hi 364");
                    derivat.pop_back(); //remove "0"
                }
                else
                {
                    derivat.push_back(Board::__tokens_inv_dict["-"]); /* x' y * x y' * - */
                }
                for (k = up-1-grasp[up-1]; k <= up-1; k++)      /* x' y * x y' * - y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (Board::__tokens_dict[derivat.back()] == "1") //"1 1 * /" -> ""
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
                    derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' y * x y' * - y y * */
                    derivat.push_back(Board::__tokens_inv_dict["/"]); /* x' y * x y' * - y y * / */
                }
            }
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "^")
        {
            int k;
            int x_low = derivat.size();
            for (k = low; k <= up-2-grasp[up-1]; k++) /* x */
            {
                derivat.push_back(postfix[k]);
            }
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 y ^ (0 ln y *)' * -> 0 (maybe problematic for y < 0, but oh well 😮‍💨)
            {
    //            puts("hi 402");
                return;
            }
            else if (derivat.back() == Board::__tokens_inv_dict["1"]) //1 y ^ (1 ln y *)' * -> 0 (because ln(1) is 0)
            {
                derivat.back() = Board::__tokens_inv_dict["0"];
    //            puts("hi 407");
                return;
            }
            else
            {
                for (k = up-1-grasp[up-1]; k <= up-1; k++) /* x y */
                {
                    derivat.push_back(postfix[k]);
                }
                if (derivat.back() == Board::__tokens_inv_dict["0"]) //x 0 ^ (x ln 0 *)' * -> 0
                {
    //                puts("hi 419");
                    derivat[x_low] = Board::__tokens_inv_dict["0"]; //change the first symbol of x to "0"
                    derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
                    return;
                }
                else if (derivat.back() == Board::__tokens_inv_dict["1"]) //x 1 ^ -> x
                {
//                    puts("hi 426");
                    derivat.pop_back(); //erase the 1
                }
                else
                {
                    derivat.push_back(Board::__tokens_inv_dict["^"]); /* x y ^ */
                }
            }

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
            if (postfix_temp.back() == Board::__tokens_inv_dict["1"]) //x ln 1 * -> x ln
            {
    //            puts("hi 452");
                postfix_temp.pop_back();
            }
            else
            {
                postfix_temp.push_back(Board::__tokens_inv_dict["*"]); /* x ln y * */
            }
            setPostfixGR(postfix_temp, grasp_temp);
            derivePostfixHelper(0, postfix_temp.size() - 1, dx, postfix_temp, grasp_temp, true); /* x y ^ (x ln y *)' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //x y ^ 0 * -> 0
            {
    //            puts("hi 455");
                derivat[x_low] = Board::__tokens_inv_dict["0"]; //change the first symbol of x to "0"
                derivat.erase(derivat.begin() + x_low + 1, derivat.end()); //erase the rest
            }
            else if (derivat.back() == Board::__tokens_inv_dict["1"]) //x y ^ 1 * -> x y ^
            {
    //            puts("hi 460");
                derivat.pop_back(); //erase (x ln y *)'
            }
            else
            {
                derivat.push_back(Board::__tokens_inv_dict["*"]); /* x y ^ (x ln y *)' * */
            }
        }

        else if (Board::__tokens_dict[postfix[up]] == "cos")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 x sin ~ * -> 0
            {
    //            puts("hi 514");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]); /* x' x */
            }
            derivat.push_back(Board::__tokens_inv_dict["sin"]); /* x' x sin */
            derivat.push_back(Board::__tokens_inv_dict["~"]); /* x' x sin ~ */
            if (derivat[x_low - 1] == Board::__tokens_inv_dict["1"]) //1 x sin ~ * -> x sin ~
            {
    //            puts("hi 526");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' x sin ~ * */
            }
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "sin")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 x cos * -> 0
            {
//                puts("hi 540");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]); /* x' x */
            }
            derivat.push_back(Board::__tokens_inv_dict["cos"]); /* x' x cos */
            if (derivat[x_low - 1] == Board::__tokens_inv_dict["1"]) //1 x cos * -> x cos
            {
//                puts("hi 551");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back(Board::__tokens_inv_dict["*"]); /* x' x cos * */
            }
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "sqrt")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 2 x sqrt * / -> 0
            {
//                puts("hi 565");
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["2"]); /* x' 2 */
            for (int k = low; k <= up-1; k++) /* x' 2 x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["sqrt"]);    /* x' 2 x sqrt */
            derivat.push_back(Board::__tokens_inv_dict["*"]);       /* x' 2 x sqrt * */
            derivat.push_back(Board::__tokens_inv_dict["/"]);       /* x' 2 x sqrt * / */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "log" || Board::__tokens_dict[postfix[up]] == "ln")
        {
            int x_prime_low = derivat.size();
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 x / -> 0
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
                //                puts("hi 563");
                derivat[x_prime_low] = Board::__tokens_inv_dict["1"]; //replace first symbol of x' with "1"
                derivat.erase(derivat.begin() + x_prime_low + 1, derivat.end()); //erase the rest
                return;
            }
            
            derivat.push_back(Board::__tokens_inv_dict["/"]);               /* x' x / */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "asin" || Board::__tokens_dict[postfix[up]] == "arcsin")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 1 x x * - sqrt / -> 0
            {
//                puts("hi 610");
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["1"]); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["*"]);   /* x' 1 x x * */
            derivat.push_back(Board::__tokens_inv_dict["-"]);   /* x' 1 x x * - */
            derivat.push_back(Board::__tokens_inv_dict["sqrt"]);   /* x' 1 x x * - sqrt */
            derivat.push_back(Board::__tokens_inv_dict["/"]);   /* x' 1 x x * - sqrt / */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "acos" || Board::__tokens_dict[postfix[up]] == "arccos")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 1 x x * - sqrt / ~ -> 0
            {
    //            puts("hi 633");
                return;
            }
            derivat.push_back(Board::__tokens_inv_dict["1"]); /* x' 1 */
            for (int k = low; k <= up-1; k++) /* x' 1 x */
            {
                derivat.push_back(postfix[k]);
            }
            for (int k = low; k <= up-1; k++) /* x' 1 x x */
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["*"]);   /* x' 1 x x * */
            derivat.push_back(Board::__tokens_inv_dict["-"]);   /* x' 1 x x * - */
            derivat.push_back(Board::__tokens_inv_dict["sqrt"]);   /* x' 1 x x * - sqrt */
            derivat.push_back(Board::__tokens_inv_dict["/"]);   /* x' 1 x x * - sqrt / */
            derivat.push_back(Board::__tokens_inv_dict["~"]);   /* x' 1 x x * - sqrt / ~ */
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "tanh")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); //x'
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 x sech x sech * * -> 0
            {
//                puts("hi 657");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++) //x' x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["sech"]); //x' x sech
            for (int k = low; k <= up-1; k++) //x' x sech x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["sech"]); //x' x sech x sech
            derivat.push_back(Board::__tokens_inv_dict["*"]); //x' x sech x sech *
            if (derivat[x_low - 1] == Board::__tokens_inv_dict["1"]) //1 x sech x sech * * -> x sech x sech * *
            {
//                puts("hi 676");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back(Board::__tokens_inv_dict["*"]);                 //x' x sech ~ x tanh * *
            }
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "sech")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); //x'
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 x sech ~ x tanh * * -> 0
            {
//                puts("hi 681");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++) //x' x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["sech"]);   //x' x sech
            derivat.push_back(Board::__tokens_inv_dict["~"]);      //x' x sech ~
            for (int k = low; k <= up-1; k++) //x' x sech ~ x
            {
                derivat.push_back(postfix[k]);
            }
            derivat.push_back(Board::__tokens_inv_dict["tanh"]);   //x' x sech ~ x tanh
            derivat.push_back(Board::__tokens_inv_dict["*"]);      //x' x sech ~ x tanh *
            if (derivat[x_low - 1] == Board::__tokens_inv_dict["1"]) //1 x sech ~ x tanh * * -> x sech ~ x tanh *
            {
//                puts("hi 699");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back(Board::__tokens_inv_dict["*"]);                 //x' x sech ~ x tanh * *
            }
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "exp")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["0"]) //0 x exp * -> 0
            {
    //            puts("hi 649");
                return;
            }
            int x_low = derivat.size();
            for (int k = low; k <= up-1; k++)
            {
                derivat.push_back(postfix[k]);      /* x' x */
            }
            derivat.push_back(Board::__tokens_inv_dict["exp"]);               /* x' x exp */
            if (derivat[x_low - 1] == Board::__tokens_inv_dict["1"]) //1 x exp * -> x exp
            {
//                puts("hi 660");
                derivat.erase(derivat.begin() + x_low - 1); //erase the "1"
            }
            else
            {
                derivat.push_back(Board::__tokens_inv_dict["*"]);               /* x' x exp * */
            }
        }
        
        else if (Board::__tokens_dict[postfix[up]] == "~")
        {
            derivePostfixHelper(low, up-1, dx, postfix, grasp, true); /* x' */
            if (derivat.back() == Board::__tokens_inv_dict["~"])
            {
                derivat.pop_back(); //two unary minuses cancel each-other
            }
            else
            {
                derivat.push_back(postfix[up]); /* x' ~ */
            }
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

//i * u_t + (1/2) * (u_xx + u_yy) - (|u|^2) * u = 0
std::vector<float> NLS_2D(Board& x) // open /Users/edwardfinkelstein/Downloads/nlw9_rcg_NLW_Part_4_wm\ copy.pdf -> `option` + `command` + `g` -> 212 -> equation 30.1
{
    std::vector<float> result;
    result.reserve(100);
    std::vector<int> grasp;
    grasp.reserve(100);
    std::vector<float> u_x, u_y;
    u_x.reserve(100);
    u_y.reserve(100);
    
    if (x.expression_type == "prefix")
    {
        //- + * i u_t * / 1 2 + u_xx u_yy * * u* u u
        //- + * i u_t * / 1 2 + u_xx u_yy * * conj u u u
        result.push_back(Board::__tokens_inv_dict["-"]);
        result.push_back(Board::__tokens_inv_dict["+"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["i"]);
        x.derivePrefix(0, x.pieces.size()-1, "x2", x.pieces, grasp);
        for (float i: x.derivat) //u_t
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        result.push_back(Board::__tokens_inv_dict["1"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        result.push_back(Board::__tokens_inv_dict["+"]);
        x.derivePrefix(0, x.pieces.size()-1, "x0", x.pieces, grasp);
        u_x = x.derivat;
        x.derivePrefix(0, x.pieces.size()-1, "x0", u_x, grasp);
        for (float i: x.derivat) //u_xx
        {
            result.push_back(i);
        }
        x.derivePrefix(0, x.pieces.size()-1, "x1", x.pieces, grasp);
        u_y = x.derivat;
        x.derivePrefix(0, x.pieces.size()-1, "x1", u_y, grasp);
        for (float i: x.derivat) //u_yy
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
//        for (float i: x.pieces) //u*
//        {
//            if (Board::__tokens_dict[i] == "i")
//            {
//                result.push_back(Board::__tokens_inv_dict["~"]);
//            }
//            result.push_back(i);
//        }
        result.push_back(Board::__tokens_inv_dict["conj"]);
        for (float i: x.pieces) //u
        {
            result.push_back(i);
        }
        for (float i: x.pieces) //u
        {
            result.push_back(i);
        }
        for (float i: x.pieces) //u
        {
            result.push_back(i);
        }
    }
    else
    {
        //  i u_t * 1 2 / u_xx u_yy + * + u* u * u * -
        //  i u_t * 1 2 / u_xx u_yy + * + u conj u * u * -
        result.push_back(Board::__tokens_inv_dict["i"]);
        x.derivePrefix(0, x.pieces.size()-1, "x2", x.pieces, grasp);
        for (float i: x.derivat) //u_t
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["1"]);
        result.push_back(Board::__tokens_inv_dict["2"]);
        result.push_back(Board::__tokens_inv_dict["/"]);
        x.derivePrefix(0, x.pieces.size()-1, "x0", x.pieces, grasp);
        u_x = x.derivat;
        x.derivePrefix(0, x.pieces.size()-1, "x0", u_x, grasp);
        for (float i: x.derivat) //u_xx
        {
            result.push_back(i);
        }
        x.derivePrefix(0, x.pieces.size()-1, "x1", x.pieces, grasp);
        u_y = x.derivat;
        x.derivePrefix(0, x.pieces.size()-1, "x1", u_y, grasp);
        for (float i: x.derivat) //u_yy
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["+"]);
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["+"]);
//        for (float i: x.pieces) //u*
//        {
//            result.push_back(i);
//            if (Board::__tokens_dict[i] == "i")
//            {
//                result.push_back(Board::__tokens_inv_dict["~"]);
//            }
//        }
        for (float i: x.pieces) //u
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["conj"]);
        for (float i: x.pieces) //u
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        for (float i: x.pieces) //u
        {
            result.push_back(i);
        }
        result.push_back(Board::__tokens_inv_dict["*"]);
        result.push_back(Board::__tokens_inv_dict["-"]);
    }
    return result;
}

//(1/2)*R'' + (1/(2r))*R' + (mu - ((S*S)/(2*r*r)))*R - R*R*R = 0
std::vector<float> VortexRadialProfile(Board& x) // open /Users/edwardfinkelstein/Downloads/nlw9_rcg_NLW_Part_4_wm\ copy.pdf -> `option` + `command` + `g` -> 213 -> equation 30.4
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
        x.derivePrefix(0, R_prime.size()-1, "x0", R_prime, grasp); //derivat will store second derivative of R_prime
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
        x.derivePostfix(0, R_prime.size()-1, "x0", R_prime, grasp); //derivat will store second derivative of R_prime
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
void SimulatedAnnealing(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXcf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, bool const_token = false)
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
    
    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &const_token]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        sync_point.arrive_and_wait();
        Board secondary(diffeq, false, 0, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token); //For perturbations
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
                    max_score = score;
                    std::scoped_lock str_lock(Board::thread_locker);
                    best_expression = x._to_infix();
                    orig_expression = x.expression();
                    best_expr_result = x._to_infix(x.diffeq_result);
                    orig_expr_result = x.expression(x.diffeq_result);
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
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    std::cout << "Best diff result = " << best_expr_result << '\n';
    std::cout << "Best diff result (original format) = " << orig_expr_result << '\n';

}

//https://arxiv.org/abs/2310.06609
void GP(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXcf& data, int depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, bool const_token = false)
{
    std::map<int, std::vector<float>> scores; //map to store the scores
    
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp-1);
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
    
    auto func = [&diffeq, &depth, &expression_type, &method, &num_fit_iter, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &const_token]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        Board x(diffeq, true, depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        sync_point.arrive_and_wait();
        Board secondary_one(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token), secondary_two(diffeq, false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token); //For crossover and mutations
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
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_expression = x._to_infix();
                orig_expression = x.expression();
                best_expr_result = x._to_infix(x.diffeq_result);
                orig_expr_result = x.expression(x.diffeq_result);
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
    
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    std::cout << "Best diff result = " << best_expr_result << '\n';
    std::cout << "Best diff result (original format) = " << orig_expr_result << '\n';
}

void PSO(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXcf& data, int real_depth = 3, int imag_depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, bool const_token = false)
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
    
    auto func = [&diffeq, &real_depth, &imag_depth, &expression_type, &method, &num_fit_iter, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &const_token]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        Board real_x(diffeq, false, real_depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token),
        imag_x(diffeq, false, imag_depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        Board x(diffeq, true, std::max(real_depth, imag_depth+1)+1, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        sync_point.arrive_and_wait();
        float score = 0, check_point_score = 0;
        std::vector<float> temp_legal_moves;
        
        size_t temp_sz;
    //    std::string expression, orig_expression, best_expression;
        
        /*
         For this setup, we don't know a-priori the number of particles, so we generate them and their corresponding velocities as needed
         */
        std::vector<float> particle_positions_real, best_positions_real, v_real, curr_positions_real;
        particle_positions_real.reserve(real_x.reserve_amount); //stores record of all current particle position indices
        best_positions_real.reserve(real_x.reserve_amount); //indices corresponding to best pieces
        curr_positions_real.reserve(real_x.reserve_amount); //indices corresponding to x.pieces
        v_real.reserve(real_x.reserve_amount); //stores record of all current particle velocities
        std::unordered_map<float, std::unordered_map<int, int>> Nsa_real;
        std::unordered_map<float, std::unordered_map<int, float>> Psa_real;
        std::unordered_map<int, float> p_i_vals_real, p_i_real;
        
        std::vector<float> particle_positions_imag, best_positions_imag, v_imag, curr_positions_imag;
        particle_positions_imag.reserve(imag_x.reserve_amount); //stores record of all current particle position indices
        best_positions_imag.reserve(imag_x.reserve_amount); //indices corresponding to best pieces
        curr_positions_imag.reserve(imag_x.reserve_amount); //indices corresponding to x.pieces
        v_imag.reserve(imag_x.reserve_amount); //stores record of all current particle velocities
        std::unordered_map<float, std::unordered_map<int, int>> Nsa_imag;
        std::unordered_map<float, std::unordered_map<int, float>> Psa_imag;
        std::unordered_map<int, float> p_i_vals_imag, p_i_imag;
        
        float rp, rg, new_v, c = 0.0f;
        int c_count = 0;
        
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
            //Build real part
            for (int i = 0; (score = real_x.complete_status()) == -1; i++) //i is the index of the token
            {
                rp = real_x.pos_dist(generator), rg = real_x.pos_dist(generator);
                temp_legal_moves = real_x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves

                if (i == static_cast<int>(particle_positions_real.size())) //Then we need to create a new particle with some initial position and velocity
                {
                    particle_positions_real.push_back(real_x.pos_dist(generator));
                    v_real.push_back(real_x.vel_dist(generator));
                }
                
                particle_positions_real[i] = trueMod(std::round(particle_positions_real[i]), temp_sz);
                real_x.pieces.push_back(temp_legal_moves[particle_positions_real[i]]); //x.pieces holds the pieces corresponding to the indices
                curr_positions_real.push_back(particle_positions_real[i]);
                if (i == static_cast<int>(best_positions_real.size()))
                {
                    best_positions_real.push_back(real_x.pos_dist(generator));
                    best_positions_real[i] = trueMod(std::round(best_positions_real[i]), temp_sz);
                }
                //https://hal.science/hal-00764996
                //https://www.researchgate.net/publication/216300408_An_off-the-shelf_PSO
                new_v = (0.721*v[i] + x.phi_1*rg*(best_positions_real[i] - particle_positions_real[i]) + x.phi_2*rp*(p_i[i] - particle_positions_real[i]) + c);
                v_real[i] = copysign(std::min(new_v, FLT_MAX), new_v);
                particle_positions_real[i] += v_real[i];
                Nsa_real[curr_positions_real[i]][i]++;
            }
            for (int i = 0; i < static_cast<int>(curr_positions_real.size()); i++)
            {
                Psa_real[curr_positions[i]][i] = (Psa[curr_positions[i]][i]+score)/Nsa[curr_positions[i]][i];
                if (Psa[curr_positions[i]][i] > p_i_vals[i])
                {
                    p_i[i] = curr_positions[i];
                }
                p_i_vals[i] = std::max(p_i_vals[i], Psa[curr_positions[i]][i]);
                
            }
            
            if (score > max_score)
            {
                for (int idx = 0; idx < static_cast<int>(curr_positions.size()); idx++)
                {
                    best_positions[idx] = curr_positions[idx];
                }
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_expression = x._to_infix();
                orig_expression = x.expression();
                best_expr_result = x._to_infix(x.diffeq_result);
                orig_expr_result = x.expression(x.diffeq_result);
            }
            
            x.pieces.clear();
            real_x.pieces.clear();
            imag_x.pieces.clear();
            curr_positions_real.clear();
            curr_positions_imag.clear();

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
    std::cout << "Best diff result (original format) = " << orig_expr_result << '\n';
}

//https://arxiv.org/abs/2205.13134
void MCTS(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXcf& data, int real_depth = 3, int imag_depth = 3, std::string expression_type = "prefix", std::string method = "LevenbergMarquardt", int num_fit_iter = 1, bool cache = true, double time = 120, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, bool const_token = false)
{
    if (num_threads == 0)
    {
        unsigned int temp = std::thread::hardware_concurrency();
        num_threads = ((temp <= 1) ? 1 : temp-1);
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
    
    auto func = [&diffeq, &real_depth, &imag_depth, &expression_type, &method, &num_fit_iter, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &const_token]()
    {
        std::random_device rand_dev;
        std::mt19937 thread_local generator(rand_dev());
        
        Board real_x(diffeq, false, real_depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token),
        imag_x(diffeq, false, imag_depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        Board x(diffeq, true, std::max(real_depth, imag_depth+1)+1, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        
        sync_point.arrive_and_wait();
        float score = 0.0f, check_point_score = 0.0f, UCT, best_act, UCT_best;
        
        std::vector<float> temp_legal_moves;
        std::unordered_map<std::string, std::unordered_map<float, float>> Qsa_real, Nsa_real;
        std::unordered_map<std::string, float> Ns_real;
        std::unordered_map<std::string, std::unordered_map<float, float>> Qsa_imag, Nsa_imag;
        std::unordered_map<std::string, float> Ns_imag;
        std::string state;
        
        float c = 1.4f; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf, top of page 8 here: https://arxiv.org/pdf/1402.6028.pdf, first formula in section 4. Experiments here: https://cesa-bianchi.di.unimi.it/Pubblicazioni/ml-02.pdf
        std::vector<std::pair<std::string, float>> moveTracker_real;
        moveTracker_real.reserve(real_x.reserve_amount);
        std::vector<std::pair<std::string, float>> moveTracker_imag;
        moveTracker_imag.reserve(imag_x.reserve_amount);
        
        temp_legal_moves.reserve(x.reserve_amount);
        state.reserve(2*x.reserve_amount);
        //        double str_convert_time = 0.0;
        auto getRealString  = [&]()
        {
            if (!real_x.pieces.empty())
            {
                state += std::to_string(real_x.pieces[real_x.pieces.size()-1]) + " ";
            }
        };
        auto getImagString  = [&]()
        {
            if (!imag_x.pieces.empty())
            {
                state += std::to_string(imag_x.pieces[imag_x.pieces.size()-1]) + " ";
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
            //Build real part
            state.clear();
            while ((score = real_x.complete_status()) == -1)
            {
                temp_legal_moves = real_x.get_legal_moves();
//                    auto start_time = Clock::now();
                getRealString();
//                    str_convert_time += timeElapsedSince(start_time);
                UCT = 0.0f;
                UCT_best = -FLT_MAX;
                best_act = -1.0f;
                std::vector<float> best_acts;
                best_acts.reserve(temp_legal_moves.size());
                
                for (float a : temp_legal_moves)
                {
                    if (Nsa_real[state].count(a))
                    {
                        UCT = Qsa_real[state][a] + c*sqrt(log(Ns_real[state])/Nsa_real[state][a]);
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
                real_x.pieces.push_back(best_act);
                moveTracker_real.push_back(make_pair(state, best_act));
                Ns_real[state]++;
                Nsa_real[state][best_act]++;
            }
            
            assert(((real_x.expression_type == "prefix") ? real_x.getPNdepth(real_x.pieces) : real_x.getRPNdepth(real_x.pieces)).first == real_x.n);
            assert(((real_x.expression_type == "prefix") ? real_x.getPNdepth(real_x.pieces) : real_x.getRPNdepth(real_x.pieces)).second);
            
            //Build imaginary part
            state.clear();
            while ((score = imag_x.complete_status()) == -1)
            {
                temp_legal_moves = imag_x.get_legal_moves();
//                    auto start_time = Clock::now();
                getImagString();
//                    str_convert_time += timeElapsedSince(start_time);
                UCT = 0.0f;
                UCT_best = -FLT_MAX;
                best_act = -1.0f;
                std::vector<float> best_acts;
                best_acts.reserve(temp_legal_moves.size());
                
                for (float a : temp_legal_moves)
                {
                    if (Nsa_imag[state].count(a))
                    {
                        UCT = Qsa_imag[state][a] + c*sqrt(log(Ns_imag[state])/Nsa_imag[state][a]);
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
                imag_x.pieces.push_back(best_act);
                moveTracker_imag.push_back(make_pair(state, best_act));
                Ns_imag[state]++;
                Nsa_imag[state][best_act]++;
            }
            
            assert(((imag_x.expression_type == "prefix") ? imag_x.getPNdepth(imag_x.pieces) : imag_x.getRPNdepth(imag_x.pieces)).first == imag_x.n);
            assert(((imag_x.expression_type == "prefix") ? imag_x.getPNdepth(imag_x.pieces) : imag_x.getRPNdepth(imag_x.pieces)).second);
            
            x.pieces.reserve(imag_x.pieces.size() + real_x.pieces.size() + 3);
            //Build total prefix expression: + real * i imag
            if (x.expression_type == "prefix")
            {
                x.pieces.push_back(Board::__tokens_inv_dict["+"]);
                for (float i: real_x.pieces)
                {
                    x.pieces.push_back(i);
                }
                x.pieces.push_back(Board::__tokens_inv_dict["*"]);
                x.pieces.push_back(Board::__tokens_inv_dict["i"]);
                for (float i: imag_x.pieces)
                {
                    x.pieces.push_back(i);
                }
            }
            //Build total postfix expression: real i imag * +
            else //postfix
            {
                for (float i: real_x.pieces)
                {
                    x.pieces.push_back(i);
                }
                x.pieces.push_back(Board::__tokens_inv_dict["i"]);
                for (float i: imag_x.pieces)
                {
                    x.pieces.push_back(i);
                }
                x.pieces.push_back(Board::__tokens_inv_dict["*"]);
                x.pieces.push_back(Board::__tokens_inv_dict["+"]);
            }
                        
            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
            score = x.complete_status(false);
            
            //backprop reward `score` for real part
            for (auto& state_action: moveTracker_real)
            {
                Qsa_real[state_action.first][state_action.second] = std::max(Qsa_real[state_action.first][state_action.second], score);
            }
            //backprop reward `score` for imaginary part
            for (auto& state_action: moveTracker_imag)
            {
                Qsa_imag[state_action.first][state_action.second] = std::max(Qsa_imag[state_action.first][state_action.second], score);
            }
            
            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_expression = x._to_infix();
                orig_expression = x.expression();
                best_expr_result = x._to_infix(x.diffeq_result);
                orig_expr_result = x.expression(x.diffeq_result);
            }
            
            real_x.pieces.clear();
            imag_x.pieces.clear();
            x.pieces.clear();
            moveTracker_real.clear();
            moveTracker_imag.clear();
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
    std::cout << "Best diff result (original format) = " << orig_expr_result << '\n';
}

void RandomSearch(std::vector<float> (*diffeq)(Board&), const Eigen::MatrixXcf& data, int real_depth = 3, int imag_depth = 3, const std::string expression_type = "prefix", const std::string method = "LevenbergMarquardt", const int num_fit_iter = 1, const bool cache = true, const double time = 120.0 /*time to run the algorithm in seconds*/, unsigned int num_threads = 0, bool const_tokens = false, float isConstTol = 1e-1f, bool const_token = false)
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
    
    auto func = [&diffeq, &real_depth, &imag_depth, &expression_type, &method, &num_fit_iter, &data, &cache, &start_time, &time, &max_score, &sync_point, &best_expression, &orig_expression, &best_expr_result, &orig_expr_result, &const_tokens, &isConstTol, &const_token]()
    {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
        
        Board real_x(diffeq, false, real_depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token),
        imag_x(diffeq, false, imag_depth, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        Board x(diffeq, true, std::max(real_depth, imag_depth+1)+1, expression_type, method, num_fit_iter, data, false, cache, const_tokens, isConstTol, const_token);
        
        sync_point.arrive_and_wait();
        float score = 0.0f;
        std::vector<float> temp_legal_moves;
        size_t temp_sz;
        while (timeElapsedSince(start_time) < time)
        {
            //Build real part
            while ((score = real_x.complete_status()) == -1)
            {
                temp_legal_moves = real_x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                assert(temp_sz);
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                real_x.pieces.emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
            }
            assert(((real_x.expression_type == "prefix") ? real_x.getPNdepth(real_x.pieces) : real_x.getRPNdepth(real_x.pieces)).first == real_x.n);
            assert(((real_x.expression_type == "prefix") ? real_x.getPNdepth(real_x.pieces) : real_x.getRPNdepth(real_x.pieces)).second);

            //Build imaginary part
            while ((score = imag_x.complete_status()) == -1)
            {
                temp_legal_moves = imag_x.get_legal_moves(); //the legal moves
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                assert(temp_sz);
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                imag_x.pieces.emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
            }
            
            assert(((imag_x.expression_type == "prefix") ? imag_x.getPNdepth(imag_x.pieces) : imag_x.getRPNdepth(imag_x.pieces)).first == imag_x.n);
            assert(((imag_x.expression_type == "prefix") ? imag_x.getPNdepth(imag_x.pieces) : imag_x.getRPNdepth(imag_x.pieces)).second);
                        
            x.pieces.reserve(imag_x.pieces.size() + real_x.pieces.size() + 3);
            //Build total prefix expression: + real * i imag
            if (x.expression_type == "prefix")
            {
                x.pieces.push_back(Board::__tokens_inv_dict["+"]);
                for (float i: real_x.pieces)
                {
                    x.pieces.push_back(i);
                }
                x.pieces.push_back(Board::__tokens_inv_dict["*"]);
                x.pieces.push_back(Board::__tokens_inv_dict["i"]);
                for (float i: imag_x.pieces)
                {
                    x.pieces.push_back(i);
                }
            }
            //Build total postfix expression: real i imag * +
            else //postfix
            {
                for (float i: real_x.pieces)
                {
                    x.pieces.push_back(i);
                }
                x.pieces.push_back(Board::__tokens_inv_dict["i"]);
                for (float i: imag_x.pieces)
                {
                    x.pieces.push_back(i);
                }
                x.pieces.push_back(Board::__tokens_inv_dict["*"]);
                x.pieces.push_back(Board::__tokens_inv_dict["+"]);
            }
            
            assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
            score = x.complete_status(false);
            
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
            real_x.pieces.clear();
            imag_x.pieces.clear();
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
    std::cout << "Best diff result (original format) = " << orig_expr_result << '\n';
}

int main()
{
//    HembergBenchmarks(20 /*numIntervals*/, 120 /*time*/, 50 /*numRuns*/);
//    AIFeynmanBenchmarks(20 /*numIntervals*/, 120 /*time*/, 50 /*numRuns*/);
    
    /*
        Then, move the generated txt files to the directories Hemberg_Benchmarks and
        AIFeynman_Benchmarks and then run PlotData.py
    */
    
    auto data = createLinspaceMatrix(1000, 3, {-10.0f, -10.0f, 0.1f}, {10.0f, 10.0f, 20.0f});
    
    MCTS(NLS_2D /*differential equation to solve*/, data /*data used to solve differential equation*/, 2 /*fixed depth of generated real part of solutions*/, 2 /*fixed depth of generated imaginary part of solutions*/, "postfix" /*expression representation*/, "AsyncPSO" /*fit method if expression contains const tokens*/, 5 /*number of fit iterations*/, true /*cache*/, 1 /*time to run the algorithm in seconds*/, 0 /*num threads*/, false /*`const_tokens`: whether to include const tokens {0, 1, 2}*/, 1e-5 /*threshold for which solutions cannot be constant*/, false /*whether to include "const" token to be optimized, though `const_tokens` must be true as well*/);

 

    return 0;
}
//git push --set-upstream origin PrefixPostfixSymbolicDifferentiator

//g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifyComplexSR_Corrected PrefixPostfixMultiThreadDiffSimplifyComplexSR_Corrected.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -ffast-math -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifyComplexSR_Corrected PrefixPostfixMultiThreadDiffSimplifyComplexSR_Corrected.cpp -g -I/opt/homebrew/opt/eigen/include/eigen3 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//Solution? 
/// sech - x2 x0 / * 1 i cos x1
//(sech((x2 - x0)) / ((1 * i) / cos(x1)))

 
