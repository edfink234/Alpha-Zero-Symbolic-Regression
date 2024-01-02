//TODO: Explore nanobind
#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include <future>         // std::async, std::future
#include <unordered_set>
#include <unordered_map>
#include <float.h>
#include <ctime>
#include <cstdlib>
#include <stack>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cfloat>
#include <cassert>
#include <span>
#include <pybind11/pybind11.h>
#include <Python.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <LBFGS.h>
#include <LBFGSB.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/AutoDiff>

using Clock = std::chrono::high_resolution_clock;

Eigen::MatrixXf generateData(int numRows, int numCols, float (*func)(const Eigen::VectorXf&))
{
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution;

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
        
        for (size_t i = 0; i < num_rows; i++)
        {
            this->features["y"](i) = this->data(i, this->num_columns - 1);
            this->rows[i] = data.row(i);
        }
        
        return *this;
    }
    
    const Eigen::VectorXf& operator[] (int i){return rows[i];}
    const Eigen::VectorXf& operator[] (const std::string& i)
    {
        return features[i];
    }
    const long numRows() {return num_rows;}
    const long numCols() {return num_columns;}

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
    static std::string inline best_expression = "";
    static std::unordered_map<std::string, std::pair<Eigen::VectorXf, int>> inline expression_dict = {};
    static float inline best_loss = FLT_MAX;
    static float inline fit_time = 0;
    
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
    Eigen::VectorXf* params; //store the parameters of the expression of the current episode after it's completed
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
    std::string expression_type;
    // Create the empty expression list.
    std::vector<float> pieces;
    bool visualize_exploration, is_primary;
    
    Board(bool primary = true, int n = 3, const std::string& expression_type = "prefix", std::string fitMethod = "PSO", int numFitIter = 1, std::string fitGradMethod = "naive_numerical", const Eigen::MatrixXf& theData = {}, bool visualize_exploration = false, bool cache = false) : is_primary{primary}, gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, fit_method{fitMethod}, num_fit_iter{numFitIter}, fit_grad_method{fitGradMethod}
    {
        if (n > 30)
        {
            throw(std::runtime_error("Complexity cannot be larger than 30, sorry!"));
        }
        
        if (is_primary)
        {
            Board::data = theData;
            Board::__num_features = data[0].size() - 1;
            
            Board::__input_vars.reserve(Board::__num_features);
            for (auto i = 0; i < Board::__num_features; i++)
            {
                Board::__input_vars.push_back("x"+std::to_string(i));
            }
            Board::__unary_operators = {"cos"};
            Board::__binary_operators = {"+", "-", "*"};
            for (std::string& i: Board::__unary_operators)
            {
                Board::__operators.push_back(i);
            }
            for (std::string& i: Board::__binary_operators)
            {
                Board::__operators.push_back(i);
            }
            Board::__other_tokens = {"const"};
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
            Board::__tokens_float.reserve(Board::action_size);
            for (int i = 1; i <= Board::action_size; ++i)
            {
                Board::__tokens_float.push_back(i);
            }
            int num_operators = Board::__operators.size();
            for (int i = 1; i <= num_operators; i++)
            {
                Board::__operators_float.push_back(i);
            }
            int num_unary_operators = Board::__unary_operators.size();
            for (int i = 1; i <= num_unary_operators; i++)
            {
                Board::__unary_operators_float.push_back(i);
            }
            for (int i = num_unary_operators + 1; i <= num_operators; i++)
            {
                Board::__binary_operators_float.push_back(i);
            }
            int ops_plus_features = num_operators + Board::__num_features;
            for (int i = num_operators + 1; i <= ops_plus_features; i++)
            {
                Board::__input_vars_float.push_back(i);
            }
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
        }
        
        this->n = n;
        this->expression_type = expression_type;
        this->pieces = {};
        this->reserve_amount = 2*pow(2,this->n)-1;
        this->visualize_exploration = visualize_exploration;
        this->pieces.reserve(reserve_amount);
        this->cache = cache;
    }
    
    float operator[](size_t index) const
    {
        if (index < Board::__tokens_float.size())
        {
            return Board::__tokens_float[index];
        }
        throw std::out_of_range("Index out of range");
    }
    
    int __num_binary_ops()
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

    int __num_unary_ops()
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

    int __num_leaves()
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
    
    int __num_consts()
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
    
    bool is_unary(float token)
    {
        return (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), token) != __unary_operators_float.end());
    }

    bool is_binary(float token)
    {
        return (std::find(__binary_operators_float.begin(), __binary_operators_float.end(), token) != __binary_operators_float.end());
    }
    
    bool is_operator(float token)
    {
        return (is_binary(token) || is_unary(token));
    }
    
    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77128902
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
        else
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
        else
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
                
                if (this->stack.size() == 1)
                {
                    this->idx++;
                    return std::make_pair(this->stack.back() - 1, true);
                }
                
                else
                {
                    int curr_max = std::max(this->stack.back(), *(this->stack.end()-2))+1;
                    for (int i = this->stack.size() - 2; i >= 1; i--)
                    {
                        curr_max = std::max(curr_max, this->stack[i-1])+1;
                    }
                    this->idx++;
                    return std::make_pair(curr_max - 1, false);
                }
            }
            
            return std::make_pair(this->stack.back() - 1, true);
        }
    }
    
    std::vector<float> get_legal_moves()
    {
        //TODO: Fix bug -> need to add operators before determining if the condition is true or not?
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
                
                pieces.push_back(Board::__binary_operators_float[0]);
                bin_allowed = (getPNdepth(pieces).first <= this->n);
                pieces[pieces.size() - 1] = Board::__unary_operators_float[0];
                una_allowed = (getPNdepth(pieces).first <= this->n);
                pieces[pieces.size() - 1] = Board::__input_vars_float[0];
                leaf_allowed = (!((num_leaves == num_binary + 1) || (getPNdepth(pieces).first < this->n && (num_leaves == num_binary))));
                pieces.pop_back();
                
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
                
                pieces.push_back(Board::__unary_operators_float[0]);
                una_allowed = ((num_leaves >= 1) && (getRPNdepth(pieces).first <= this->n));
                
                pieces[pieces.size() - 1] = Board::__input_vars_float[0];
                leaf_allowed = (getRPNdepth(pieces).first <= this->n);

                pieces.pop_back();
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
        int const_index = ((expression_type == "postfix") ? 0 : params->size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            if (std::find(Board::__other_tokens_float.begin(), Board::__other_tokens_float.end(), pieces[i]) != Board::__other_tokens_float.end())
            {
                temp += ((i!=sz) ? std::to_string((*params)(const_index)) + " " : std::to_string((*params)(const_index)));
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
                stack.push(((!show_consts) || (std::find(Board::__other_tokens.begin(), Board::__other_tokens.end(), token) == Board::__other_tokens.end())) ? token : std::to_string((*params)(const_counter++)));
            }
            else if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), pieces[i]) != Board::__unary_operators_float.end()) // Unary operator
            {
                std::string operand = std::move(stack.top());
                stack.pop();
                result = token + "(" + operand + ")";
                stack.push(result);
            }
            else // binary operator
            {
                std::string right_operand = std::move(stack.top());
                stack.pop();
                std::string left_operand = std::move(stack.top());
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
        return std::move(stack.top());
    }

    Eigen::VectorXf expression_evaluator(const Eigen::VectorXf& params)
    {
        std::stack<Eigen::VectorXf> stack;
        size_t const_count = 0;
        bool is_prefix = (expression_type == "prefix");
        for (int i = (is_prefix ? (pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            std::string token = Board::__tokens_dict[pieces[i]];

            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
            {
                if (token == "const")
                {
                    stack.push(Eigen::VectorXf::Ones(data.numRows())*params(const_count++));
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
                    Eigen::VectorXf temp = std::move(stack.top());
                    stack.pop();
                    stack.push(temp.array().cos());
                }
            }
            else // binary operator
            {
                Eigen::VectorXf left_operand = std::move(stack.top());
                stack.pop();
                Eigen::VectorXf right_operand = std::move(stack.top());
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
            }
        }
        return std::move(stack.top());
    }
    
    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> expression_evaluator(std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& parameters)
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
//                    std::cout << "\nparam[" << const_count << "] = " << params[const_count].value() << '\n';
                    stack.push(Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic>::Constant(data.numRows(), parameters[const_count++]));
                    
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
                    Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> temp = std::move(stack.top());
                    stack.pop();
                    stack.push(temp.array().cos());
                }
            }
            else // binary operator
            {
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> left_operand = std::move(stack.top());
                stack.pop();
                Eigen::Vector<Eigen::AutoDiffScalar<Eigen::VectorXf>, Eigen::Dynamic> right_operand = std::move(stack.top());
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
            }
        }
        return std::move(stack.top());
    }
    
    void AsyncPSO()
    {
        auto start_time = Clock::now();
        Eigen::VectorXf particle_positions(params->size()), x(params->size());
        Eigen::VectorXf v(params->size());
        float rp, rg;

        for (size_t i = 0; i < params->size(); i++)
        {
            particle_positions(i) = x(i) = pos_dist(gen);
            v(i) = vel_dist(gen);
        }

        float swarm_best_score = loss_func(expression_evaluator(*params),data["y"]);
        float fpi = loss_func(expression_evaluator(particle_positions),data["y"]);
        float temp, fxi;
        
        if (fpi > swarm_best_score)
        {
            *params = particle_positions;
            swarm_best_score = fpi;
        }
        
        auto UpdateParticle = [&](int i)
        {
            for (int j = 0; j < this->num_fit_iter; j++)
            {
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i) = K*(v(i) + phi_1*rp*(particle_positions(i) - x(i)) + phi_2*rg*((*params)(i) - x(i)));
                x(i) += v(i);
                
                fpi = loss_func(expression_evaluator(particle_positions),data["y"]); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions),data["y"]); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
                    (*params)(i) = particle_positions(i);
                    swarm_best_score = fpi;
                }
            }
        };
        
        std::vector<std::future<void>> particles;
        particles.reserve(params->size());
        for (int i = 0; i < params->size(); i++)
        {
            particles.push_back(std::async(std::launch::async | std::launch::deferred, UpdateParticle, i));
        }
        for (auto& i: particles)
        {
            i.get();
        }
        Board::fit_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9);
    }
    
    void PSO()
    {
        auto start_time = Clock::now();
        Eigen::VectorXf particle_positions(params->size()), x(params->size());
        Eigen::VectorXf v(params->size());
        float rp, rg;

        for (size_t i = 0; i < params->size(); i++)
        {
            particle_positions(i) = x(i) = pos_dist(gen);
            v(i) = vel_dist(gen);
        }

        float swarm_best_score = loss_func(expression_evaluator(*params),data["y"]);
        float fpi = loss_func(expression_evaluator(particle_positions),data["y"]);
        float temp, fxi;
        
        if (fpi > swarm_best_score)
        {
            *params = particle_positions;
            swarm_best_score = fpi;
        }
        for (int j = 0; j < this->num_fit_iter; j++)
        {
            for (unsigned short i = 0; i < params->size(); i++) //number of particles
            {
                rp = pos_dist(gen), rg = pos_dist(gen);
                v(i) = K*(v(i) + phi_1*rp*(particle_positions(i) - x(i)) + phi_2*rg*((*params)(i) - x(i)));
                x(i) += v(i);
                
                fpi = loss_func(expression_evaluator(particle_positions),data["y"]); //current score
                temp = particle_positions(i); //save old position of particle i
                particle_positions(i) = x(i); //update old position to new position
                fxi = loss_func(expression_evaluator(particle_positions),data["y"]); //calculate the score with the new position
                if (fxi < fpi) //if the new vector is worse:
                {
                    particle_positions(i) = temp; //reset particle_positions[i]
                }
                else if (fpi > swarm_best_score)
                {
//                    printf("Iteration %d: Changing param %d from %f to %f. Score from %f to %f\n", j, i, (*params)[i], particle_positions[i], swarm_best_score, fpi);
                    (*params)(i) = particle_positions(i);
                    swarm_best_score = fpi;
                }
            }
        }
        
        Board::fit_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9);
    }
    
    Eigen::AutoDiffScalar<Eigen::VectorXf> grad_func(std::vector<Eigen::AutoDiffScalar<Eigen::VectorXf>>& inputs)
    {
        return MSE(expression_evaluator(inputs), data["y"]);
    }
    
    /*
     x: parameter vector: (x_0, x_1, ..., x_{x.size()-1})
     g: gradient evaluated at x: (g_0(x_0), g_1(x_1), ..., g_{g.size()-1}(x_{x.size()-1}))
     */
    float operator()(Eigen::VectorXf& x, Eigen::VectorXf& grad)
    {
        if (this->fit_method == "LBFGS" || this->fit_method == "LBFGSB")
        {
            float mse = MSE(expression_evaluator(x), data["y"]);
            if (this->fit_grad_method == "naive_numerical")
            {
                float low_b, temp, fac;
                for (int i = 0; i < x.size(); i++) //finite differences wrt x evaluated at the current values x(i)
                {
                    //https://stackoverflow.com/a/38855586/18255427
                    temp = x(i);
                    x(i) -= 0.00001f;
                    low_b = MSE(expression_evaluator(x), data["y"]);
                    x(i) = temp + 0.00001f;
                    grad(i) = (MSE(expression_evaluator(x), data["y"]) - low_b) / 0.00002f ;
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
    
    void LBFGS()
    {
        auto start_time = Clock::now();
        LBFGSpp::LBFGSParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSSolver<float, LBFGSpp::LineSearchMoreThuente> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;
        
        Eigen::VectorXf eigenVec = *params;
        float mse = MSE(expression_evaluator(*params), data["y"]);
        try
        {
            solver.minimize((*this), eigenVec, fx);
        }
        catch (std::runtime_error& e){}
        
//        printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
//            printf("mse = %f -> fx = %f\n", mse, fx);
            *params = eigenVec;
        }
        Board::fit_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9);
    }
    
    void LBFGSB()
    {
        auto start_time = Clock::now();
        LBFGSpp::LBFGSBParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        //https://lbfgspp.statr.me/doc/LineSearchBacktracking_8h_source.html
        LBFGSpp::LBFGSBSolver<float> solver(param); //LineSearchBacktracking, LineSearchBracketing, LineSearchMoreThuente, LineSearchNocedalWright
        float fx;
        
        Eigen::VectorXf eigenVec = *params;
        float mse = MSE(expression_evaluator(*params), data["y"]);
        try
        {
            solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -std::numeric_limits<float>::infinity()), Eigen::VectorXf::Constant(eigenVec.size(), std::numeric_limits<float>::infinity()));
//            solver.minimize((*this), eigenVec, fx, Eigen::VectorXf::Constant(eigenVec.size(), -10.f), Eigen::VectorXf::Constant(eigenVec.size(), 10.f));
        }
        catch (std::runtime_error& e){}
        
//        printf("mse = %f -> fx = %f\n", mse, fx);
        if (fx < mse)
        {
//            printf("mse = %f -> fx = %f\n", mse, fx);
            *params = eigenVec;
        }
        Board::fit_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9);
    }
    
    int values()
    {
        return data.numRows();
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
    
    void LevenbergMarquardt()
    {
        auto start_time = Clock::now();
        Eigen::LevenbergMarquardt<decltype(*this), float> lm(*this);
        Eigen::VectorXf eigenVec = *params;
        lm.parameters.maxfev = this->num_fit_iter;
//        std::cout << "ftol (Cost function change) = " << lm.parameters.ftol << '\n';
//        std::cout << "xtol (Parameters change) = " << lm.parameters.xtol << '\n';

        lm.minimize(eigenVec);
        if (MSE(expression_evaluator(eigenVec), data["y"]) < MSE(expression_evaluator(*params), data["y"]))
        {
            *params = std::move(eigenVec);
        }
        
//        std::cout << "Iterations = " << lm.nfev << '\n';
        Board::fit_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9);
    }
    
    float fitFunctionToData()
    {
        if (params->size())
        {
            if (this->fit_method == "PSO")
            {
                PSO();
            }
            else if (this->fit_method == "AsyncPSO")
            {
                AsyncPSO();
            }
            else if (this->fit_method == "LBFGS")
            {
                LBFGS();
            }
            else if (this->fit_method == "LBFGSB")
            {
                LBFGSB();
            }
            else if (this->fit_method == "LevenbergMarquardt")
            {
                LevenbergMarquardt();
            }
        }
        
        return loss_func(expression_evaluator(*params),data["y"]);
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
                //TODO: call some plotting function, e.g. ROOT CERN plotting API, Matplotlib from the Python-C API, Plotly if we want a web application for this, etc. The plotting function could also have the fitted constants (rounded of course), but then this if statement would need to be moved down to below the fitFunctionToData call in this `complete_status` method.
            }
            
            if (is_primary)
            {
                std::string expression_string;
                expression_string.reserve(8*pieces.size());
                for (float i: pieces){expression_string += std::to_string(i)+" ";}
                Board::expression_dict[expression_string].second++;
                
                this->params = &Board::expression_dict[expression_string].first;
                if (!(this->params->size()))
                {
                    this->params->resize(__num_consts());
                    this->params->setOnes();
                }

                return fitFunctionToData();
            }
            return 0.0f;
        }
    }
    const Eigen::VectorXf& operator[] (int i){return data[i];}
    const Eigen::VectorXf& operator[] (const std::string& i)
    {
        return data[i];
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
};

//Data Board::data;

// 2.5382*cos(x_3) + x_0^2 - 0.5
// postfix = "const x3 cos * x0 x0 * const - +"
// prefix = "+ * const cos x3 - * x0 x0 const"
float exampleFunc(const Eigen::VectorXf& x)
{
    return 2.5382*cos(x[3]) + (x[0]*x[0]) - 0.5f;
//    return 5*cos(x[1]+x[3])+x[4];
}

//https://dl.acm.org/doi/pdf/10.1145/3449639.3459345?casa_token=Np-_TMqxeJEAAAAA:8u-d6UyINV6Ex02kG9LthsQHAXMh2oxx3M4FG8ioP0hGgstIW45X8b709XOuaif5D_DVOm_FwFo
//https://core.ac.uk/download/pdf/6651886.pdf
void SimulatedAnnealing(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true)
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
    Board secondary(false, 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache); //For perturbations
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
    float score = 0.0f, max_score = 0.0f, check_point_score = 0.0f;
    
    std::vector<float> current;
    std::vector<std::pair<int, int>> sub_exprs;
    std::vector<float> temp_legal_moves;
    std::uniform_int_distribution<int> rand_depth_dist(0, x.n);
    size_t temp_sz;
    std::string expression, orig_expression, best_expression;
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
        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
        if ((score > max_score) || (x.pos_dist(generator) < P(score-max_score)))
        {
            current = x.pieces; //update current expression
            expression = x._to_infix();
            orig_expression = x.expression();
            max_score = score;
            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
            std::cout << "Best expression = " << expression << '\n';
            std::cout << "Best expression (original format) = " << orig_expression << '\n';
            best_expression = std::move(expression);
        }
        else
        {
            x.pieces = current; //reset perturbed state to current state
        }
        T = r*T;
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
        
        assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces) : secondary.getRPNdepth(secondary.pieces)).first == secondary.n);
        assert(((secondary.expression_type == "prefix") ? secondary.getPNdepth(secondary.pieces) : secondary.getRPNdepth(secondary.pieces)).second);
        
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
            x.pieces.insert(start, secondary.pieces.begin(), secondary.pieces.end()); //could be a move operation: secondary.pieces doesn't need to be in a defined state after this.
        }
        
        //Step 5: Evaluate the new mutated `x.pieces` and update score if needed
        score = x.complete_status(false);
        updateScore(pow(ratio, 1.0f/(i+1)));
        
    };
    
    for (int i = 0; max_score < stop; i++)
    {
        if (i && (i%50000 == 0))
        {
            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
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
    std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
}

//https://arxiv.org/abs/2310.06609
void GP(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true)
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
    Board secondary_one(false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache), secondary_two(false, (depth > 0) ? depth-1 : 0, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache); //For crossover and mutations
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
    float score = 0.0f, max_score = 0.0f, mut_prob = 0.8f, cross_prob = 0.2f, rand_mut_cross;
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
    std::string expression, orig_expression, best_expression;
    
    auto updateScore = [&]()
    {
        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).first == x.n);
        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.pieces) : x.getRPNdepth(x.pieces)).second);
        if (score > max_score)
        {
            expression = x._to_infix();
            orig_expression = x.expression();
            max_score = score;
            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
            std::cout << "Best expression = " << expression << '\n';
            std::cout << "Best expression (original format) = " << orig_expression << '\n';
            best_expression = std::move(expression);
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
        
        assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).first == secondary_one.n);
        assert(((secondary_one.expression_type == "prefix") ? secondary_one.getPNdepth(secondary_one.pieces) : secondary_one.getRPNdepth(secondary_one.pieces)).second);

        
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
        //in `individual_1` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_1`.
        secondary_one.get_indices(sub_exprs_1, individual_1.first);
        
        //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
        //in `individual_2` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_2`.
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
    
    for (int ngen = 0; max_score < stop; ngen++)
    {
        if (ngen && (ngen%5 == 0))
        {
            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
        }
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
}

void PSO(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true)
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
    
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
    float score = 0, max_score = 0, check_point_score = 0;
    std::vector<float> temp_legal_moves;
    size_t temp_sz;
    std::string expression, orig_expression, best_expression;
    
    auto trueMod = [](int N, int M)
    {
        return ((N % M) + M) % M;
    };
    
    /*
     For this setup, we don't know a-priori the number of particles, so we generate them and their corresponding velocities as needed
     */
    std::vector<float> particle_positions, best_positions, v, curr_positions;
    particle_positions.reserve(x.reserve_amount); //stores record of all current particle position indices
    best_positions.reserve(x.reserve_amount); //indices corresponding to best pieces
    curr_positions.reserve(x.reserve_amount); //indices corresponding to x.pieces
    v.reserve(x.reserve_amount); //stores record of all current particle velocities
    float rp, rg, new_pos, new_v, noise, c = 0.0f;
    int c_count = 0;
    std::unordered_map<float, std::unordered_map<int, int>> Nsa;
    std::unordered_map<float, std::unordered_map<int, float>> Psa;
    std::unordered_map<int, float> p_i_vals, p_i;
    
    /*
     Idea: In this implementation of PSO,
     
     The traditional PSO initializes the particle positions to be between 0 and 1. However, in this application,
     the particle positions are discrete values and any of the legal integer tokens (moves). The
     velocities are continuous-valued and perturb the postions, which are subsequently constrained by rounding to
     the nearest whole number then taking the modulo w.r.t. the # of allowed legal moves.
     */
    
    for (int iter = 0; (score < stop/* && Board::expression_dict.size() <= 2000000*/); iter++)
    {
        if (iter && (iter%50000 == 0))
        {
            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
            std::cout << "check_point_score = " << check_point_score
            << ", max_score = " << max_score << ", c = " << c << '\n';
            if (check_point_score == max_score)
            {
                c_count++;
                std::uniform_real_distribution<float> temp(-c_count, c_count);
                std::cout << "c: " << c << " -> ";
                c = temp(generator);
                std::cout << c << '\n';
            }
            else
            {
                std::cout << "c: " << c << " -> ";
                c = 0.0f; //if new best found, reset c and try to exploit the new best
                c_count = 0;
                std::cout << c << '\n';
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
            expression = x._to_infix();
            orig_expression = x.expression();
            max_score = score;
            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
            std::cout << "Best expression = " << expression << '\n';
            std::cout << "Best expression (original format) = " << orig_expression << '\n';
            best_expression = std::move(expression);
        }
        x.pieces.clear();
        curr_positions.clear();
    }
}

//https://arxiv.org/abs/2205.13134
void MCTS(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true)
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
    float score = 0.0f, max_score = 0.0f, check_point_score = 0.0f, UCT, best_act, UCT_best;
    std::vector<float> temp_legal_moves;
    std::unordered_map<std::string, std::unordered_map<float, float>> Qsa, Nsa;
    std::string state;
    std::string expression, orig_expression, best_expression;
    std::unordered_map<std::string, float> Ns;
    float c = 1.4f; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf
    std::vector<std::pair<std::string, float>> moveTracker;
    moveTracker.reserve(x.reserve_amount);
    temp_legal_moves.reserve(x.reserve_amount);
    state.reserve(2*x.reserve_amount);
    double str_convert_time = 0.0;
    
    auto getString  = [&]()
    {
        if (!x.pieces.empty())
            state += std::to_string(x.pieces[x.pieces.size()-1]) + " ";
    };
    
    for (int i = 0; (score < stop/* && Board::expression_dict.size() <= 2000000*/); i++)
    {
        if (i && (i%50000 == 0))
        {
            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
            std::cout << "check_point_score = " << check_point_score
            << ", max_score = " << max_score << ", c = " << c << '\n';
            if (check_point_score == max_score)
            {
                std::cout << "c: " << c << " -> ";
                c += 1.4;
                std::cout << c << '\n';
            }
            else
            {
                std::cout << "c: " << c << " -> ";
                c = 1.4; //if new best found, reset c and try to exploit the new best
                std::cout << c << '\n';
            }
            check_point_score = max_score;
        }
        state.clear();
        while ((score = x.complete_status()) == -1)
        {
            temp_legal_moves = x.get_legal_moves();
            auto start_time = Clock::now();
            getString();
            str_convert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
            UCT = 0.0f;
            UCT_best = -FLT_MAX;
            best_act = -1.0f;
            
            for (auto& a : temp_legal_moves)
            {
                if (Nsa[state].count(a))
                {
                    UCT = Qsa[state][a] + c*sqrt(log(Ns[state])/Nsa[state][a]);
                }
                else
                {
                    UCT = FLT_MAX; //highest -> explore it
                }
                if (UCT > UCT_best)
                {
                    best_act = a;
                    UCT_best = UCT;
                }
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
            expression = x._to_infix();
            orig_expression = x.expression();
            max_score = score;
            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
            std::cout << "Best expression = " << expression << '\n';
            std::cout << "Best expression (original format) = " << orig_expression << '\n';
            std::cout << "str_convert_time (s) = " << str_convert_time << '\n';
            std::cout << "Board::fit_time (s) = " << Board::fit_time << '\n';
            best_expression = std::move(expression);
        }
        x.pieces.clear();
        moveTracker.clear();
    }
}

void RandomSearch(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical", bool cache = true)
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false, cache);
    std::cout << Board::data["y"] << '\n';
    std::cout << x.fit_method << '\n';
    std::cout << Board::data << '\n';
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev()); // Mersenne Twister random number generator
    float score = 0, max_score = 0;
    std::vector<float> temp_legal_moves;
    size_t temp_sz;
    std::string expression, orig_expression, best_expression;
    
//    std::ofstream out(x.expression_type == "prefix" ? "PN_expressions.txt" : "RPN_expressions.txt");
//    std::cout << "stop = " << stop << '\n';
    for (int i = 0; (/*score < stop && */Board::expression_dict.size() <= 100000); i++)
    {
//        std::cout << "iter " << i << '\n';
        while ((score = x.complete_status()) == -1)
        {
            temp_legal_moves = x.get_legal_moves(); //the legal moves
//            std::cout << "temp_legal_moves.size() = " << temp_legal_moves.size() << '\n';
//            printf("temp_legal_moves: ");
//            for (float i : temp_legal_moves) { std::cout << Board::__tokens_dict[i] << ' '; }puts("");
            temp_sz = temp_legal_moves.size(); //the number of legal moves
            std::uniform_int_distribution<int> distribution(0, temp_sz - 1);
 // A random integer generator which generates an index corresponding to an allowed move

            x.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move

        }

//        out << "Iteration " << i << ": Original expression = " << x.expression() << ", Infix Expression = " << expression << '\n';
        if (score > max_score)
        {
            
            expression = x._to_infix();
            orig_expression = x.expression();
            max_score = score;
            std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
            std::cout << "Best expression = " << expression << '\n';
            std::cout << "Best expression (original format) = " << orig_expression << '\n';
            std::cout << "Unique expressions = " << Board::expression_dict.size() << '\n';
            best_expression = std::move(expression);
        }
        x.pieces.clear();
    }
//    out.close();
    std::cout << "\nUnique expressions = " << Board::expression_dict.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
}

PyObject* convertVectorToPythonList(const std::vector<float>& inputVector)
{
    PyObject* pyList = PyTuple_New(inputVector.size());

    for (size_t i = 0; i < inputVector.size(); ++i)
    {
        PyObject* pyFloat = PyFloat_FromDouble(static_cast<double>(inputVector[i]));
        PyTuple_SetItem(pyList, i, pyFloat);
    }

    return pyList;
}

int main() {
    
    //This comment block is testing the Python-C API
//    Py_Initialize();
//    PyObject* pName = PyUnicode_DecodeFSDefault("scipy");
//    PyObject* pModule = PyImport_Import(pName);
//    Py_XDECREF(pName);
//    std::cout << std::boolalpha << (pModule == NULL) << '\n';
//    PyObject* pFunc = PyObject_GetAttrString(pModule, "optimize.curve_fit");
////    PyObject* pArgs = Py_BuildValue("ff", 1.0f, 1.0f);
//    std::vector<float> myVector = {1.0f, 2.0f, 3.0f};
//    PyObject* pArgs = convertVectorToPythonList(myVector);
//    PyObject* pStr = PyObject_Str(pArgs);
//    const char* cstr = PyUnicode_AsUTF8(pStr);
//    puts(cstr);
//    exit(1);
    Eigen::MatrixXf data = generateData(100, 6, exampleFunc);
//    std::cout << data << "\n\n";
    auto start_time = Clock::now();
    RandomSearch(data, 10, "prefix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/);
//    MCTS(data, 3, "postfix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical", true);
//    PSO(data, 3, "postfix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical", true);
//    GP(data, 29, "postfix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/);
//    SimulatedAnnealing(data, 3, "postfix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical", true /*cache*/);
    auto end_time = Clock::now();
    std::cout << "Time difference = "
          << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()/1e9 << " seconds" << '\n';

    return 0;
}

//git push --set-upstream origin prefix_and_postfix_cpp_implementation




