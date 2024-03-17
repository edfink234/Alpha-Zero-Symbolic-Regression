#include <vector>
#include <array>
#include <iostream>
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
#include <unsupported/Eigen/NonLinearOptimization>
#include <boost/unordered/concurrent_flat_map.hpp>

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
};

float MSE(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
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
//    static std::unordered_map<std::string, std::pair<Eigen::VectorXf, bool>> inline expression_dict = {};
    static boost::concurrent_flat_map<std::string, std::pair<Eigen::VectorXf, bool>> inline expression_dict = {};
    static float inline best_loss = FLT_MAX;
    static std::atomic<float> inline fit_time = 0.0;

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
    
    static int inline action_size;
    static std::once_flag inline initialization_flag;  // Flag for std::call_once
    
    size_t reserve_amount;
    int num_fit_iter;
    
    std::vector<int> stack;
    int depth = 0, num_binary = 0, num_leaves = 0, idx = 0;
    static std::unordered_map<float, std::string> inline __tokens_dict; //Converts number to string
    static std::unordered_map<std::string, float> inline __tokens_inv_dict; //Converts string to number
    static std::unordered_map<bool, std::unordered_map<bool, std::unordered_map<bool, std::vector<float>>>> inline una_bin_leaf_legal_moves_dict;

    int n; //depth of RPN/PN tree
    std::string expression_string;
    static std::mutex inline thread_locker; //static because it needs to protect static members
    std::vector<float> pieces; // Create the empty expression list.
    bool is_primary;
    
    Board(bool primary = true, int n = 3, int numFitIter = 1, const Eigen::MatrixXf& theData = {}) : gen{rd()}, num_fit_iter{numFitIter}, is_primary{primary}
    {
        if (n > 30)
        {
            throw(std::runtime_error("Complexity cannot be larger than 30, sorry!"));
        }
        
        this->n = n;
        this->pieces = {};
        this->reserve_amount = 2*pow(2,this->n)-1;
        this->pieces.reserve(this->reserve_amount);
        
        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                Board::data = theData;
                
                Board::__num_features = data[0].size() - 1;
                Board::__input_vars.clear();
                Board::expression_dict.clear();
                Board::__input_vars.reserve(Board::__num_features);
                for (auto i = 0; i < Board::__num_features; i++)
                {
                    Board::__input_vars.push_back("x"+std::to_string(i));
                }
                Board::__unary_operators = {};//{"sin", "sqrt", "cos"};
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
    std::pair<int, bool> getPNdepth(const std::vector<float>& expression, size_t start = 0, size_t stop = 0, bool modify = false, bool binary = false, bool unary = false, bool leaf = false)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }
        
        if (stop == 0)
        {
            stop = expression.size();
        }

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
        
        return std::make_pair(this->depth - 1, this->num_leaves == this->num_binary + 1);
    }
    
    std::vector<float> get_legal_moves()
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
        return Board::una_bin_leaf_legal_moves_dict[una_allowed][bin_allowed][leaf_allowed];
        
    }

    Eigen::VectorXf expression_evaluator(const Eigen::VectorXf& params) const
    {
        std::stack<const Eigen::VectorXf> stack;
        size_t const_count = 0;
        for (int i = (pieces.size() - 1); i >= 0; i--)
        {
            std::string token = Board::__tokens_dict[pieces[i]];
            if (std::find(Board::__operators_float.begin(), Board::__operators_float.end(), pieces[i]) == Board::__operators_float.end()) // leaf
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
                    stack.push((left_operand.array() + right_operand.array()));
                }
                else if (token == "-")
                {
                    stack.push((left_operand.array() - right_operand.array()));
                }
                else if (token == "*")
                {
                    stack.push((left_operand.array() * right_operand.array()));
                }
                else if (token == "/")
                {
                    stack.push((left_operand.array() / right_operand.array()));
                }
                else if (token == "^")
                {
                    stack.push((left_operand.array().pow(right_operand.array())));
                }
            }
        }
        return stack.top();
    }
    
    /*
     x: parameter vector: (x_0, x_1, ..., x_{x.size()-1})
     g: gradient evaluated at x: (g_0(x_0), g_1(x_1), ..., g_{g.size()-1}(x_{x.size()-1}))
     */
    float operator()(Eigen::VectorXf& x, Eigen::VectorXf& grad)
    {
        grad = (this->expression_evaluator(x) - Board::data["y"]);
        return 0.f;
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
        lm.minimize(this->params);
        if (MSE(expression_evaluator(this->params), Board::data["y"]) < score_before)
        {
            improved = true;
        }
        Board::fit_time = Board::fit_time + (timeElapsedSince(start_time));
        return improved;
    }
    
    float fitFunctionToData()
    {
        float loss;
        if (this->params.size())
        {
            bool improved = true;
            improved = LevenbergMarquardt();
            Eigen::VectorXf temp_vec;
            
            if (improved)
            {
                if (Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.visit(this->expression_string, [&](auto& x)
                    {
                        x.second.first = this->params;
                    });
                }
                else
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, std::make_pair(this->params, false));
                }
            }
            Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
            {
                temp_vec = x.second.first;
            });
            
            loss = loss_func(expression_evaluator(temp_vec),Board::data["y"]);
            
            Board::expression_dict.insert_or_assign(this->expression_string, std::make_pair(temp_vec, false));
        }
        else
        {
            loss = loss_func(expression_evaluator(this->params),Board::data["y"]);
            Board::expression_dict.insert_or_assign(this->expression_string, std::make_pair(this->params, false));
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
    float complete_status()
    {
        if (this->pieces.empty())
        {
            this->stack.clear();
            this->idx = 0;
            this->depth = 0, this->num_binary = 0, this->num_leaves = 0;
        }
        auto [depth, complete] =  getPNdepth(pieces, 0 /*start*/, 0 /*stop*/, true /*modify*/);
        if (!complete || depth < this->n) //Expression not complete
        {
            return -1;
        }
        else
        {
            if (is_primary)
            {
                this->expression_string.clear();
                this->expression_string.reserve(8*pieces.size());
                
                for (float i: pieces){this->expression_string += std::to_string(i)+" ";}
                //https://www.boost.org/doc/libs/develop/libs/unordered/doc/html/unordered.html#concurrent_flat_map
                bool in_use;
                
                if (Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
                    {
                        in_use = x.second.second;
                    });
                }
                else
                {
                    Board::expression_dict.insert_or_assign(this->expression_string, std::make_pair(Eigen::VectorXf(), false));
                    in_use = false;
                }
                
                if (in_use)
                {
                    this->params.setOnes(this->__num_consts());
                    return loss_func(expression_evaluator(this->params),Board::data["y"]);
                }

                if (Board::expression_dict.contains(this->expression_string))
                {
                    Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
                    {
                        this->params = x.second.first;
                    });
                }
                else
                {
                    puts("is this ever called?");
                }
                
                if (!this->params.size())
                {
                    this->params.setOnes(this->__num_consts());
                    bool in_use;
                    if (Board::expression_dict.contains(this->expression_string))
                    {
                        Board::expression_dict.cvisit(this->expression_string, [&](const auto& x)
                        {
                            in_use = x.second.second;
                        });
                    }
                    else
                    {
                        in_use = false;
                    }
                    
                    if (in_use)
                    {
                        return loss_func(expression_evaluator(this->params),Board::data["y"]);
                    }

                    Board::expression_dict.insert_or_assign(this->expression_string, std::make_pair(this->params, in_use));
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
};

float exampleFunc(const Eigen::VectorXf& x)
{
    return 8.0f / (2.0f + x[0]*x[0] + x[1]*x[1]);
}

void RandomSearch(const Eigen::MatrixXf& data, const int depth = 3, const int num_fit_iter = 1, const double time = 120.0 /*time to run the algorithm in seconds*/, const int interval = 20 /*number of equally spaced points in time to sample the best score thus far*/, const char* filename = "" /*name of file to save the results to*/, const int num_runs = 50 /*number of runs*/, unsigned int num_threads = 0)
{
    std::map<int, std::vector<double>> scores; //unordered_map to store the scores
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
        
        auto func = [&depth, &num_fit_iter, &data, &start_time, &time, &max_score, &sync_point]()
        {
            std::random_device rand_dev;
            std::mt19937 thread_local generator(rand_dev()); // Mersenne Twister random number generator
            Board x(true, depth, num_fit_iter, data);
            sync_point.arrive_and_wait();
            float score = 0;
            std::vector<float> temp_legal_moves;
            size_t temp_sz;
            while (timeElapsedSince(start_time) < time)
            {
                while ((score = x.complete_status()) == -1)
                {
                    temp_legal_moves = x.get_legal_moves(); //the legal moves
                    temp_sz = temp_legal_moves.size(); //the number of legal moves
                    std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                    {
                        x.pieces.emplace_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
                    }
                }

                if (score > max_score)
                {
                    max_score = score;
                }
                x.pieces.clear();
            }
        };
        
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads[i] = std::thread(func);
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

int main()
{
//    boost::concurrent_flat_map<int, std::string> map;

      // Insert key-value pairs
//    map.insert_or_assign(1, "One");
//    map.insert_or_assign(2, "Two");
//    map.insert_or_assign(3, "Three");
//    std::string temp;
//    
//    size_t num_visits = map.visit(3, [&](const auto& x)
//    {
//        temp = x.second;
//    });
//    std::cout << temp << '\n';
//    std::cout << num_visits << '\n';

//
//      // Access values
//      std::cout << map[1] << std::endl;  // Output: "One"
//      std::cout << map[2] << std::endl;  // Output: "Two"
//      std::cout << map[3] << std::endl;  // Output: "Three"
//    
    RandomSearch(generateData(20, 3, exampleFunc, -3.0f, 3.0f), 4 /*fixed depth*/, 5, 4 /*time to run the algorithm in seconds*/, 2 /*number of equally spaced points in time to sample the best score thus far*/, "ExampleFuncPreRandomSearchMultiThread.txt" /*name of file to save the results to*/, 1 /*number of runs*/, 0 /*num threads*/);

    return 0;
}
