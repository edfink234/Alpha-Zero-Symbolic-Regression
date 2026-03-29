#include <vector>
#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <algorithm>
#include <future>         // std::async, std::future
#include <unordered_map>
#include <unordered_set>
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
#include <boost/unordered/concurrent_flat_set.hpp>
#include <boost/spirit/include/qi.hpp> //For fast string-to-double conversion!
#include "MLP_Vec.h"
#define RANDOM_SEED 42 //fixed random seed!

using Clock = std::chrono::high_resolution_clock;

//Returns the number of seconds since `start_time`
template <typename T>
double timeElapsedSince(T start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
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

Eigen::MatrixXf generateData(int numRows, int numCols, float (*func)(const Eigen::VectorXf&), float min = -3.0f, float max = 3.0f)
{
    // Initialize random number generator
    #ifndef RANDOM_SEED
        std::random_device rd;
        std::mt19937 thread_local gen(rd());
    #else
        std::mt19937 thread_local gen(RANDOM_SEED);
    #endif
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

bool parse_double_spirit(const std::string& s, double& out)
{
    namespace qi   = boost::spirit::qi;
    namespace ascii= boost::spirit::ascii;

    auto f = s.begin(), l = s.end();
    // Skip leading/trailing ASCII whitespace; require full consumption (eoi).
    // qi::double_ already yields ±inf/NaN where appropriate.
    bool ok = qi::phrase_parse(f, l, qi::double_ >> qi::eoi, ascii::space, out);
    return ok; // f==l guaranteed by eoi
}

double Stod(const std::string& param)
{
    double val;
    parse_double_spirit(param, val);
    return val;
}

bool checkEqual(const std::string &str1, const std::string &str2)
{
    if (!isdouble(str1))
    {
        return false;
    }
    double val1; parse_double_spirit(str1, val1);
    double val2 = 0.0;
    if (str2 == "0")
    {
        val2 = 0.0;
    }
    else if (str2 == "1")
    {
        val2 = 1.0;
    }
    else if (str2 == "-1")
    {
        val2 = -1.0;
    }
    return (val1 == val2);
}

int trueMod(int N, int M)
{
    return ((N % M) + M) % M;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    for (const auto& i: vec)
    {
        os << i << ' ';
    }
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::deque<T>& vec)
{
    for (const auto& i: vec)
    {
        os << i << ' ';
    }
    return os;
}

template <typename T>
std::string vec_to_str(const std::vector<T>& vec)
{
    std::stringstream ss;
    for (const auto& i: vec)
    {
        ss << i << ' ';
    }
    return ss.str();
}

class Data
{
    Eigen::MatrixXf data;
    std::unordered_map<std::string, Eigen::VectorXf> features;
    long num_columns, num_rows;
    
public:
    
    Data() = default; //so we can have a static Data attribute
    std::vector<Eigen::VectorXf> labels;
    std::vector<Eigen::VectorXf> rows;
    std::vector<Eigen::VectorXf> feature_matrix;
    // Assignment operator
    Data& operator=(const Eigen::MatrixXf& theData)
    {
        this->data = theData;
        this->num_columns = data.cols();
        this->num_rows = data.rows();
        std::string idx;

        for (size_t i = 0; i < this->num_columns - 1; i++) //for each column
        {
            idx = "x"+std::to_string(i);
            this->features[idx] = Eigen::VectorXf(this->num_rows); //create a key-value pair of the form ("x{i}": Vector(num_rows))
            for (size_t j = 0; j < this->num_rows; j++)
            {
                this->features[idx](j) = this->data(j,i);
            }
            this->feature_matrix.push_back(this->features[idx]);
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
    void print()
    {
        for (size_t i = 0; i < this->num_columns - 1; i++)
        {
            std::cout << "x" << i << '\t';
        }
        std::cout << "y\n";
        std::cout << data << '\n';
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

float loss_func(const Eigen::VectorXf& actual, const Eigen::VectorXf& predicted)
{
    return (1.0f/(1.0f+MSE(actual, predicted)));
}

struct Board
{
    static boost::concurrent_flat_set<std::string> inline expression_set = {}; //TODO: change to concurrent_set
    static float inline best_loss = FLT_MAX;
    static std::atomic<float> inline fit_time = 0.0;
    
    static constexpr float K = 0.0884956f;
    static constexpr float phi_1 = 2.8f;
    static constexpr float phi_2 = 1.3f;
    static int inline __num_features;
    static std::vector<std::string> inline __input_vars;
    static std::vector<std::string> inline __unary_operators;
    static std::vector<std::string> inline __binary_operators;
    static std::unordered_set<std::string> inline __unary_operators_uset;
    static std::unordered_set<std::string> inline __binary_operators_uset;
    static std::vector<std::string> inline __operators;
    static std::vector<std::string> inline __other_tokens;
    static std::vector<std::string> inline __tokens;
    static Data inline data;
    static std::mutex inline thread_locker;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> vel_dist, pos_dist;
    
    static int inline action_size;
    static std::once_flag inline initialization_flag;  // Flag for std::call_once
    size_t reserve_amount;
    bool cache;
    std::vector<int> stack;
    int depth = 0, num_binary = 0, num_leaves = 0, idx = 0;
    static std::unordered_map<bool, std::unordered_map<bool, std::unordered_map<bool, std::vector<std::string>>>> inline una_bin_leaf_legal_moves_dict;
    std::vector<int> simplify_grasp;

    int n; //depth of RPN/PN tree
    std::string expression_type, expression_string;
    bool visualize_exploration, is_primary;
    MultiLayerPerceptron srnn;
    const unsigned long epochs;
    
//    Board(bool primary = true /*only relevant for GP and SimulatedAnnealing*/, int n = 3, const std::string& expression_type = "prefix", const Eigen::MatrixXf& theData = {}, bool visualize_exploration = false, bool cache = false, std::vector<int> layers = {}, std::deque<std::string> layer_types = {}, const unsigned long num_epochs = 1000, float bias = 1.0f, float eta = 0.5f, float theta = 0.01f, float gamma = 0.9f, float epsilon = 0.1f, float beta_1 = 0.9f, float beta_2 = 0.999f, float lambda = 0.01f /*weight decay AdamW*/)
    Board(int n, const std::string& expression_type, bool cache) : gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, is_primary{false}, srnn{}, epochs{0}
    {
        this->n = n;
        this->expression_type = expression_type;
        srnn.pieces = {};
        this->reserve_amount = 2*std::pow(2,this->n)-1;
        srnn.pieces.reserve(this->reserve_amount);
        this->cache = cache;
    }
    
    Board(int n, const std::string& expression_type, const Eigen::MatrixXf& theData, bool visualize_exploration, bool cache, std::vector<int> layers, std::deque<std::string> layer_types, const unsigned long num_epochs, float bias, float eta, float theta, float gamma, float epsilon, float beta_1, float beta_2, float lambda)
        : gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, is_primary{true}, srnn{layers, layer_types, bias, eta, theta, gamma, "SR", expression_type, epsilon, beta_1, beta_2, lambda}, epochs{num_epochs}
    {
        if (n > 30)
        {
            throw(std::runtime_error("Complexity cannot be larger than 30, sorry!"));
        }
        //puts("here at 211");
        this->n = n; //depth of expression
        this->expression_type = expression_type;
        srnn.pieces = {};
        this->visualize_exploration = visualize_exploration;
        this->reserve_amount = 2*std::pow(2,this->n)-1;
        srnn.pieces.reserve(this->reserve_amount);
        this->cache = cache;
        //puts("here at 219");
        if (is_primary)
        {
            std::call_once(initialization_flag, [&]()
            {
                //puts("here at 224");
                Board::data = theData;
                std::cout << "Board::data = " << Board::data << '\n';
                Board::__num_features = data[0].size() - 1;
                Board::__input_vars.clear();
                Board::expression_set.clear();
                Board::__input_vars = {"w_k", "eta", "theta", "gamma", "epsilon", "beta_1", "beta_2", "d_ij", "value", "d_ij_nest", "velocity_k", "gradient_k", "g_t_k", "expt_grad_squared_k", "delta_w_t_k", "expt_weight_squared_k", "delta_w_t_k_ada_delta", "m_t_k", "v_t_k", "m_t_k_hat", "v_t_k_hat", "t"};
                Board::__unary_operators = srnn.__unary_operators; //{"cos", "exp", "sqrt", "sin", "asin", "ln", "tanh", "acos", "~"};
                Board::__binary_operators = srnn.__binary_operators;//{"+", "-", "*", "/", "^"};
                std::copy(Board::__unary_operators.begin(), Board::__unary_operators.end(), std::inserter(Board::__unary_operators_uset, Board::__unary_operators_uset.end()));
                std::copy(Board::__binary_operators.begin(), Board::__binary_operators.end(), std::inserter(Board::__binary_operators_uset, Board::__binary_operators_uset.end()));
                puts("Board::__unary_operators_uset");
                for (const std::string& i: Board::__unary_operators_uset) {std::cout << i << ' ';}puts("");
                puts("Board::__binary_operators_uset");
                for (const std::string& i: Board::__binary_operators_uset) {std::cout << i << ' ';}puts("");
                Board::__operators.clear();
                Board::__operators = srnn.__operators;
//                for (std::string& i: Board::__unary_operators)
//                {
//                    Board::__operators.push_back(i);
//                }
//                for (std::string& i: Board::__binary_operators)
//                {
//                    Board::__operators.push_back(i);
//                }
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
//                puts("here at 257");
                Board::una_bin_leaf_legal_moves_dict[true][true][true] = Board::__tokens;
                Board::una_bin_leaf_legal_moves_dict[true][true][false] = Board::__operators;
                Board::una_bin_leaf_legal_moves_dict[true][false][true] = Board::__unary_operators; //1
                Board::una_bin_leaf_legal_moves_dict[true][false][false] = Board::__unary_operators;
                Board::una_bin_leaf_legal_moves_dict[false][true][true] = Board::__binary_operators; //2
                Board::una_bin_leaf_legal_moves_dict[false][true][false] = Board::__binary_operators;
                
                for (const std::string &i: Board::__input_vars)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                    Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                    Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                }
                for (const std::string& i: Board::__other_tokens)
                {
                    Board::una_bin_leaf_legal_moves_dict[true][false][true].push_back(i); //1
                    Board::una_bin_leaf_legal_moves_dict[false][true][true].push_back(i); //2
                    Board::una_bin_leaf_legal_moves_dict[false][false][true].push_back(i); //3
                }
                std::cout << "Board::__unary_operators.size() = " << Board::__unary_operators.size() << '\n';
                std::cout << "Board::__binary_operators.size() = " << Board::__binary_operators.size() << '\n';
                std::cout << "Board::__tokens.size() = " << Board::__tokens.size() << '\n';
                for (const std::string& i: Board::__tokens) {std::cout << i << ' ';}puts("");

            });
        }
//        exit(1);
    }
    
    std::string operator[](size_t index) const
    {
        if (index < Board::__tokens.size())
        {
            return Board::__tokens[index];
        }
        throw std::out_of_range("Index out of range");
    }
    
    int __num_binary_ops() const
    {
        int count = 0;
        for (const std::string& token : srnn.pieces)
        {
            if (std::find(Board::__binary_operators.begin(), Board::__binary_operators.end(), token) != Board::__binary_operators.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_unary_ops() const
    {
        int count = 0;
        for (const std::string& token : srnn.pieces)
        {
            if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), token) != Board::__unary_operators.end())
            {
                count++;
            }
        }
        return count;
    }

    int __num_leaves() const
    {
        int count = 0;

        for (const std::string& token : srnn.pieces)
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

        for (const std::string& token : srnn.pieces)
        {
            if (token.substr(0,5) == "const")
            {
                count++;
            }
        }
        return count;
    }
    
    bool is_unary(const std::string& token) const
    {
        return (Board::__unary_operators_uset.find(token) != Board::__unary_operators_uset.end());
    }
    
    bool is_binary(const std::string& token) const
    {
        return (Board::__binary_operators_uset.find(token) != Board::__binary_operators_uset.end());
    }
    
    bool is_operator(const std::string& token) const
    {
        return (is_binary(token) || is_unary(token));
    }
    
    bool is_const(const std::string& token) const
    {
        return ((!is_unary(token)) && (!is_binary(token)));
    }
    
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

    //((1+5) * 1) -> 1 5 1 * +, + 1 * 5 1 -> traverse into a tree data-structure -> apply simplification algorithm
    //-> traverse tree again to get prefix/postfix simplified expression -> convert to infix
    //prefix/postifx -> simplified prefix/postfix
    void graspSimplifyPrefixHelper(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp, std::vector<std::string>& new_expression, bool setGRvar = false)
    {
        if (!setGRvar)
        {
            grasp.clear();
            setPrefixGR(expression, grasp);
        }
    //    print_container(expression, low, up);
    //    print_container(new_expression, 0, new_expression.size() - 1);
        if (expression[low] == "+" || expression[low] == "-") // +/- x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]);
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true);
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true);
            int second_arg_idx_high = new_expression.size();
            int step;

            if (checkEqual(new_expression[first_arg_idx_high], "0")) //+/- x 0 -> x
            {
                //puts("hi 177");
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //remove +/- operator at beginning
            }

            else if (checkEqual(new_expression[first_arg_idx_low], "0"))
            {
                if (expression[low] == "+") //+ 0 y -> y
                {
                    //puts("hi 176");
                    new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + first_arg_idx_high); //remove '+' and '0'
                }
                else //- 0 y -> ~ y
                {
                    //puts("hi 184");
                    new_expression[op_idx] = "~";
                    new_expression.erase(new_expression.begin() + first_arg_idx_low); //'0'
                }
            }
            
            else if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) //+/- nan x -> nan, +/- x nan -> nan, +/- nan nan -> nan
            {
                //puts("hi 273");
                assert(new_expression[op_idx] == expression[low]);
                new_expression[op_idx] = "nan"; //change "+/-" to "nan";
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
            }
            
            else if ((expression[low] == "-") && ((step = (second_arg_idx_high - first_arg_idx_high)) == (first_arg_idx_high - first_arg_idx_low)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) //- x x
            {
                //puts("hi 221");
                assert(new_expression[op_idx] == expression[low]);
                new_expression[op_idx] = "0"; //change "-" to "0";
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
            }
        }
        else if (expression[low] == "*") //* x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); //*
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); //* x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); //* x y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            if ((new_expression[first_arg_idx_high] == "nan") || (new_expression[first_arg_idx_low] == "nan")) //* nan x -> nan, * x nan -> nan, * nan nan -> nan
            {
                //puts("hi 300");
                new_expression[op_idx] = "nan"; //change '*' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "0")) //* x 0 -> 0 (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to * x 0, which is 0)
            {
                //puts("hi 239");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) //* 0 x -> 0
            {
                //puts("hi 245");
                new_expression[op_idx] = "0"; //change '*' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "1")) //* x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to * x 1, which is x)
            {
                //puts("hi 251");
                //erase the '1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //erase the '*'
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) //* 1 x -> x
            {
                //puts("hi 265");
                new_expression.erase(new_expression.begin() + op_idx, new_expression.begin() + op_idx + 2); //erase the '*' and the '1'
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "-1")) //* x -1 -> ~ x (because, since prefix operators come at the beginning, if the beginning of the second argument of '*' is -1, then the whole second argument MUST be -1, therefore the expression reduces to * x -1, which is ~ x)
            {
    //            puts("hi 382");
                //erase the '-1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression[op_idx] = "~"; //change the '*' to a '~'
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "-1")) //* -1 x -> ~ x
            {
    //            puts("hi 396");
                new_expression[op_idx] = "~"; //change the '*' to a '~'
                new_expression.erase(new_expression.begin() + op_idx + 1); //erase the '-1'
            }
        }
        else if (expression[low] == "/") // / x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // /
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // / x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // / x y
            int second_arg_idx_high = new_expression.size();
            int step;
            
            if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) // / nan x -> nan, / x nan -> nan, / nan nan -> nan
            {
                //puts("hi 350");
                new_expression[op_idx] = "nan"; //change '/' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((checkEqual(new_expression[first_arg_idx_low], "0")) && (checkEqual(new_expression[first_arg_idx_high], "0"))) // / 0 0 -> nan
            {
                //puts("hi 290");
                new_expression[op_idx] = "nan"; //change '/' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "0")) // / x 0 -> nan (for now, because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to / x 0, which is, for now, assumed to be nan for simplicity)
            {
                //puts("hi 282");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[op_idx] = "nan";//(new_expression[first_arg_idx_low] != "~") ? "inf": "-inf"; //change '/' to 'inf' or '-inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // / 0 x -> 0
            {
                //puts("hi 295");
                new_expression[op_idx] = "0"; //change '/' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "1")) // / x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to / x 1, which is x)
            {
                //puts("hi 301");
                //erase the '1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //erase the '/'
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "-1")) // / x -1 -> ~ x (because, since prefix operators come at the beginning, if the beginning of the second argument of '/' is -1, then the whole second argument MUST be -1, therefore the expression reduces to / x -1, which is -x)
            {
                //puts("hi 454");
                //erase the '-1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression[op_idx] = "~"; //change the '/' to a '~'
            }
            else if ((expression[low] == "/") && ((step = (second_arg_idx_high - first_arg_idx_high)) == (first_arg_idx_high - first_arg_idx_low)) && (areExpressionRangesEqual(first_arg_idx_low, first_arg_idx_high, step, new_expression))) // / x x
            {
                //puts("hi 315");
                assert(new_expression[op_idx] == expression[low]);
                new_expression[op_idx] = "1"; //change "-" to "1";
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.begin() + second_arg_idx_high);
            }

            //TODO:
                /*
                x*y       y
                ---  -->  -
                x*z       z
                */
        }
        else if (expression[low] == "^") // ^ x y
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // ^
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ^ x
            int first_arg_idx_high = new_expression.size();
            graspSimplifyPrefixHelper(expression, temp+1, temp+1+grasp[temp+1], grasp, new_expression, true); // ^ x y
            //int second_arg_idx_high = new_expression.size();
            //int step;
            if ((new_expression[first_arg_idx_low] == "nan") || (new_expression[first_arg_idx_high] == "nan")) // ^ nan x -> nan, ^ x nan -> nan, ^ nan nan -> nan
            {
                //puts("hi 417");
                new_expression[op_idx] = "nan"; //change '^' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "0")) //^ x 0 -> 1 (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to ^ x 0, which is 1)
            {
                //puts("hi 334");
                new_expression[op_idx] = "1"; //change '^' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // ^ 0 x -> nan (for now)
            {
                //puts("hi 340");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[op_idx] = "nan";//new_expression[op_idx] = "0"; //change '^' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_high], "1")) // ^ x 1 -> x (because, since prefix operators come at the beginning, if the beginning of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to ^ x 1, which is 1)
            {
                //puts("hi 346");
                //erase the '1' at the end
                if (first_arg_idx_high == static_cast<int>(new_expression.size()) - 1)
                {
                    new_expression.pop_back();
                }
                else
                {
                    new_expression.erase(new_expression.begin() + first_arg_idx_high, new_expression.end());
                }
                new_expression.erase(new_expression.begin() + op_idx); //erase the '^'
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // ^ 1 x -> 1
            {
                //puts("hi 360");
                new_expression[op_idx] = "1"; //change '^' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "cos") // cos x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // cos
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // cos x
            if (new_expression[first_arg_idx_low] == "nan") // cos nan -> nan
            {
                //puts("hi 464");
                new_expression[op_idx] = "nan"; //change 'cos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // cos 0 -> 1
            {
    //            puts("hi 374");
                new_expression[op_idx] = "1"; //change 'cos' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf" || new_expression[first_arg_idx_low] == "-inf") // cos +/- inf -> nan
            {
    //            puts("hi 554");
                new_expression[op_idx] = "nan"; //change 'cos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // cos ~ inf -> nan
            {
    //            puts("hi 560");
                new_expression[op_idx] = "nan"; //change 'cos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "sin") // sin x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // sin
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sin x
            if (new_expression[first_arg_idx_low] == "nan") // sin nan -> nan
            {
                //puts("hi 484");
                new_expression[op_idx] = "nan"; //change 'sin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // sin 0 -> 0
            {
                //puts("hi 388");
                new_expression[op_idx] = "0"; //change 'sin' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf" || new_expression[first_arg_idx_low] == "-inf") // sin +/- inf -> nan
            {
    //            puts("hi 586");
                new_expression[op_idx] = "nan"; //change 'sin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // sin ~ inf -> nan
            {
    //            puts("hi 592");
                new_expression[op_idx] = "nan"; //change 'sin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "tanh") // tanh x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // tanh
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // tanh x
            if (new_expression[first_arg_idx_low] == "nan") // tanh nan -> nan
            {
                //puts("hi 504");
                new_expression[op_idx] = "nan"; //change 'tanh' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // tanh 0 -> 0
            {
                //puts("hi 402");
                new_expression[op_idx] = "0"; //change 'tanh' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // tanh inf -> 1
            {
                //puts("hi 408");
                new_expression[op_idx] = "1"; //change 'tanh' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // tanh -inf -> -1
            {
                //puts("hi 414");
                new_expression[op_idx] = "-1"; //change 'tanh' to '-1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // tanh ~ inf -> 1
            {
    //            puts("hi 421");
                new_expression[op_idx] = "-1"; //change 'tanh' to '-1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "sech") // sech x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // sech
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sech x

            if (new_expression[first_arg_idx_low] == "nan") // sech nan -> nan
            {
                //puts("hi 548");
                new_expression[op_idx] = "nan"; //change 'sech' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // sech 0 -> 1
            {
                //puts("hi 416");
                new_expression[op_idx] = "1"; //change 'sech' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // sech inf -> 0
            {
                //puts("hi 440");
                new_expression[op_idx] = "0"; //change 'sech' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // sech -inf -> 0
            {
    //            puts("hi 446");
                new_expression[op_idx] = "0"; //change 'sech' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // sech ~ inf -> 0
            {
    //            puts("hi 452");
                new_expression[op_idx] = "0"; //change 'sech' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "~") // ~ x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // ~
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ~ x

            if (new_expression[first_arg_idx_low] == "nan") // ~ nan -> nan
            {
                //puts("hi 592");
                new_expression[op_idx] = "nan"; //change '~' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // ~ 0 -> 0
            {
    //            puts("hi 466");
                new_expression[op_idx] = "0"; //change '~' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // ~ 1 -> -1
            {
                //puts("hi 696");
                new_expression[op_idx] = "-1"; //change '~' to '-1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "-1")) // ~ -1 -> 1
            {
                //puts("hi 702");
                new_expression[op_idx] = "1"; //change '~' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // ~ inf -> -inf
            {
                //puts("hi 507");
                new_expression[op_idx] = "-inf"; //change '~' to '-inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // ~ -inf -> inf
            {
                //puts("hi 702");
                new_expression[op_idx] = "inf"; //change '~' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // ~ ~ inf -> inf
            {
                //puts("hi 708");
                new_expression[op_idx] = "inf"; //change '~' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "exp") // exp x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // exp
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // exp x
            
            if (new_expression[first_arg_idx_low] == "nan") // exp nan -> nan
            {
                //puts("hi 618");
                new_expression[op_idx] = "nan"; //change 'exp' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // exp 0 -> 1
            {
                //puts("hi 521");
                new_expression[op_idx] = "1"; //change 'exp' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // exp inf -> inf
            {
                //puts("hi 527");
                new_expression[op_idx] = "inf"; //change 'exp' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // exp -inf -> 0
            {
                //puts("hi 741");
                new_expression[op_idx] = "0"; //change 'exp' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // exp ~ inf -> 0
            {
                //puts("hi 747");
                new_expression[op_idx] = "0"; //change 'exp' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if ((expression[low] == "log") || (expression[low] == "ln")) // ln x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // ln
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // ln x
            
            if (new_expression[first_arg_idx_low] == "nan") // ln nan -> nan
            {
                //puts("hi 726");
                new_expression[op_idx] = "nan"; //change 'ln' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // ln 0 -> -inf
            {
                //puts("hi 732");
                new_expression[op_idx] = "-inf"; //change 'ln' to '-inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // ln inf -> inf
            {
                //puts("hi 738");
                new_expression[op_idx] = "inf"; //change 'ln' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // ln -inf -> nan
            {
                //puts("hi 780");
                new_expression[op_idx] = "nan"; //change 'ln' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // ln ~ inf -> nan
            {
                //puts("hi 786");
                new_expression[op_idx] = "nan"; //change 'ln' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if ((expression[low] == "asin") || (expression[low] == "arcsin")) // asin x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // asin
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // asin x
            
            if (new_expression[first_arg_idx_low] == "nan") // asin nan -> nan
            {
                //puts("hi 753");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // asin 0 -> 0
            {
                //puts("hi 759");
                new_expression[op_idx] = "0"; //change 'asin' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // asin inf -> nan
            {
                //puts("hi 765");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // asin -inf -> nan
            {
                //puts("hi 819");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // asin ~ inf -> nan
            {
                //puts("hi 825");
                new_expression[op_idx] = "nan"; //change 'asin' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if ((expression[low] == "acos") || (expression[low] == "arccos")) // acos x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // acos
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // acos x
            
            if (new_expression[first_arg_idx_low] == "nan") // acos nan -> nan
            {
                //puts("hi 781");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // acos 1 -> 0
            {
                //puts("hi 787");
                new_expression[op_idx] = "0"; //change 'acos' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // arccos inf -> nan
            {
                //puts("hi 793");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // acos -inf -> nan
            {
                //puts("hi 858");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // acos ~ inf -> nan
            {
                //puts("hi 864");
                new_expression[op_idx] = "nan"; //change 'acos' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        else if (expression[low] == "sqrt") // sqrt x
        {
            int op_idx = new_expression.size();
            new_expression.push_back(expression[low]); // sqrt
            int temp = low+1+grasp[low+1];
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // sqrt x
            
            if (new_expression[first_arg_idx_low] == "nan") // sqrt nan -> nan
            {
    //            puts("hi 808");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "0")) // sqrt 0 -> 0
            {
    //            puts("hi 814");
                new_expression[op_idx] = "0"; //change 'sqrt' to '0'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "1")) // sqrt 1 -> 1
            {
    //            puts("hi 820");
                new_expression[op_idx] = "1"; //change 'sqrt' to '1'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "inf") // sqrt inf -> inf
            {
    //            puts("hi 826");
                new_expression[op_idx] = "inf"; //change 'sqrt' to 'inf'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression[first_arg_idx_low] == "-inf") // sqrt -inf -> nan
            {
    //            puts("hi 903");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "inf")) // sqrt ~ inf -> nan
            {
    //            puts("hi 909");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression[first_arg_idx_low], "-1")) // sqrt -1 -> nan
            {
    //            puts("hi 915");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression[first_arg_idx_low] == "~") && ((first_arg_idx_low+1) < (new_expression.size()))  && (new_expression[first_arg_idx_low+1] == "1")) // sqrt ~ 1 -> nan
            {
    //            puts("hi 921");
                new_expression[op_idx] = "nan"; //change 'sqrt' to 'nan'
                new_expression.erase(new_expression.begin() + op_idx + 1, new_expression.end()); //erase the rest
            }
        }
        
        else if (expression[low] == "abs") // abs x
        {
            new_expression.push_back(expression[low]); // abs
            int temp = low+1+grasp[low+1];
            graspSimplifyPrefixHelper(expression, low+1, temp, grasp, new_expression, true); // abs x
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

    void graspSimplifyPrefix(std::vector<std::string>& expression, int low, int up, std::vector<int>& grasp)
    {
        std::vector<std::string> new_expression;
        new_expression.reserve(expression.size());
        graspSimplifyPrefixHelper(expression, low, up, grasp, new_expression, false);
        expression = new_expression;
    }

    //scans entire `expression` for the following:
    //     bin_op number1 number2 -> numberResult
    //     un_op number1 -> numberResult
    //the function repeatedly iterates over `expression` until no more
    //instances of the above are found.
    //+ cos - + 1 3 x + * cos 2 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
    //+ cos - 4 x + * cos 2 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
    //+ cos - 4 x + * -0.42 sin 3 + arcsin - 8 7 sin tanh cos * 3 4
    //+ cos - 4 x + * -0.42 0.41 + arcsin - 8 7 sin tanh cos * 3 4
    void simplifyPN_Helper(std::vector<std::string>& expression)
    {
        if (expression.empty()) return;

        bool isdouble1, isdouble2, isConst1, isConst2;
        thread_local std::vector<std::string> temp;
        temp.clear();
        temp.reserve(expression.size());

        for (size_t i = 0; i < expression.size(); ++i)
        {
            // Convenience bounds flags
            bool has1 = (i + 1 < expression.size());
            bool has2 = (i + 2 < expression.size());

            // ==== BINARY OPERATORS ====
            if (is_binary(expression[i]))
            {
                // If we don't have two operands, just copy token and move on
                if (!has1 || !has2)
                {
                    temp.push_back(expression[i]);
                    continue;
                }

                isdouble1 = isdouble(expression[i+1]);
                isdouble2 = isdouble(expression[i+2]);

                if (isdouble1 && isdouble2)
                {
                    if (expression[i] == "+")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) + Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "-")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) - Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "*")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) * Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "/")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(Stod(expression[i+1]) / Stod(expression[i+2]))
                        ));
                        i += 2;
                        continue;
                    }
                    else if (expression[i] == "^")
                    {
                        temp.push_back(simplifyString(
                            to_string_general(std::pow(Stod(expression[i+1]), Stod(expression[i+2])))
                        ));
                        i += 2;
                        continue;
                    }
                }

                isConst1 = is_const(expression[i+1]);
                isConst2 = is_const(expression[i+2]);

                if ((isConst1 && isConst2) &&
                    ((expression[i+1].find("nan") != std::string::npos) ||
                     (expression[i+2].find("nan") != std::string::npos)))
                {
                    // binary_op nan x = binary_op x nan = nan
                    temp.push_back("nan");
                    i += 2;
                    continue;
                }
                else if (expression[i] == "-")
                {
                    if ((isConst1 && isConst2) && (expression[i+1] == expression[i+2])) // - x x => 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "0")) // - 0 x -> ~ x
                    {
                        temp.push_back("~");
                        temp.push_back(expression[i+2]);
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) // - x 0 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "*")
                {
                    if (checkEqual(expression[i+1], "0") && isConst2) //* 0 x -> 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) //* x 0 -> 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "1") && isConst2) //* 1 x -> x
                    {
                        temp.push_back(expression[i+2]);
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "1") && isConst1) //* x 1 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "+")
                {
                    if (checkEqual(expression[i+1], "0") && isConst2) // + 0 x -> x
                    {
                        temp.push_back(expression[i+2]);
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) // + x 0 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "/")
                {
                    if (checkEqual(expression[i+1], "0") && isConst2) // / 0 x -> 0
                    {
                        temp.push_back("0");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "0") && isConst1) // / x 0 -> nan
                    {
                        temp.push_back("nan");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "1") && isConst1) // / x 1 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                    else if (isConst1 && isConst2 && (expression[i+1] == expression[i+2])) // / x x -> 1
                    {
                        temp.push_back("1");
                        i += 2;
                        continue;
                    }
                }
                else if (expression[i] == "^")
                {
                    if (checkEqual(expression[i+2], "0") && isConst1) // ^ x 0 -> 1
                    {
                        temp.push_back("1");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "0") && isConst2) // ^ 0 x -> nan
                    {
                        temp.push_back("nan");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+1], "1") && isConst2) // ^ 1 x -> 1
                    {
                        temp.push_back("1");
                        i += 2;
                        continue;
                    }
                    else if (checkEqual(expression[i+2], "1") && isConst1) // ^ x 1 -> x
                    {
                        temp.push_back(expression[i+1]);
                        i += 2;
                        continue;
                    }
                }

                // If none of the simplifications fired, just copy the operator.
                temp.push_back(expression[i]);
                continue;
            }

            // ==== UNARY OPERATORS WITH NUMERIC ARG ====
            else if (is_unary(expression[i]) && has1 && isdouble(expression[i+1]))
            {
                if (expression[i] == "cos")
                {
                    temp.push_back(simplifyString(
                        to_string_general(cos(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "~")
                {
                    if (expression[i+1] == "inf") //~ inf -> -inf
                    {
                        temp.push_back("-inf");
                    }
                    else if (checkEqual(expression[i+1], "0"))
                    {
                        temp.push_back("0");
                    }
                    else
                    {
                        temp.push_back(simplifyString(
                            to_string_general(-(Stod(expression[i+1])))
                        ));
                    }
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sin")
                {
                    temp.push_back(simplifyString(
                        to_string_general(sin(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if ((expression[i] == "ln") || (expression[i] == "log"))
                {
                    temp.push_back(simplifyString(
                        to_string_general(log(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "asin" || expression[i] == "arcsin")
                {
                    temp.push_back(simplifyString(
                        to_string_general(asin(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "acos" || expression[i] == "arccos")
                {
                    temp.push_back(simplifyString(
                        to_string_general(acos(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "exp")
                {
                    temp.push_back(simplifyString(
                        to_string_general(exp(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sech")
                {
                    temp.push_back(simplifyString(
                        to_string_general(1 / cosh(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "tanh")
                {
                    temp.push_back(simplifyString(
                        to_string_general(tanh(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sqrt")
                {
                    temp.push_back(simplifyString(
                        to_string_general(sqrt(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else if (expression[i] == "abs")
                {
                    temp.push_back(simplifyString(
                        to_string_general(abs(Stod(expression[i+1])))
                    ));
                    i += 1;
                    continue;
                }
                else
                {
                    temp.push_back(expression[i]);
                    continue;
                }
            }

            // ==== UNARY OPERATORS WITH SYMBOLIC ARG ====
            else if (is_unary(expression[i]))
            {
                // If we don't even have one operand, just copy and continue
                if (!has1)
                {
                    temp.push_back(expression[i]);
                    continue;
                }

                if (expression[i] == "~" && expression[i+1] == "~")
                {
                    // ~~x -> x (we just skip both ~)
                    i += 1;
                    continue;
                }
                else if (expression[i] == "exp" && (expression[i+1] == "ln" || expression[i+1] == "log"))
                {
                    // exp ln x -> x (we skip exp, ln; x will be processed later)
                    i += 1;
                    continue;
                }
                else if (expression[i+1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                {
                    // ln exp x -> x (skip ln, exp)
                    i += 1;
                    continue;
                }
                else if (expression[i] == "cos" && (expression[i+1] == "acos" || expression[i+1] == "arccos"))
                {
                    // cos(acos x) -> x; skip cos, acos
                    i += 1;
                    continue;
                }
                else if ((expression[i] == "cos") && (expression[i+1] == "~")) // cos(-x) = cos(x)
                {
                    temp.push_back(expression[i]);
                    i += 1; // skip the ~
                    continue;
                }
                else if (expression[i+1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                {
                    // acos(cos x) -> x; skip acos, cos
                    i += 1;
                    continue;
                }
                else if (expression[i] == "sin" && (expression[i+1] == "asin" || expression[i+1] == "arcsin"))
                {
                    // sin(asin x) -> x
                    i += 1;
                    continue;
                }
                else if (expression[i+1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                {
                    // asin(sin x) -> x
                    i += 1;
                    continue;
                }
                else if ((expression[i] == "sech") && (expression[i+1] == "~")) // sech(-x) = sech(x)
                {
                    temp.push_back(expression[i]);
                    i += 1; // skip the ~
                    continue;
                }
                else
                {
                    temp.push_back(expression[i]);
                    continue;
                }
            }

            // ==== NOT AN OPERATOR ====
            else
            {
                temp.push_back(expression[i]);
                continue;
            }
        }

        if (expression.size() != temp.size())
        {
            expression = temp;
        }
    }

    void simplifyPN(std::vector<std::string>& expression)
    {
        size_t size_before, size_after;
        do
        {
            size_before = expression.size();
            simplifyPN_Helper(expression);
            this->simplify_grasp.reserve(expression.size());
            graspSimplifyPrefix(expression, 0, expression.size() - 1, this->simplify_grasp);
            simplifyPN_Helper(expression);
            size_after = expression.size();
        } while (size_before != size_after);
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
            
            if (checkEqual(new_expression.back(), "0")) // x 0 +/- -> x
            {
                //puts("hi 181");
                new_expression.pop_back();
            }
            
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) // 0 x +/- -> x
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
            
            else if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x +/- -> nan, x nan +/- -> nan, nan nan +/- -> nan
            {
                //puts("hi 269");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
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
            
            if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x * -> nan, x nan * -> nan, nan nan * -> nan
            {
                //puts("hi 297");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "0")) // x 0 * -> 0 (because, since postfix operators come at the end, if the end of the second argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 *, which is 0)
            {
                //puts("hi 235");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) //0 x * -> 0 (because, since postfix operators come at the end, if the end of the first argument of '*' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x *, which is 0)
            {
                //puts("hi 241");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "1")) // x 1 * -> x (because, since postfix operators come at the end, if the end of the second argument of '*' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 *, which is x)
            {
                //puts("hi 247");
                new_expression.pop_back(); //erase the '1'
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "1")) //1 x * -> x (because, since postfix operators come at the end, if the end of the first argument of '*' is 1, then the whole first argument MUST be 1, therefore the expression reduces to 1 x *, which is x)
            {
                //puts("hi 252");
                new_expression.erase(new_expression.begin() + first_arg_idx_high - 1); //erase the '1'
            }
            else if (checkEqual(new_expression.back(), "-1")) // x -1 * -> x ~ (because, since postfix operators come at the end, if the end of the second argument of '*' is -1, then the whole second argument MUST be -1, therefore the expression reduces to x -1 *, which is x ~)
            {
    //            puts("hi 368");
                new_expression.back() = "~"; //change the '-1' to a "~"
                
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "-1")) //-1 x * -> x ~ (because, since postfix operators come at the end, if the end of the first argument of '*' is -1, then the whole first argument MUST be -1, therefore the expression reduces to -1 x *, which is x ~)
            {
    //            puts("hi 374");
                new_expression.erase(new_expression.begin() + first_arg_idx_high - 1); //erase the '-1'
                new_expression.push_back("~"); //add a "~" at the end
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
            
            if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x / -> nan, x nan / -> nan, nan nan / -> nan
            {
                //puts("hi 339");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if ((checkEqual(new_expression.back(), "0")) && (checkEqual(new_expression[first_arg_idx_high - 1], "0"))) // 0 0 / -> nan
            {
                //puts("hi 279");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "0")) // x 0 / -> nan (for now, because, since postfix operators come at the end, if the end of the second argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 /, which is, for now, assumed to be nan for simplicity)
            {
                //puts("hi 280");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[first_arg_idx_low] = "nan";//(new_expression[first_arg_idx_high - 1] == "~") ? "-inf" : "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) //0 x / -> 0 (because, since postfix operators come at the end, if the end of the first argument of '/' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x /, which is 0)
            {
                //puts("hi 286");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "1")) // x 1 / -> x (because, since postfix operators come at the end, if the end of the second argument of '/' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 /, which is x)
            {
                //puts("hi 292");
                new_expression.pop_back(); //erase the '1'
            }
            else if (checkEqual(new_expression.back(), "-1")) // x -1 / -> x ~ (because, since postfix operators come at the end, if the end of the second argument of '/' is -1, then the whole second argument MUST be -1, therefore the expression reduces to x -1 /, which is x ~)
            {
    //            puts("hi 425");
                new_expression.back() = "~"; //change the '-1' to a "~"
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
                 
                  x              1
                ------ --> x * ------
                number         number
                 
                x num * num1 * -> x num num1 * *
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
            
            if ((new_expression[first_arg_idx_high - 1] == "nan") || (new_expression.back() == "nan")) //nan x ^ -> nan, x nan ^ -> nan, nan nan ^ -> nan
            {
                //puts("hi 397");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "0")) // x 0 ^ -> 1 (because, since postfix operators come at the end, if the end of the second argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to x 0 ^, which is 1)
            {
                //puts("hi 318");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "0")) //0 x ^ -> nan (for now, because, since postfix operators come at the end, if the end of the first argument of '^' is 0, then the whole second argument MUST be 0, therefore the expression reduces to 0 x ^, which is, for now, assumed to be nan for simplicity)
            {
                //puts("hi 324");
                //TODO: need to come up with a more robust way that actually checks if this is nan anywhere;
                //for now we weed it out because annoying not to; giving this up seems like the better deal...
                new_expression[first_arg_idx_low] = "nan";// "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest of x and y
            }
            else if (checkEqual(new_expression.back(), "1")) // x 1 ^ -> x (because, since postfix operators come at the end, if the end of the second argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to x 1 ^, which is x)
            {
                //puts("hi 330");
                new_expression.pop_back(); //erase the '1'
            }
            else if (checkEqual(new_expression[first_arg_idx_high - 1], "1")) //1 x ^ -> 1 (because, since postfix operators come at the end, if the end of the first argument of '^' is 1, then the whole second argument MUST be 1, therefore the expression reduces to 1 x ^, which is 1)
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
            
            if (new_expression.back() == "nan") // nan cos -> nan (because, since postfix operators come at the end, if the end of the argument of 'cos' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan cos, which is nan)
            {
                //puts("hi 439");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 cos -> 1 (because, since postfix operators come at the end, if the end of the argument of 'cos' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 cos, which is 1)
            {
                //puts("hi 350");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "inf") || (new_expression.back() == "-inf")) // +/- inf cos -> nan (because, since postfix operators come at the end, if the end of the argument of 'cos' is +/- inf, then the whole argument MUST be +/- inf, therefore the expression reduces to +/- inf cos, which is nan)
            {
    //            puts("hi 499");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ cos -> nan (because, since postfix operators come at the end, if the end of the argument of 'cos' is inf ~, then the whole argument MUST be inf ~, therefore the expression reduces to inf ~ cos, which is nan)
            {
    //            puts("hi 505");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "sin") //x sin
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan sin -> nan (because, since postfix operators come at the end, if the end of the argument of 'sin' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan sin, which is nan)
            {
                //puts("hi 462");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 sin -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sin' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sin, which is 0)
            {
                //puts("hi 365");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "inf") || (new_expression.back() == "-inf")) // +/- inf sin -> nan (because, since postfix operators come at the end, if the end of the argument of 'sin' is +/- inf, then the whole argument MUST be +/- inf, therefore the expression reduces to +/- inf sin, which is nan)
            {
    //            puts("hi 533");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sin -> nan (because, since postfix operators come at the end, if the end of the argument of 'sin' is inf ~, then the whole argument MUST be inf ~, therefore the expression reduces to inf ~ sin, which is nan)
            {
    //            puts("hi 539");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "tanh") //x tanh
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan tanh -> nan (because, since postfix operators come at the end, if the end of the argument of 'tanh' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan tanh, which is nan)
            {
                //puts("hi 485");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 tanh -> 0 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 tanh, which is 0)
            {
                //puts("hi 380");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf tanh -> 1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf tanh, which is 1)
            {
                //puts("hi 386");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
            {
                //puts("hi 392");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ tanh -> -1 (because, since postfix operators come at the end, if the end of the argument of 'tanh' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf tanh, which is -1)
            {
                //puts("hi 392");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
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
            
            if (new_expression.back() == "nan") // nan sech -> nan (because, since postfix operators come at the end, if the end of the argument of 'sech' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan sech, which is nan)
            {
                //puts("hi 525");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 sech -> 1 (because, since postfix operators come at the end, if the end of the argument of 'sech' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sech, which is 1)
            {
                //puts("hi 395");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf sech, which is 0)
            {
    //            puts("hi 419");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
            {
                //puts("hi 425");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sech -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sech' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sech, which is 0)
            {
                //puts("hi 431");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
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
            
            if (new_expression.back() == "nan") // nan ~ -> nan (because, since postfix operators come at the end, if the end of the argument of '~' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan ~, which is nan)
            {
                //puts("hi 565");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 ~ -> 0 (because, since postfix operators come at the end, if the end of the argument of '~' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 ~, which is 0)
            {
    //            puts("hi 445");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "1")) // 1 ~ -> -1 (because, since postfix operators come at the end, if the end of the argument of '~' is 1, then the whole argument MUST be 1, therefore the expression reduces to 1 ~, which is -1)
            {
    //            puts("hi 663");
                new_expression[first_arg_idx_low] = "-1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "-1")) // -1 ~ -> 1 (because, since postfix operators come at the end, if the end of the argument of '~' is -1, then the whole argument MUST be -1, therefore the expression reduces to -1 ~, which is 1)
            {
    //            puts("hi 669");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf ~ -> -inf (because, since postfix operators come at the end, if the end of the argument of '~' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf ~, which is -inf)
            {
                //puts("hi 507");
                new_expression[first_arg_idx_low] = "-inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf ~ -> inf (because, since postfix operators come at the end, if the end of the argument of '~' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ~, which is inf)
            {
    //            puts("hi 669");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ ~ -> 0 (because, since postfix operators come at the end, if the end of the argument of '~' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ~, which is inf)
            {
    //            puts("hi 675");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
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
            
            if (new_expression.back() == "nan") // nan exp -> nan (because, since postfix operators come at the end, if the end of the argument of 'exp' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan exp, which is nan)
            {
                //puts("hi 593");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 exp -> 1 (because, since postfix operators come at the end, if the end of the argument of 'exp' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 exp, which is 1)
            {
                //puts("hi 524");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf exp -> inf (because, since postfix operators come at the end, if the end of the argument of 'exp' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf exp, which is inf)
            {
                //puts("hi 530");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf exp -> 0 (because, since postfix operators come at the end, if the end of the argument of 'exp' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf exp, which is 0)
            {
    //            puts("hi 697");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ exp -> 0 (because, since postfix operators come at the end, if the end of the argument of 'exp' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf exp, which is 0)
            {
    //            puts("hi 703");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if ((expression[up] == "ln") || (expression[up] == "log")) //x ln
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan ln -> nan (because, since postfix operators come at the end, if the end of the argument of 'ln' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan ln, which is nan)
            {
    //            puts("hi 707");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 ln -> -inf (because, since postfix operators come at the end, if the end of the argument of 'ln' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 ln, which is -inf)
            {
    //            puts("hi 713");
                new_expression[first_arg_idx_low] = "-inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf ln -> inf (because, since postfix operators come at the end, if the end of the argument of 'ln' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf ln, which is inf)
            {
    //            puts("hi 719");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf ln -> nan (because, since postfix operators come at the end, if the end of the argument of 'ln' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ln, which is nan)
            {
    //            puts("hi 750");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ ln -> nan (because, since postfix operators come at the end, if the end of the argument of 'ln' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf ln, which is nan)
            {
    //            puts("hi 756");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if ((expression[up] == "asin") || (expression[up] == "arcsin")) //x asin
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan asin, which is nan)
            {
    //            puts("hi 735");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 asin -> 0 (because, since postfix operators come at the end, if the end of the argument of 'asin' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 asin, which is 0)
            {
    //            puts("hi 741");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf asin, which is nan)
            {
    //            puts("hi 747");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf asin, which is nan)
            {
    //            puts("hi 790");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ asin -> nan (because, since postfix operators come at the end, if the end of the argument of 'asin' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf asin, which is nan)
            {
    //            puts("hi 796");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if ((expression[up] == "acos") || (expression[up] == "arccos")) //x acos
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan acos, which is nan)
            {
    //            puts("hi 763");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "1")) // 1 acos -> 0 (because, since postfix operators come at the end, if the end of the argument of 'acos' is 1, then the whole argument MUST be 1, therefore the expression reduces to 1 acos, which is 0)
            {
    //            puts("hi 769");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf acos, which is nan)
            {
    //            puts("hi 775");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf acos, which is nan)
            {
    //            puts("hi 830");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ acos -> nan (because, since postfix operators come at the end, if the end of the argument of 'acos' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf acos, which is nan)
            {
    //            puts("hi 836");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else
            {
                new_expression.push_back(expression[up]);
            }
        }
        else if (expression[up] == "sqrt") //x sqrt
        {
            int first_arg_idx_low = new_expression.size();
            graspSimplifyPostfixHelper(expression, low, up-1, grasp, new_expression, true); //x
            
            if (new_expression.back() == "nan") // nan sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is nan, then the whole argument MUST be nan, therefore the expression reduces to nan sqrt, which is nan)
            {
    //            puts("hi 791");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "0")) // 0 sqrt -> 0 (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is 0, then the whole argument MUST be 0, therefore the expression reduces to 0 sqrt, which is 0)
            {
    //            puts("hi 797");
                new_expression[first_arg_idx_low] = "0";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "1")) // 1 sqrt -> 1 (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is 1, then the whole argument MUST be 1, therefore the expression reduces to 1 sqrt, which is 1)
            {
    //            puts("hi 803");
                new_expression[first_arg_idx_low] = "1";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (checkEqual(new_expression.back(), "-1")) // -1 sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is -1, then the whole argument MUST be -1, therefore the expression reduces to -1 sqrt, which is nan)
            {
    //            puts("hi 870");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "inf") // inf sqrt -> inf (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is inf, then the whole argument MUST be inf, therefore the expression reduces to inf sqrt, which is inf)
            {
    //            puts("hi 809");
                new_expression[first_arg_idx_low] = "inf";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if (new_expression.back() == "-inf") // -inf sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sqrt, which is nan)
            {
    //            puts("hi 876");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
            }
            else if ((new_expression.back() == "~") && (new_expression.size() >= 2) && ((*(new_expression.end() - 2)) == "inf")) // inf ~ sqrt -> nan (because, since postfix operators come at the end, if the end of the argument of 'sqrt' is -inf, then the whole argument MUST be -inf, therefore the expression reduces to -inf sqrt, which is nan)
            {
    //            puts("hi 882");
                new_expression[first_arg_idx_low] = "nan";
                new_expression.erase(new_expression.begin() + first_arg_idx_low + 1, new_expression.end()); //erase the rest
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
        if (expression.size() < 2) return; //just an operand
        
        bool isdouble1, isdouble2, isConst1, isConst2;
        thread_local std::vector<std::string> temp;
        temp.clear();
        temp.reserve(expression.size());
    //    printf("expression before = ");for (const auto& i: expression){std::cout << i << ' ';}puts("");
        for (int i = static_cast<int>(expression.size()) - 1; i >= 0; i--)
        {
    //        std::cout << "i = " << i << '\n';// << expression[i] << '\n';
            if (is_binary(expression[i]))
            {
                isdouble1 = isdouble(expression[i-1]);
                isdouble2 = isdouble(expression[i-2]);
                
                if (isdouble1 && isdouble2)
                {
                    if (expression[i] == "+")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) + Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "-")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) - Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "*")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) * Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "/")
                    {
                        temp.push_back(simplifyString(to_string_general(Stod(expression[i-2]) / Stod(expression[i-1]))));
                        i -= 2;
                        continue;
                    }
                    else if (expression[i] == "^")
                    {
                        temp.push_back(simplifyString(to_string_general(std::pow(Stod(expression[i-2]), Stod(expression[i-1])))));
    //                            printf("hi 566, res = %s\n", expression[i].c_str());
                        i -= 2;
                        continue;
                    }
                }
                
                isConst1 = is_const(expression[i-1]);
                isConst2 = is_const(expression[i-2]);
                
                if ((isConst1 && isConst2) && ((expression[i-1].find("nan") != std::string::npos) || (expression[i-2].find("nan") != std::string::npos))) //x nan binary_op = nan x binary_op = nan
                {
    //                        puts("hi 549");
                    temp.push_back("nan");
                    i -= 2;
                    continue;
                }
                else if (expression[i] == "-")
                {
                    if ((isConst1 && isConst2) && (expression[i-1] == expression[i-2])) //x x - => 0
                    {
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "0") && isConst1) //"0 x -" -> "x ~"
                    {
                        temp.push_back("~");
                        temp.push_back(expression[i-1]);
                        

                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0")) //"x 0 -" -> "x"
                    {
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "*")
                {
                    if (checkEqual(expression[i-2], "0") && isConst1) //"0 x *" -> "0"
                    {
                        //puts("hi 131");
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0") && isConst2) //"x 0 *" -> "0"
                    {
                        //puts("hi 139");
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "1") && isConst1) //"1 x *" -> "x"
                    {
                        //puts("hi 147");
                        temp.push_back(expression[i-1]);
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "1") && isConst2) //"x 1 *" -> "x"
                    {
                        //puts("hi 155");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "+")
                {
                    if (checkEqual(expression[i-2], "0") && isConst1) //"0 x +" -> "x"
                    {
                        //puts("hi 167");
                        temp.push_back(expression[i-1]);
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0") && isConst2) //"x 0 +" -> "x"
                    {
                        //puts("hi 175");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "/")
                {
                    if (checkEqual(expression[i-2], "0") && isConst1) // "0 x /" -> "0"
                    {
                        //puts("hi 187");
                        temp.push_back("0");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "1") && isConst2) // "x 1 /" -> "x"
                    {
                        //puts("hi 195");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "0") && isConst2) // "x 0 /" -> "nan"
                    {
                        //puts("hi 859");
                        temp.push_back("nan");
                        i -= 2;
                        continue;
                    }
                    else if (isConst1 && isConst2 && (expression[i-1] == expression[i-2])) // "x x /" -> "1"
                    {
                        //puts("hi 203");
                        temp.push_back("1");
                        i -= 2;
                        continue;
                    }
                }

                else if (expression[i] == "^")
                {
                    if (checkEqual(expression[i-1], "0") && isConst2) // "x 0 ^" -> "1"
                    {
                        //puts("hi 223");
                        temp.push_back("1");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "0") && isConst1) // "0 x ^" -> "nan"
                    {
                        //puts("hi 880");
                        temp.push_back("nan");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-2], "1") && isConst1) // "1 x ^" -> "1"
                    {
                        //puts("hi 231");
                        temp.push_back("1");
                        i -= 2;
                        continue;
                    }
                    else if (checkEqual(expression[i-1], "1") && isConst2) // "x 1 ^" -> "x"
                    {
                        //puts("hi 239");
                        temp.push_back(expression[i-2]);
                        i -= 2;
                        continue;
                    }
                }
                temp.push_back(expression[i]);
                continue;
            }
            
            else if (is_unary(expression[i]) && isdouble(expression[i-1]))
            {
                if (expression[i] == "cos")
                {
                    temp.push_back(simplifyString(to_string_general(cos(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "~")
                {
                    if (checkEqual(expression[i-1], "0")) //0 ~ -> 0
                    {
                        //puts("hi 917");
                        temp.push_back("0");
                    }
                    else if (expression[i-1] == "inf") //inf ~ -> -inf
                    {
                        //puts("hi 924");
                        temp.push_back("-inf"); // change 'inf' to '-inf'
                    }
                    else
                    {
                        temp.push_back(simplifyString(to_string_general(-(Stod(expression[i-1])))));
                    }
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sin")
                {
                    temp.push_back(simplifyString(to_string_general(sin(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if ((expression[i] == "ln") || (expression[i] == "log"))
                {
                    temp.push_back(simplifyString(to_string_general(log(Stod(expression[i-1]))))); // Natural log (ln)
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "asin" || expression[i] == "arcsin")
                {
                    temp.push_back(simplifyString(to_string_general(asin(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "acos" || expression[i] == "arccos")
                {
                    temp.push_back(simplifyString(to_string_general(acos(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "exp")
                {
                    temp.push_back(simplifyString(to_string_general(exp(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sech")
                {
                    temp.push_back(simplifyString(to_string_general(1.0 / cosh(Stod(expression[i-1]))))); // sech(x) = 1 / cosh(x)
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "tanh")
                {
                    temp.push_back(simplifyString(to_string_general(tanh(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sqrt")
                {
                    temp.push_back(simplifyString(to_string_general(sqrt(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "abs")
                {
                    temp.push_back(simplifyString(to_string_general(abs(Stod(expression[i-1])))));
                    i -= 1;
                    continue;
                }
            }
            
            else if (is_unary(expression[i]))
            {
                if (expression[i] == "~" && expression[i-1] == "~")
                {
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "exp" && (expression[i-1] == "ln" || expression[i-1] == "log"))
                {
                    //puts("hi 360");
                    i -= 1;
                    continue;
                }
                else if (expression[i-1] == "exp" && (expression[i] == "ln" || expression[i] == "log"))
                {
                    //puts("hi 368");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "cos" && (expression[i-1] == "acos" || expression[i-1] == "arccos"))
                {
                    //puts("hi 408");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "cos" && expression[i-1] == "~") //cos(-x) -> cos(x)
                {
                    //puts("hi 688");
                    temp.push_back(expression[i]);
                    i -= 1;
                    continue;
                }
                else if (expression[i-1] == "cos" && (expression[i] == "acos" || expression[i] == "arccos"))
                {
                    //puts("hi 416");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sin" && (expression[i-1] == "asin" || expression[i-1] == "arcsin"))
                {
                    //puts("hi 424");
                    i -= 1;
                    continue;
                }
                else if (expression[i-1] == "sin" && (expression[i] == "asin" || expression[i] == "arcsin"))
                {
                    //puts("hi 432");
                    i -= 1;
                    continue;
                }
                else if (expression[i] == "sech" && expression[i-1] == "~") //'x ~ sech' -> 'x sech'
                {
                    temp.push_back(expression[i]);
                    i -= 1;
                    continue;
                }
                else
                {
                    temp.push_back(expression[i]);
                    continue;
                }
            }
            else
            {
                temp.push_back(expression[i]);
                continue;
            }
        }
        
        if (expression.size() != temp.size()) //if expression was simplified, assign it to original expression vector
        {
            expression.resize(temp.size());
            for (size_t i = 0; i < expression.size(); i++)
            {
                expression[i] = temp[expression.size() - i - 1];
            }
        }
    //    printf("expression after = ");for (const auto& i: expression){std::cout << i << ' ';}puts("");

    }

    
    void simplifyRPN(std::vector<std::string>& expression)
    {
        size_t size_before, size_after;
        do
        {
            size_before = expression.size();
            simplifyRPN_Helper(expression);
            this->simplify_grasp.reserve(expression.size());
            graspSimplifyPostfix(expression, 0, expression.size() - 1, this->simplify_grasp);
            simplifyRPN_Helper(expression);
            size_after = expression.size();
        } while (size_before != size_after);
    }
    
    /*
     Returns a pair containing the depth of the sub-expression from start to stop, and whether or not it's complete
     Algorithm adopted from here: https://stackoverflow.com/a/77180279
     */
    std::pair<int, bool> getPNdepth(const std::vector<std::string>& expression, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool binary = false, bool unary = false, bool leaf = false)
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
    std::pair<int, bool> getRPNdepth(const std::vector<std::string>& expression, size_t start = 0, size_t stop = 0, bool cache = false, bool modify = false, bool unary = false, bool leaf = false)
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
    
    std::vector<std::string> get_legal_moves()
    {
        if (this->expression_type == "prefix")
        {
            if (srnn.pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the operators...
            {
                if (this->n != 0) // if the depth is not 0
                {
                    return Board::__operators;
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
                if (Board::__binary_operators.size() > 0)
                {
                    srnn.pieces.push_back(Board::__binary_operators[0]);
                    bin_allowed = (getPNdepth(srnn.pieces).first <= this->n);
                }
                if (Board::__unary_operators.size() > 0)
                {
                    srnn.pieces[srnn.pieces.size() - 1] = Board::__unary_operators[0];
                    una_allowed = (getPNdepth(srnn.pieces).first <= this->n);
                }
                srnn.pieces[srnn.pieces.size() - 1] = Board::__input_vars[0];
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
                if (Board::__unary_operators.size() > 0)
                {
                    srnn.pieces.push_back(Board::__unary_operators[0]);
                    una_allowed = ((num_leaves >= 1) && (getRPNdepth(srnn.pieces).first <= this->n));
                }
                
                srnn.pieces[srnn.pieces.size() - 1] = Board::__input_vars[0];
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
        for (size_t i = 0; i <= sz; i++)
        {
            temp += ((i!=sz) ? srnn.pieces[i] + " " : srnn.pieces[i]);
        }
        return temp;
    }
    
    std::string _to_infix(bool show_consts = true)
    {
        std::stack<std::string> stack;
        bool is_prefix = (expression_type == "prefix");
        std::string result, token;
        
        for (int i = (is_prefix ? (srnn.pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < srnn.pieces.size())); (is_prefix ? (i--) : (i++)))
        {
            token = srnn.pieces[i];

            if (std::find(Board::__operators.begin(), Board::__operators.end(), token) == Board::__operators.end()) // leaf
            {
                stack.push(token);
            }
            else if (std::find(Board::__unary_operators.begin(), Board::__unary_operators.end(), token) != Board::__unary_operators.end()) // Unary operator
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
        if (this->srnn.pieces.empty())
        {
            if (this->is_primary)
            {
                this->srnn.reset_params();
            }
            this->stack.clear();
            this->idx = 0;
            if (this->expression_type == "prefix")
            {
                this->depth = 0, this->num_binary = 0, this->num_leaves = 0;
            }
        }
        //structured binding :)
        auto [depth, complete] = ((this->expression_type == "prefix") ?
                                   getPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/) :
                                   getRPNdepth(srnn.pieces, 0 /*start*/, 0 /*stop*/, this->cache && cache /*cache*/, true /*modify*/));
        if (!complete || depth < this->n) //Expression not complete
        {
            return -1;
        }
        else
        {
            if (this->is_primary)
            {
                if (!cache)
                {
                    this->srnn.reset_params();
                }
                this->expression_string.clear();
                for (const std::string& i: this->srnn.pieces){this->expression_string += i+" ";}
                Board::expression_set.insert(this->expression_string);
                return (1.0f/(1.0f+this->srnn.train(data.rows, data.labels, this->epochs)));
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
    
    //Computes the grasp of an arbitrary element srnn.pieces[i],
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
    
    //Adds pairs containing the starting and stopping indices for each
    //depth-n sub-expression in the expression individual
    void get_indices(std::vector<std::pair<int, int>>& sub_exprs, std::vector<std::string>& individual)
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

    void setPrefixGR(const std::vector<std::string>& prefix, std::vector<int>& grasp)
    {
        grasp.reserve(prefix.size());
        for (size_t k = 0; k < prefix.size(); ++k)
        {
            grasp.push_back(GR(k, prefix));
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

//https://arxiv.org/abs/2310.06609
std::vector<std::pair<std::vector<std::string>, float>>
     GP(const Eigen::MatrixXf& data,
        int depth = 3,
        std::string expression_type = "prefix",
        bool cache = true,
        double time = 120 /*time to run the algorithm in seconds*/,
        const char* filename = "" /*name of file to save the results to*/,
        unsigned int num_threads = 0,
        std::vector<int> layers = {},
        std::deque<std::string> layer_types = {},
        const unsigned long num_epochs = 100,
        float bias = 1.0f,
        float eta = 0.5f,
        float theta = 0.01f,
        float gamma = 0.9f,
        float epsilon = 0.1f,
        float beta_1 = 0.9f,
        float beta_2 = 0.999f,
        float lambda = 0.01f /*weight decay AdamW*/,
        const std::vector<std::pair<std::vector<std::string>, float>>& seed_individuals = {})
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
    std::atomic<float> max_score{0.0}; //an atomic float variable called `max_score` that's initialized to 0
    std::string best_expression, orig_expression;
    
    auto start_time = Clock::now();
    
    /*
     Inside of thread:
     */
    
    auto func = [&depth, &expression_type, &data, &cache, &start_time, &time, &max_score, &sync_point, &layers, &layer_types, &num_epochs, &bias, &eta, &theta, &gamma, &epsilon, &beta_1, &beta_2, &lambda, &seed_individuals, &best_expression, &orig_expression](int thread_num)
    {
        std::random_device rand_dev;
        // Use a combination of the device, the index, and time for maximum entropy
        unsigned int seed = rand_dev() ^ (
            (static_cast<unsigned int>(std::time(0)) << 16) |
            (static_cast<unsigned int>(thread_num)));
        std::mt19937 generator(seed); // Mersenne Twister random number generator
        Board x(depth, expression_type, data, false, cache, layers, layer_types, num_epochs, bias, eta, theta, gamma, epsilon, beta_1, beta_2, lambda);
        
        sync_point.arrive_and_wait();
        Board secondary_one((depth > 0) ? depth-1 : 0, expression_type, cache), secondary_two((depth > 0) ? depth-1 : 0, expression_type, cache); //For crossover and mutations
        float score = 0.0f, mut_prob = 0.8f, rand_mut_cross;
        constexpr int init_population = 5;
        std::vector<std::pair<std::vector<std::string>, float>> individuals = seed_individuals;
        std::pair<std::vector<std::string>, float> individual_1, individual_2;
        std::vector<std::pair<int, int>> sub_exprs_1, sub_exprs_2;
        individuals.reserve(2*init_population);
        std::vector<std::string> temp_legal_moves;
        std::uniform_int_distribution<int> rand_depth_dist(0, x.n - 1), selector_dist(0, init_population - 1);
        int rand_depth, rand_individual_idx_1, rand_individual_idx_2;
        std::uniform_real_distribution<float> rand_mut_cross_dist(0.0f, 1.0f);
        size_t temp_sz;
        
        auto updateScore = [&]()
        {
    //        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).first == x.n);
    //        assert(((x.expression_type == "prefix") ? x.getPNdepth(x.srnn.pieces) : x.getRPNdepth(x.srnn.pieces)).second);
            if (score > max_score)
            {
                max_score = score;
                std::scoped_lock str_lock(Board::thread_locker);
                best_expression = x._to_infix();
                orig_expression = x.expression();
                std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
                std::cout << "Best expression = " << best_expression << '\n';
                std::cout << "Best expression (original format) = " << orig_expression << '\n';
            }
        };
        
        //Step 1, generate init_population expressions
        for (int i = 0; i < init_population;)
        {
//                puts("hi");
            x.srnn.pieces.clear();
            while ((score = x.complete_status()) == -1) //this while-loop generates one weight-update-rule expression
            {
                temp_legal_moves = x.get_legal_moves(); //the legal moves
                
                assert(temp_legal_moves.size());
                temp_sz = temp_legal_moves.size(); //the number of legal moves
                std::uniform_int_distribution<int> distribution(0, temp_sz - 1); // A random integer generator which generates an index corresponding to an allowed move
                x.srnn.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
            }
            
            updateScore();
            if (!std::isnan(score))
            {
                individuals.push_back(std::make_pair(x.srnn.pieces, score));
                i++;
            }
        }
        
        auto Mutation = [&](int n)
        {
            //Step 1: Generate a random depth-n sub-expression `secondary_one.srnn.pieces`
            secondary_one.srnn.pieces.clear();
            sub_exprs_1.clear();
            secondary_one.n = n; //set the depth of `secondary_one.srnn.pieces` to the argument `n`
//                puts("secondary_one.n = n; done");
            while (secondary_one.complete_status() == -1)
            {
                temp_legal_moves = secondary_one.get_legal_moves();
                assert(temp_legal_moves.size() > 0);
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
            if (!sub_exprs_1.size()) //If sub_exprs_1 is empty
            {
                throw std::runtime_error("\nSecondary pieces = \n" + vec_to_str(secondary_one.srnn.pieces) + "Primary pieces = \n" + vec_to_str(x.srnn.pieces) + "\nSecondary pieces = " + std::to_string(secondary_one.srnn.pieces.size()) + "\nPrimary pieces size = " + std::to_string(x.srnn.pieces.size()));
            }
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
        
        auto Crossover = [&](int n) //depth-n trees to swap between secondary_one and secondary_two
        {
            sub_exprs_1.clear(); //stores all depth-n subtrees in secondary_one
            sub_exprs_2.clear(); //stores all depth-n subtrees in secondary_two
            secondary_one.n = n;
            secondary_two.n = n;
            
            //Picks the first random expression in the population of expressions called `individuals`
            rand_individual_idx_1 = selector_dist(generator);
            individual_1 = individuals[rand_individual_idx_1];
            
            //Picks the second random expression in the population of expressions called `individuals`
            do {
                rand_individual_idx_2 = selector_dist(generator);
            } while (rand_individual_idx_2 == rand_individual_idx_1); //Make sure we don't pick the same expression
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

            x.srnn.pieces = individual_1.first; //assigning the first cross-over'd individual to the primary Board object's pieces vector
            score = x.complete_status(false); //getting the score of the first cross-over'd individual
            updateScore(); //updating the best score achieved thus far
            individuals.push_back(std::make_pair(x.srnn.pieces, score)); //adding the first cross-over'd individual to the expression population
            
            x.srnn.pieces = individual_2.first; //assigning the second cross-over'd individual to the primary Board object's pieces vector
            score = x.complete_status(false); //getting the score of the second cross-over'd individual
            updateScore(); //updating the best score achieved thus far
            individuals.push_back(std::make_pair(x.srnn.pieces, score)); //adding the first cross-over'd individual to the expression population
        };

        if (!x.srnn.pieces.size())
        {
            throw std::runtime_error("Primary pieces size = 0");
        }
        puts("Starting evolution now...");
        for (/*int ngen = 0*/; (timeElapsedSince(start_time) < time); /*ngen++*/)
        {
//            if (ngen && (ngen%5 == 0))
//            {
//                std::cout << "Unique expressions = " << Board::expression_set.size() << '\n';
//            }
            //Produce N additional individuals through crossover and mutation
            for (int n = 0; n < init_population /*size of initial population*/; n++)
            {
                //Step 1: Generate a random number between 0 and 1 called `rand_mut_cross`
                rand_mut_cross = rand_mut_cross_dist(generator);
                
                //Step 2: Generate a random uniform int from 0 to x.n - 1 called `rand_depth`
                rand_depth = rand_depth_dist(generator); //depth of expression(s) to perform mutation or crossover with
                
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
            [](std::pair<std::vector<std::string>, float>& individual_1, std::pair<std::vector<std::string>, float>& individual_2)
            {
                return individual_1.second > individual_2.second;
            }); //sorts the individuals in the population from highest to lowest score (so highest score -> first element, second highest score -> second element, etc.)
            individuals.resize(init_population); //keep only the best `init_population` individuals.
        }
        return individuals;
    };
    
//        for (unsigned int i = 0; i < num_threads; i++)
//        {
//            threads[i] = std::thread(func, i+1);
//        }
//
//        for (unsigned int i = 0; i < num_threads; i++)
//        {
//            threads[i].join();
//        }
    
    std::vector<std::future<std::vector<std::pair<std::vector<std::string>, float>>>> futures;

    for (unsigned int i = 0; i < num_threads; i++)
    {
        futures.push_back(std::async(std::launch::async, func, i + 1));
    }
    
    std::vector<std::pair<std::vector<std::string>, float>> allIndividuals;
    
    for (unsigned int i = 0; i < num_threads; i++) //looping over each thread
    {
        std::vector<std::pair<std::vector<std::string>, float>> result = futures[i].get();  // blocks until ready -> get the population of thread i
        // use result
        for (const auto& ind_pair: result) //push-back all individuals of thread-i's population into allIndividuals
        {
            allIndividuals.push_back(ind_pair);
        }
    }
    
    std::cout << "\nUnique expressions = " << Board::expression_set.size() << '\n';
    std::cout << "Time spent fitting = " << Board::fit_time << " seconds\n";
    std::cout << "Best score = " << max_score << ", MSE = " << (1/max_score)-1 << '\n';
    std::cout << "Best expression = " << best_expression << '\n';
    std::cout << "Best expression (original format) = " << orig_expression << '\n';
    
     return allIndividuals;
}

int main()
{
    const std::unordered_map<std::string, float (*)(const Eigen::VectorXf&)> func_map =
    {
        {"Hemberg_1", Hemberg_1},
        {"Hemberg_2", Hemberg_2},
        {"Hemberg_3", Hemberg_3},
        {"Hemberg_4", Hemberg_4},
        {"Hemberg_5", Hemberg_5},
        {"Feynman_1", Feynman_1},
        {"Feynman_2", Feynman_2},
        {"Feynman_3", Feynman_3},
        {"Feynman_4", Feynman_4},
        {"Feynman_5", Feynman_5},
    };
    constexpr const char* filename = "temp_config.txt";
    constexpr const char* tempMSE_filename = "MSE_temp.txt";
    std::string tempInpBuf;

    std::vector<int> layers;
    std::deque<std::string> layer_types;
    std::string benchmark_type, weight_update_rule;
    float eta, theta, gamma, epsilon, beta_1, beta_2, lambda;
    
    std::ifstream finObj(filename);
    //1. read in `layers`
    std::getline(finObj, tempInpBuf);
    std::stringstream ss(tempInpBuf);
    std::string token;
    while (ss >> token)
    {
        layers.push_back(std::stoi(token));
    }
//    std::cout << "\nlayers = " << layers;
    //2. read in layer types
    std::getline(finObj, tempInpBuf);
    ss = std::stringstream(tempInpBuf);
    while (ss >> token)
    {
        layer_types.push_back(token);
    }
//    std::cout << "\nlayer-types = " << layer_types << '\n';
    //3. read in benchmark type
    std::getline(finObj, benchmark_type);
//    std::cout << "benchmark_type = " << benchmark_type << ", size(benchmark_type) = " << benchmark_type.length() << '\n';
    //4. read in weight-update rule
    std::getline(finObj, weight_update_rule);
//    std::cout << "weight_update_rule = " << weight_update_rule << ", size(weight_update_rule) = " << weight_update_rule.length() << '\n';
    //5. Read in eta, theta, gamma, epsilon, beta_1, beta_2, lambda
    finObj >> token;
    eta = std::stof(token);
    finObj >> token;
    theta = std::stof(token);
    finObj >> token;
    gamma = std::stof(token);
    finObj >> token;
    epsilon = std::stof(token);
    finObj >> token;
    beta_1 = std::stof(token);
    finObj >> token;
    beta_2 = std::stof(token);
    finObj >> token;
    lambda = std::stof(token);
//    std::cout << "eta = " << eta
//    << "\ntheta = " << theta
//    << "\ngamma = " << gamma
//    << "\nepsilon = " << epsilon
//    << "\nbeta_1 = " << beta_1
//    << "\nbeta_2 = " << beta_2
//    << "\nlambda = " << lambda
//    << '\n';

    auto start_time = Clock::now();
    MultiLayerPerceptron mlp(
         layers,
         layer_types,
         /* bias = */ 1.0f,
         /*eta = */ eta,
         /*theta = */ theta,
         /*gamma = */ gamma,
         /*weight_update = */ weight_update_rule,
         /*expression_type = */ "prefix", //IRRELEVANT
         /*float epsilon = */ epsilon,
         /*float beta_1 = */ beta_1,
         /*float beta_2 = */ beta_2,
         /*float lambda = */ lambda /*weight decay AdamW*/);
    
    Eigen::MatrixXf my_temp_test_data = generateData(20 /*rows*/, layers[0]+1 /*columns*/, func_map.at(benchmark_type) /*function of two variables to compute the values for the third column*/, -3.0f, 3.0f);
    Data my_test_data;
    my_test_data = my_temp_test_data;
//    my_test_data.print();
    float MSE = mlp.train(my_test_data.rows, my_test_data.labels, 10);
//    std::cout << "\nFINAL MSE = " << MSE << '\n';
    std::ofstream tempMSE(tempMSE_filename);
    tempMSE << MSE << '\n';
    tempMSE.close();
//    system((std::string("cat ")+tempMSE_filename).c_str());
    exit(1);

    
//    MultiLayerPerceptron mlp(
//         std::vector<int>{2,10,9,8,10,8,1},
//         std::deque<std::string>{"sigmoid", "sigmoid", "sigmoid", "none", "none", "none"},
//         /* bias = */ 1.0f,
//         /*eta = */ 0.0001f,
//         /*theta = */ 0.8f,
//         /*gamma = */ 0.9f,
//         /*weight_update = */ "NAG",
//         /*expression_type = */ "prefix", //IRRELEVANT
//         /*float epsilon = */ 0.1f,
//         /*float beta_1 = */ 0.9f,
//         /*float beta_2 = */ 0.999f,
//         /*float lambda = */ 0.01f /*weight decay AdamW*/);

//    GP(generateData(20 /*rows*/, 3 /*columns*/, func_map.at("Hemberg_2") /*function of two variables to compute the values for the third column*/, -3.0f, 3.0f),
//       5 /*fixed depth*/,
//       "postfix",
//       true /*cache*/,
//       100 /*time to run the algorithm in seconds*/,
//       "Hemberg_1PreRandomSearchMultiThread.txt" /*name of file to save the results to*/,
//       0 /*num threads*/,
//       {2,10,5,5,1} /*Neural Network number of perceptrons in i'th layers; first layer is number of inputs (input-layer) */,
//       std::deque<std::string>{"sigmoid", "sigmoid", "none", "none"},
//       10 /*num_epochs*/,
//       /* bias = */ 1.0f,
//       /*eta = */ 0.5f,
//       /*theta = */ 0.01f,
//       /*gamma = */ 0.9f,
//       /*epsilon = */ 1e-8f,
//       /*beta_1 = */ 0.9f,
//       /*beta_2 = */ 0.999f,
//       /*lambda = */ 0.01f);
    
    std::cout << "Time Elapsed = " << timeElapsedSince(start_time) << " seconds" << '\n';
    
    return 0;
}
//git push --set-upstream origin NeuralNetworkWeightUpdate
/*
 Outline for NeuralNetworkWeightUpdate
 
 ===============
 NUM_EPOCHS = 10
 ===============
 1. 10 benchmarks -> 1410 established weight-update rule configs
    A. For each benchmark, 3 neural nets -> 141 established weight-update rule configs
        I.  Each neural net has 5, 6, 7 layers (including the input layer) with N inputs and 1 output
            a. Neural net 1: {N, {2, "sigmoid"}, {7, "sigmoid"}, {6, "sigmoid"}, {1, "none"}}
                i. Established Weight-Update Rules -> 5+6+6+6+8+4+6+6 = 47 configs
                    - Gradient-Descent: η ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} -> 5 configs
                    - Heavy Ball: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3} -> 6 configs
                    - Nesterov: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3} -> 6 configs
                    - AdaGrad: ε ∈ {1e-8, 1e-6}, η ∈ {1e-3, 3e-3, 1e-2} -> 6 configs
                    - RMSProp: ε ∈ {1e-8}, η ∈ {1e-5, 3e-5, 1e-4, 3e-4}, Ɣ ∈ {0.9, 0.99} -> 8 configs
                    - AdaDelta: ε ∈ {1e-6, 1e-8}, Ɣ ∈ {0.95, 0.99} -> 4 configs
                    - Adam: η ∈ {1e-5, 3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9, 0.95}, β_2 ∈ {0.999} -> 6 configs
                    - AdamW: λ ∈ {1e-5, 1e-4, 1e-3}, η ∈ {3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9}, β_2 ∈ {0.999} -> 6 configs
                ii. SR: start with empty population and continue until N SR-updates-rules that perform 80% of the best established weight-update rule -> returns individuals ("last population")
            b. Neural net 2: {N, {6, "sigmoid"}, {8, "sigmoid"}, {1, "sigmoid}, {5, "none"}, {1, "none"}}
                i. Established Weight-Update Rules
                    - Gradient-Descent: η ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
                    - Heavy Ball: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - Nesterov: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - AdaGrad: ε ∈ {1e-8, 1e-6}, η ∈ {1e-3, 3e-3, 1e-2}
                    - RMSProp: ε ∈ {1e-8}, η ∈ {1e-5, 3e-5, 1e-4, 3e-4}, Ɣ ∈ {0.9, 0.99}
                    - AdaDelta: ε ∈ {1e-6, 1e-8}, Ɣ ∈ {0.95, 0.99}
                    - Adam: η ∈ {1e-5, 3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9, 0.95}, β_2 ∈ {0.999}
                    - AdamW: λ ∈ {1e-5, 1e-4, 1e-3}, η ∈ {3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9}, β_2 ∈ {0.999}
                ii. SR: start with last population and continue until N SR-updates-rules that perform 80% of the best established weight-update rule -> returns individuals
            c. Neural net 3: {N, {10, "sigmoid}, {9, "sigmoid"}, {8, "sigmoid"}, {10, "none"}, {8, "none"}, {1, "none"}}
                i. Established Weight-Update Rules
                    - Gradient-Descent: η ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}
                    - Heavy Ball: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - Nesterov: θ ∈ {0.8, 0.9}, η ∈ {1e-4, 3e-4, 1e-3}
                    - AdaGrad: ε ∈ {1e-8, 1e-6}, η ∈ {1e-3, 3e-3, 1e-2}
                    - RMSProp: ε ∈ {1e-8}, η ∈ {1e-5, 3e-5, 1e-4, 3e-4}, Ɣ ∈ {0.9, 0.99}
                    - AdaDelta: ε ∈ {1e-6, 1e-8}, Ɣ ∈ {0.95, 0.99}
                    - Adam: η ∈ {1e-5, 3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9, 0.95}, β_2 ∈ {0.999}
                    - AdamW: λ ∈ {1e-5, 1e-4, 1e-3}, η ∈ {3e-5, 1e-4}, ε ∈ {1e-8}, β_1 ∈ {0.9}, β_2 ∈ {0.999}
                ii. SR: start with last population and continue until N SR-updates-rules that perform 80% of the best established weight-update rule -> returns individuals

 Option 1 (Key: Depth, Value: NumThreads): {1: 1, 5: 1, 2: 2, 4: 2, 3: 2}
 Option 2 (Fix Depth to 5): `std::thread::hardware_concurrency()` threads on depth 5 and simplify all expressions => leaning towards this option
 
 */

//g++ -Wall -std=c++20 -o NeuralNetworks_VecSR NeuralNetworks_VecSR.cpp MLP_Vec.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//g++ -Wall -std=c++20 -o NeuralNetworks_VecSR NeuralNetworks_VecSR.cpp MLP_Vec.cpp -g -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

//1                  1         0
//
//x x - => x-x                 0
//x x + => x+x     x x +
