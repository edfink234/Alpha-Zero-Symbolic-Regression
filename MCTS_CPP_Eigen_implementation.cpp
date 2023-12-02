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
                
    static std::unordered_map<float, std::string> inline __tokens_dict; //Converts number to string
    static std::unordered_map<std::string, float> inline __tokens_inv_dict; //Converts string to number

    int n; //depth of RPN tree, TODO: change to short
    std::string expression_type;
    // Create the empty expression list.
    std::vector<float> pieces;
    bool visualize_exploration, is_primary;
    
    Board(bool primary = true, int n = 3, const std::string& expression_type = "prefix", std::string fitMethod = "PSO", int numFitIter = 1, std::string fitGradMethod = "naive_numerical", const Eigen::MatrixXf& theData = {}, bool visualize_exploration = false) : is_primary{primary}, gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, fit_method{fitMethod}, num_fit_iter{numFitIter}, fit_grad_method{fitGradMethod}
    {
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
            Board::__operators = {"cos", "+", "-", "*"};
            Board::__other_tokens = {"const"};
            Board::__tokens = {"cos", "+", "-", "*"};
            
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
        }

        this->n = n;
        this->expression_type = expression_type;
        this->pieces = {};
        this->reserve_amount = 2*pow(2,this->n)-1;
        this->visualize_exploration = visualize_exploration;
        this->pieces.reserve(reserve_amount);
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
        
    std::pair<int, bool> getPNdepth(const std::vector<float>& expression)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }

        std::vector<int> stack;
        int depth = 0, num_binary = 0, num_leaves = 0;
        
        for (float val : expression)
        {
            if (is_binary(val))
            {
                stack.push_back(2);  // Number of operands
                num_binary++;
            }
            else if (is_unary(val))
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
    
    std::pair<int, bool> getRPNdepth(const std::vector<float>& expression)
    {
        if (expression.empty())
        {
            return std::make_pair(0, false);
        }

        std::stack<int> stack;
        bool complete = true;

        for (float token : expression)
        {
            if (is_unary(token))
            {
                stack.top() += 1;
            }
            else if (is_operator(token))
            {
                int op2 = std::move(stack.top());
                stack.pop();
                int op1 = std::move(stack.top());
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
            int op2 = std::move(stack.top());
            stack.pop();
            int op1 = std::move(stack.top());
            stack.pop();
            stack.push(std::max(op1, op2) + 1);
            complete = false;
        }

        return std::make_pair(stack.top() - 1, complete);
    }

    
    std::vector<float> get_legal_moves()
    {
        if (this->expression_type == "prefix")
        {
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the operators
            {
                return Board::__operators_float;
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
            
            //__tokens_dict: converts float to string
            std::vector<float> _pieces(pieces);
            
            _pieces.push_back(Board::__binary_operators_float[0]);
            
            std::vector<float> temp;
            
            if (getPNdepth(_pieces).first <= this->n)
            {
                temp.insert(temp.end(), Board::__binary_operators_float.begin(), Board::__binary_operators_float.end());
            }
            _pieces[_pieces.size() - 1] = Board::__unary_operators_float[0];
            if (getPNdepth(_pieces).first <= this->n)
            {
                temp.insert(temp.end(), Board::__unary_operators_float.begin(), Board::__unary_operators_float.end());
            }

            _pieces[_pieces.size() - 1] = Board::__input_vars_float[0];
            //The number of leaves can never exceed number of binary + 1 in any RPN expression
            if (!((num_leaves == num_binary + 1) || (getPNdepth(_pieces).first < this->n && (num_leaves == num_binary))))
            {
                temp.insert(temp.end(), Board::__input_vars_float.begin(), Board::__input_vars_float.end()); //leaves allowed
                if (std::find(Board::__unary_operators_float.begin(), Board::__unary_operators_float.end(), _pieces[_pieces.size()-2]) == Board::__unary_operators_float.end())
                {
                    temp.insert(temp.end(), Board::__other_tokens_float.begin(), Board::__other_tokens_float.end());
                }
            }
            return temp;
        }
        else //postfix
        {
            
            std::vector<float> temp;
            if (this->pieces.empty()) //At the beginning, self.pieces is empty, so the only legal moves are the features and const
            {
                temp.insert(temp.end(), Board::__input_vars_float.begin(), Board::__input_vars_float.end());
                temp.insert(temp.end(), Board::__other_tokens_float.begin(), Board::__other_tokens_float.end());
                return temp;
            }
            int num_binary = this->__num_binary_ops();
            int num_leaves = this->__num_leaves();
            
            if ((num_binary != num_leaves - 1))//  && ((std::find(Board::__other_tokens_float.begin(), Board::__other_tokens_float.end(), pieces.back()) == Board::__other_tokens_float.end()) ||  (*(pieces.end()-1) != *(pieces.end()-2))))
            {

                temp.insert(temp.end(), Board::__binary_operators_float.begin(), Board::__binary_operators_float.end());
            }
            std::vector<float> _pieces(pieces);

            _pieces.push_back(Board::__unary_operators_float[0]);
            if ((num_leaves >= 1) && (getRPNdepth(_pieces).first <= this->n) && (std::find(Board::__other_tokens_float.begin(), Board::__other_tokens_float.end(), _pieces[_pieces.size()-2]) == Board::__other_tokens_float.end())) //unary_op(const) is not allowed
            {
                temp.insert(temp.end(), Board::__unary_operators_float.begin(), Board::__unary_operators_float.end());
            }
            _pieces[_pieces.size() - 1] = Board::__input_vars_float[0];
            if (getRPNdepth(_pieces).first <= this->n)
            {
                temp.insert(temp.end(), Board::__input_vars_float.begin(), Board::__input_vars_float.end());
                temp.insert(temp.end(), Board::__other_tokens_float.begin(), Board::__other_tokens_float.end());
            }
            return temp;
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
    checks if it is a complete RPN expression.
    Returns the score of the expression if complete,
    where 0 <= score <= 1 and -1 if not complete or if
    the desired depth has not been reached.
    */
    float complete_status()
    {
        auto [depth, complete] =  ((expression_type == "prefix") ? getPNdepth(pieces) : getRPNdepth(pieces)); //structured binding :)

        if (!complete || depth < this->n) //Expression not complete
        {
            return -1;
        }
        else
        {
            std::string expression_string;
            expression_string.reserve(8*pieces.size());
            
            for (float i: pieces){expression_string += std::to_string(i)+" ";}
            
            if (visualize_exploration)
            {
                //TODO: call some plotting function, e.g. ROOT CERN plotting API, Matplotlib from the Python-C API, Plotly if we want a web application for this, etc. The plotting function could also have the fitted constants (rounded of course), but then this if statement would need to be moved down to below the fitFunctionToData call in this `complete_status` method.
            }
            
            if (is_primary)
            {
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
    void GB(int z, int& ind)
    {
        do
        {
            ind = ((expression_type == "prefix") ? ind+1 : ind-1);
            if (is_unary(this->pieces[ind]))
            {
                GB(1, ind);
            }
            else if (is_binary(this->pieces[ind]))
            {
                GB(2, ind);
            }
            --z;
        } while (z);
    }
    
    //Computes the grasp of an arbitrary element pieces[i],
    //from https://www.jstor.org/stable/43998756 (bottom of pg. 165)
    int GR(int i)
    {
        int start = i;
        int& ptr_lgb = start;
        if (is_unary(this->pieces[i]))
        {
            GB(1, ptr_lgb);
        }
        else if (is_binary(this->pieces[i]))
        {
            GB(2, ptr_lgb);
        }
        return ((expression_type == "prefix") ? ( ptr_lgb - i) : (i - ptr_lgb));
    }
    
    void get_indices(std::vector<std::pair<int, int>>& sub_exprs, std::pair<std::vector<float>, float>& individual)
    {
        int temp;
        for (size_t k = 0; k < individual.first.size(); k++)
        {
            temp = k;
            if (is_unary(individual.first[k]))
            {
                int& ptr_GB = temp;
                GB(1, ptr_GB);
                sub_exprs.push_back(std::make_pair( ((k < ptr_GB) ? k: ptr_GB), ((k > ptr_GB) ? k: ptr_GB)));
            }
            else if (is_binary(individual.first[k]))
            {
                int& ptr_GB = temp;
                GB(2, ptr_GB);
                sub_exprs.push_back(std::make_pair( ((k < ptr_GB) ? k: ptr_GB), ((k > ptr_GB) ? k: ptr_GB)));
            }
            else
            {
                sub_exprs.push_back(std::make_pair(k, k));
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

void GP(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical")
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false);
    Board secondary_one(false, 0, expression_type), secondary_two(false, 0, expression_type); //For crossover and mutations
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
    
    //Step 1, generate init_population expressions
    for (int i = 0; i < init_population; i++)
    {
        while ((score = x.complete_status()) == -1)
        {
            temp_legal_moves = x.get_legal_moves(); //the legal moves
            temp_sz = temp_legal_moves.size(); //the number of legal moves
            std::uniform_int_distribution<int> distribution(0, temp_sz - 1);
 // A random integer generator which generates an index corresponding to an allowed move

            x.pieces.push_back(temp_legal_moves[distribution(generator)]); //make the randomly chosen valid move
        }
        
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
        
        //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
        //in `individual` and store them in an std::vector<std::pair<int, int>>
        //called `sub_exprs_1`.
        individual_1 = individuals[selector_dist(generator)];
        secondary_one.get_indices(sub_exprs_1, individual_1);
        
        //Step 3: Generate a uniform int from 0 to sub_exprs.size() - 1 called `mut_ind`
        //...
        
        //Step 4: Substitute sub_exprs[mut_ind] in individual with mut_sub_expr
        //...
        
    };
    
    auto Crossover = [&](int n)
    {
        secondary_one.pieces.clear();
        secondary_two.pieces.clear();
        secondary_one.n = n;
        secondary_two.n = n;
        
        while (secondary_one.complete_status() == -1)
        {
            temp_legal_moves = secondary_one.get_legal_moves();
            std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
            secondary_one.pieces.push_back(temp_legal_moves[distribution(generator)]);
        }
        
        while (secondary_two.complete_status() == -1)
        {
            temp_legal_moves = secondary_two.get_legal_moves();
            std::uniform_int_distribution<int> distribution(0, temp_legal_moves.size() - 1);
            secondary_two.pieces.push_back(temp_legal_moves[distribution(generator)]);
        }
        
        rand_individual_idx_1 = selector_dist(generator);
        individual_1 = individuals[rand_individual_idx_1];
        
        do {
            rand_individual_idx_2 = selector_dist(generator);
        } while (rand_individual_idx_2 == rand_individual_idx_1);
        individual_2 = individuals[rand_individual_idx_2];
    
        //Step 1: Identify the starting and stopping index pairs of all depth-n sub-expressions
        //in `individual_1` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_1`.
        //...
        secondary_one.get_indices(sub_exprs_1, individual_1);
        
        //Step 2: Identify the starting and stopping index pairs of all depth-n sub-expressions
        //in `individual_2` and store them in an std::vector<std::pair<int, int>> called `sub_exprs_2`.
        //...
        secondary_two.get_indices(sub_exprs_2, individual_2);
        
        //Step 3: Generate a random uniform int from 0 to sub_exprs_1.size() - 1 called `mut_ind_1`
        //...
        
        //Step 4: Generate a random uniform int from 0 to sub_exprs_2.size() - 1 called `mut_ind_2`
        //...
        
        //Step 5: Swap sub_exprs_1[mut_ind_1] in individual_1 with sub_exprs_2[mut_ind_2] in individual_2
        //...
    };
    
    for (int ngen = 0; score < stop; ngen++)
    {
        //Produce N additional individuals through crossover and mutation
        
        //Step 1: Generate a random number between 0 and 1 called `rand_mut_cross`
        rand_mut_cross = rand_mut_cross_dist(generator);
        
        //Step 2: Generate a random uniform int from 0 to x.n - 1 called `rand_depth`
        rand_depth = rand_depth_dist(generator);
              
        //Step 3:
        
        //Step 4: Select mutation if 0 <= rand_mut_cross <= mut_prob, else select crossover
        
        //Step 5: Call functions
        
        
    }

}

void PSO(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical")
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false);
    
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
    std::unordered_map<float, std::unordered_map<int, int>> Nsa;
    std::unordered_map<float, std::unordered_map<int, float>> Psa;
    std::unordered_map<int, float> p_i_vals, p_i;
    
    /*
     Idea: In this implementation of PSO,
     
     The traditional PSO initializes the particle positions to be between 0 and 1. However, for my application,
     the particle positions are
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
                std::cout << "c: " << c << " -> ";
                c += x.vel_dist(generator);
                std::cout << c << '\n';
            }
            else
            {
                std::cout << "c: " << c << " -> ";
                c = 0.0f; //if new best found, reset c and try to exploit the new best
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

void MCTS(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical")
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false);
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

void RandomSearch(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "LevenbergMarquardt", int num_fit_iter = 1, const std::string& fit_grad_method = "naive_numerical")
{
    Board x(true, depth, expression_type, method, num_fit_iter, fit_grad_method, data, false);
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
    for (int i = 0; (score < stop/* && Board::expression_dict.size() <= 2000000*/); i++)
    {
        while ((score = x.complete_status()) == -1)
        {
            temp_legal_moves = x.get_legal_moves(); //the legal moves
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
//    MCTS(data, 3, "postfix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical");
    PSO(data, 3, "postfix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical");
//    RandomSearch(data, 3, "postfix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical");
//    GP(data, 3, "prefix", 1.0f, "LevenbergMarquardt", 5, "naive_numerical");
    auto end_time = Clock::now();
    std::cout << "Time difference = "
          << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()/1e9 << " seconds" << '\n';

    return 0;
}
