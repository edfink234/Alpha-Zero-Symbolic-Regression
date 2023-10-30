//TODO: Explore pybind11: https://pybind11.readthedocs.io/en/stable/basics.html
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
#include <cassert>
#include <pybind11/pybind11.h>
#include <Python.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <LBFGS.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/LevenbergMarquardt>

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
    const long num_columns, num_rows;
    
public:
    
    Data(const Eigen::MatrixXf& theData) : data{theData}, num_columns{data.cols()}, num_rows{data.rows()}
    {
//        auto num_columns = data.cols(); //number of columns - 1
//        auto num_rows = data.rows();
        
        for (size_t i = 0; i < num_columns - 1; i++) //for each column
        {
            features["x"+std::to_string(i)] = Eigen::VectorXf(num_rows);
            for (size_t j = 0; j < num_rows; j++)
            {
                features["x"+std::to_string(i)](j) = data(j,i);
            }
        }
        
        features["y"] = Eigen::VectorXf(num_rows);
        rows.resize(num_rows);
        
        for (size_t i = 0; i < num_rows; i++)
        {
            features["y"](i) = data(i,num_columns - 1);
            rows[i] = data.row(i);
        }
        
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
    static std::string inline best_expression = "";
    static std::unordered_map<std::string, std::pair<Eigen::VectorXf, int>> inline expression_dict = {};
    static float inline best_loss = FLT_MAX;
    static std::vector<std::string> inline init_expression = {};
    static float inline fit_time = 0;
    
    static constexpr float K = 0.0884956f;
    static constexpr float phi_1 = 2.8f;
    static constexpr float phi_2 = 1.3f;
    int __num_features;
    std::vector<std::string> __input_vars;
    std::vector<std::string> __unary_operators;
    std::vector<std::string> __binary_operators;
    std::vector<std::string> __operators;
    std::vector<std::string> __other_tokens;
    std::vector<std::string> __tokens;
    std::vector<float> __tokens_float;
    Eigen::VectorXf* params; //store the parameters of the expression of the current episode after it's completed
    Data data;
    
    std::vector<float> __operators_float;
    std::vector<float> __unary_operators_float;
    std::vector<float> __binary_operators_float;
    std::vector<float> __input_vars_float;
    std::vector<float> __other_tokens_float;
    
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> vel_dist, pos_dist;
    
    int action_size;
    size_t reserve_amount;
    int num_fit_iter;
    std::string fit_method;
                
    std::unordered_map<float, std::string> __tokens_dict; //Converts number to string
    std::unordered_map<std::string, float> __tokens_inv_dict; //Converts string to number

    int n; //depth of RPN tree
    std::string expression_type;
    // Create the empty expression list.
    std::vector<float> pieces;
    bool visualize_exploration;
    
    Board(const Eigen::MatrixXf& theData, int n = 3, const std::string& expression_type = "prefix", bool visualize_exploration = false, std::string fitMethod = "PSO", int numFitIter = 1) : data{theData}, gen{rd()}, vel_dist{-1.0f, 1.0f}, pos_dist{0.0f, 1.0f}, fit_method{fitMethod}, num_fit_iter{numFitIter}
    {
        this->__num_features = data[0].size() - 1;
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
        this->reserve_amount = 2*pow(2,this->n)-1;
        
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
        //TODO: Generated Expressions should always be irreducible, this isn't GP!
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
            
            if ((num_binary != num_leaves - 1))//  && ((std::find(this->__other_tokens_float.begin(), this->__other_tokens_float.end(), pieces.back()) == this->__other_tokens_float.end()) ||  (*(pieces.end()-1) != *(pieces.end()-2))))
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
                
//                string_pieces.back() = __other_tokens[0];
//                string_pieces.reserve(string_pieces.size() + 3);
//                string_pieces.push_back(__input_vars[0]);
//                string_pieces.push_back(__binary_operators[0]);
//                string_pieces.push_back(__binary_operators[0]);
//                //1.94696, 15.9098, 0.0949083, 6.13413, 2.26467
//                //9.57711, 2.6431, 4.44798, 1.03814, 1.33934

//                if (getRPNdepth(string_pieces).first <= this->n) //Can only add constant if
//                {
//                    temp.insert(temp.end(), this->__other_tokens_float.begin(), this->__other_tokens_float.end());
//                }
                temp.insert(temp.end(), this->__other_tokens_float.begin(), this->__other_tokens_float.end());
            }
            return temp;
        }
    }
    
    //Returns a string form of the expression stored in the vector<float> attribute pieces
    std::string expression()
    {
        std::string temp;
        temp.reserve(2*pieces.size());
        size_t sz = pieces.size() - 1;
        int const_index = ((expression_type == "postfix") ? 0 : params->size()-1);
        for (size_t i = 0; i <= sz; i++)
        {
            if (std::find(__other_tokens_float.begin(), __other_tokens_float.end(), pieces[i]) != __other_tokens_float.end())
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
                temp += ((i!=sz) ? __tokens_dict[pieces[i]] + " " : __tokens_dict[pieces[i]]);
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
            std::string token = __tokens_dict[pieces[i]];

            if (std::find(__operators_float.begin(), __operators_float.end(), pieces[i]) == __operators_float.end()) // leaf
            {
                stack.push(((!show_consts) || (std::find(__other_tokens.begin(), __other_tokens.end(), token) == __other_tokens.end())) ? token : std::to_string((*params)(const_counter++)));
            }
            else if (std::find(__unary_operators_float.begin(), __unary_operators_float.end(), pieces[i]) != __unary_operators_float.end()) // Unary operator
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
            std::string token = __tokens_dict[pieces[i]];

            if (std::find(__operators_float.begin(), __operators_float.end(), pieces[i]) == __operators_float.end()) // leaf
            {
                if (token == "const")
                {
                    stack.push(Eigen::VectorXf::Ones(data.numRows())*params(const_count++));
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
    
    float operator()(Eigen::VectorXf& x, Eigen::VectorXf& grad)
    {
        float mse = MSE(expression_evaluator(x), data["y"]), low_b, temp;
        for (int i = 0; i < x.size(); i++)
        {
//            x(i) += 0.00001f;
//            grad(i) = (MSE(expression_evaluator(x), data["y"]) - mse) / 0.00001f ;
//            
////            printf("grad[%d] = %f\n",i,grad[i]);
//            x(i) -= 0.00001f;
            temp = x(i);
            x(i) -= 0.00001f;
            low_b = MSE(expression_evaluator(x), data["y"]);
            x(i) += 0.00002f;
            grad(i) = (MSE(expression_evaluator(x), data["y"]) - low_b) / 0.00002f ;

            //            printf("grad[%d] = %f\n",i,grad[i]);
            x(i) = temp;
        }
        
        return mse;
    }
    
    void LBFGS()
    {
        auto start_time = Clock::now();
        LBFGSpp::LBFGSParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = this->num_fit_iter;
        LBFGSpp::LBFGSSolver<float> solver(param);
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
    
    //TODO: Add other methods besides PSO (e.g. Eigen::LevenbergMarquardt, calling scipy from the Python-C API, etc.)
    float fitFunctionToData(std::string method = "PSO")
    {
        if (params->size())
        {
            if (method == "PSO")
            {
                PSO();
            }
            else if (method == "AsyncPSO")
            {
                AsyncPSO();
            }
            else if (method == "LBFGS")
            {
                LBFGS();
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
            Board::expression_dict[expression_string].second++;
            if (visualize_exploration)
            {
                //TODO: call some plotting function, e.g. ROOT CERN plotting API, Matplotlib from the Python-C API, Plotly if we want a web application for this, etc. The plotting function could also have the fitted constants (rounded of course), but then this if statement would need to be moved down to below the fitFunctionToData call in this `complete_status` method.
            }
            
            this->params = &Board::expression_dict[expression_string].first;
            if (!(this->params->size()))
            {
                this->params->resize(num_consts);
                this->params->setOnes();
            }

            return fitFunctionToData(this->fit_method);
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
};

// 2.5382*cos(x_3) + x_0^2 - 0.5
// postfix = "const x3 cos * x0 x0 * const - +"
// prefix = "+ * const cos x3 - * x0 x0 const"
float exampleFunc(const Eigen::VectorXf& x)
{
//    return 2.5382*cos(x[3]) + (x[0]*x[0]) - 0.5f;
    return 5*cos(x[1]+x[3])+x[4];
}

void MCTS(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "PSO", int num_fit_iter = 1)
{
    Board x(data, depth, expression_type, false, method, num_fit_iter);
    std::cout << x.data << '\n';
    
    float score = 0, max_score = 0, check_point_score = 0, UCT, best_act, UCT_best;
    std::vector<float> temp_legal_moves;
    std::unordered_map<std::string, std::unordered_map<float, float>> Qsa, Nsa;
    std::string state;
    std::string expression, orig_expression, best_expression;
    std::unordered_map<std::string, float> Ns;
    float c = 1.4; //"controls the balance between exploration and exploitation", see equation 2 here: https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/uct.pdf
    std::vector<std::pair<std::string, float>> moveTracker;
    moveTracker.reserve(x.reserve_amount);
    temp_legal_moves.reserve(x.reserve_amount);
    state.reserve(2*x.reserve_amount);
    
    auto getString = [](const std::vector<float>& pieces)
    {
        return std::accumulate(pieces.begin(), pieces.end(), std::string(), [](const std::string& a, float b) { return a + (a.empty() ? "" : " ") + std::to_string(b); });
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
        while ((score = x.complete_status()) == -1)
        {
            temp_legal_moves = x.get_legal_moves();
            state = std::move(getString(x.pieces)); //get string of current pieces
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
            best_expression = std::move(expression);
        }
        x.pieces.clear();
        moveTracker.clear();
    }
    
}

void RandomSearch(const Eigen::MatrixXf& data, int depth = 3, std::string expression_type = "prefix", float stop = 0.8f, std::string method = "PSO", int num_fit_iter = 1)
{
    Board x(data, depth, expression_type, false, method, num_fit_iter);

//    std::cout << x["y"] << '\n';
//    std::cout << x.fit_method << '\n';
//    std::cout << x.data << '\n';
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
            expression = x._to_infix();//((score > 0.99f) ? x._to_infix(true, true) : x._to_infix());
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
    MCTS(data, 4, "prefix", 1.0f, "LBFGS", 5);
//
    auto end_time = Clock::now();
    std::cout << "Time difference = "
          << std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()/1e9 << " seconds" << '\n';

    return 0;
}

