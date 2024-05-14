#include "MLP_Vec.h"
#include <cassert>
#include <stack>

//open /Users/edwardfinkelstein/LinkedIn/Ex_Files_Neural_Networks/Exercise\ Files/03_03/NeuralNetworks/*cpp /Users/edwardfinkelstein/LinkedIn/Ex_Files_Neural_Networks/Exercise\ Files/03_03/NeuralNetworks/*h
//cd /Users/edwardfinkelstein/LinkedIn/Ex_Files_Neural_Networks/Exercise\ Files/03_03/NeuralNetworks/

// Return a new Perceptron object with the specified number of inputs
Perceptron::Perceptron(int inputs, float bias, bool is_output, std::string&& output_type)
{
    this->bias = bias;
    this->weights.resize(inputs);

    // Use Eigen's random number generation to initialize the weights
    this->weights = Eigen::VectorXf::Random(inputs);
    this->is_output = is_output;
    this->output_type = output_type;
}

// Run the perceptron. x is a vector with the input values.
float Perceptron::run(const Eigen::VectorXf& x)
{
    assert(x.size() == weights.size());
    float dot_product = x.dot(weights) + bias;
    return (is_output && output_type != "sigmoid") ? dot_product : sigmoid(dot_product);
}

// Set the weights. w_init is a vector with the weights.
void Perceptron::set_weights(const Eigen::VectorXf& w_init)
{
    if (w_init.size() == 0)
    {
        // Handle empty input vector case
        return;
    }

    // Copy all elements except the last one to the weights vector
    this->weights.head(w_init.size() - 1) = w_init.head(w_init.size() - 1);

    // Set the last element of w_init as the bias
    this->bias = w_init(w_init.size() - 1);
}

// Evaluate the sigmoid function for the floating point input x.
float Perceptron::sigmoid(float x)
{
    return 1.0f/(1.0f + exp(-x));
}

float Perceptron::scale_between(float unscaled_num, float min, float max, float max_allowed, float min_allowed)
{
    return (max_allowed - min_allowed) *
    (unscaled_num - min) / (max - min) + min_allowed;
}

// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(std::vector<int> layers, float bias, float eta, std::string&& output_type, const std::string& expression_type, const std::string& weight_update)
{
    // Set up the signal handler
    signal(SIGINT, signalHandler);
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;
    this->output_type = output_type;
    this->expression_type = expression_type;
    this->weight_update = weight_update;
    
    size_t mlp_sz = this->layers.size();

    for (int i = 0; i < mlp_sz; i++) //for each layer
    {
        this->values.emplace_back(Eigen::VectorXf::Zero(layers[i])); //add vector of values
        this->network.emplace_back(); //add vector of neurons
        this->d.emplace_back(Eigen::VectorXf::Zero(layers[i]));
        if (i > 0)  //network[0] is the input layer, so it has no neurons, i.e., it's empty
        {
            for (int j = 0; j < this->layers[i]; j++)
            {
                bool last_layer = (i == mlp_sz - 1);
                this->network[i].emplace_back(this->layers[i-1], this->bias, last_layer, last_layer ? output_type : "none");
            }
        }
    }
}


void MultiLayerPerceptron::set_weights(std::vector<Eigen::MatrixXf>&& w_init)
{
    // Write all the weights into the neural network.
    // w_init is a vector of vectors of vectors of floats.
    for (int i = 1; i < network.size(); i++)
    { //first layer is the input layer so they're no neurons there
        for (int j = 0; j < layers[i]; j++)
        { //for each neuron
            network[i][j].set_weights(w_init[i-1].row(j));
        }
    }
}

void MultiLayerPerceptron::reset_weights()
{
    static std::random_device rd;  // Obtain a random number from hardware
    static std::mt19937 gen(rd()); // Seed the generator
    static std::uniform_real_distribution<> distr; // Define the range
    // Write all the weights into the neural network.
    // w_init is a vector of vectors of vectors of floats.
    for (int i = 1; i < network.size(); i++)
    { //first layer is the input layer so they're no neurons there
        for (int j = 0; j < layers[i]; j++)
        { //for each neuron
            network[i][j].weights = Eigen::VectorXf::Random(network[i][j].weights.size());
            network[i][j].bias = distr(gen);
        }
        
    }
}

void MultiLayerPerceptron::print_weights()
{
    std::cout << '\n';
    for (int i = 1; i < network.size(); i++)
    { //first layer is the input layer so they're no neurons there
        for (int j = 0; j < layers[i]; j++)
        {
            std::cout << "Layer " << i+1 << " Neuron " << j+1 << ": ";
            for (auto &it: network[i][j].weights)
            {
                std::cout << it <<"   ";
            }
            std::cout << network[i][j].bias << '\n';
        }
    }
    std::cout << '\n';
}

Eigen::VectorXf MultiLayerPerceptron::run(const Eigen::VectorXf& x)
{
    // Run an input forward through the neural network.
    // x is a vector with the input values.
    //Returns: vector with output values, i.e., the last element in the values vector
    this->values[0] = x;
    for (int i = 1; i < network.size(); i++) //for each layer
    {
        for (int j = 0; j < this->layers[i]; j++) //for each neuron
        {
            this->values[i][j] = this->network[i][j].run(this->values[i-1]);
            //run the previous layer output values through this neuron `this->network[i][j]`
        }
    }
    return this->values.back();
}

/*/
 MSE = 1/n * \sum_{i = 0}^{n-1} (y_i - o_i)^2
 */
float MultiLayerPerceptron::mse(const Eigen::VectorXf& x, const Eigen::VectorXf& y)
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    return (x - y).squaredNorm() / x.size();
}

// Run a single (x,y) pair with the backpropagation algorithm.
float MultiLayerPerceptron::bp(const Eigen::VectorXf& x, const Eigen::VectorXf& y)
{
    
    // Backpropagation Step by Step:
    
    // STEP 1: Feed a sample to the network `this->run(x)`
    // STEP 2: Calculate the MSE
    float MSE = this->mse(y, this->run(x));

    // STEP 3: Calculate the output error terms
    
//        delta_k = d MSE(y_k, o_k) / d w_k
//                = d MSE(y_k, sigmoid(x_k*w_k + b_k)) / d w_k
//                = d ((1/n) * \sum_{i = 0}^{n-1} (y_{k_{i}} - sigmoid(x_k*w_k + b_k)_{k_{i}})^2) / d w_k
//                = (1/n) * d (\sum_{i = 0}^{n-1} (y_{k_{i}} - sigmoid(x_k*w_k + b_k)_{k_{i}})^2) / d w_k
//                = d (y_{k} - sigmoid(x_k*w_k + b_k))^2) / d w_k
//                = 2 * (y_{k} - o_{k}) * d (- o_{k}) / d w_k
//                = -2 * (y_k - o_k) * o_k * (1 - o_k)
//                \propto o_k * (1 - o_k) * (y_k - o_k)
    
//                = d (y_{k} - (x_k*w_k + b_k))^2) / d w_k
//                = 2 * (y_{k} - (x_k*w_k + b_k)) * d (- (x_k*w_k + b_k)) / d w_k
//                = -2 * (y_k - (x_k*w_k + b_k)) * x_k
    //TODO: Fix this so it is correct!
    for (int i = 0; i < this->layers.back(); i++)
    {
        float o_k = this->values.back()[i];
        this->d.back()[i] = ((output_type == "sigmoid") ? (o_k * (1 - o_k) * (y[i] - o_k)) : (y[i] - o_k));//* x.sum();
    }
    
    // STEP 4: Calculate the error term of each unit on each layer
    for (int i = network.size()-2; i > 0; i--) //for each layer (starting from the one before the output layer and ending at and including the layer right before the input layer)
    {
        for (int h = 0; h < layers[i]; h++) //for each neuron in layer i
        {
            assert(layers[i] == network[i].size());
            float fwd_error = 0.0f;
            //fwd_error = \sum_{k \in \mathrm{outs}} w_{kh} \delta_k
            for (int k = 0; k < layers[i+1]; k++) //for each neuron in layer i+1
            {
                fwd_error += network[i+1][k].weights[h]*d[i+1][k];
            }
            
            //\delta_h = o_h*(1-o_h)*fwd_error
            this->d[i][h] = this->values[i][h] * (1 - this->values[i][h]) * fwd_error;
        }
    }
    
    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (int i = 1; i < network.size(); i++) //for each layer
    {
        for (int j = 0; j < layers[i]; j++)
        {
            for (int k = 0; k < layers[i-1]; k++) //weights
            {
                if (this->weight_update == "basic")
                {
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] + this->eta*this->d[i][j]*this->values[i-1][k];
                }
                else if (this->weight_update == "SR")
                {
                    this->network[i][j].weights[k] = this->expression_evaluator(this->network[i][j].weights[k], this->eta, this->d[i][j], this->values[i-1][k]);
                }
            }
            //bias: https://stackoverflow.com/a/13342725/18255427
            this->network[i][j].bias += this->eta*this->d[i][j];
            //https://online.stat.psu.edu/stat501/lesson/1/1.2
        }
    }
    return MSE;
}

float MultiLayerPerceptron::train(const std::vector<Eigen::VectorXf>& x_train, const std::vector<Eigen::VectorXf>& y_train, unsigned long num_epochs, bool interactive)
{
    if (x_train.size() != y_train.size())
    {
        throw std::runtime_error("# of x_train rows != # of y_train rows");
    }
//    puts("Press ctrl-c to continue");
    float MSE;
    unsigned long int num_rows = x_train.size();
    for (unsigned long epoch = 0; ((num_epochs != 0) ? (epoch < num_epochs) : true); epoch++)
    {
        MSE = 0.0;
        
        for (unsigned long i = 0; i < num_rows; i++)
        {
            MSE += this->bp(x_train[i], y_train[i]);
            
        }
        MSE /= num_rows;

        if (interactive)
        {
            if (epoch % 100 == 0)
            {
                std::cout<<"MSE = "<<MSE<< '\r' << std::flush;
            }
            if (MultiLayerPerceptron::interrupted)
            {
                std::cout << "\nInterrupted by Ctrl-C. Exiting loop.\n";
                std::cout<<"MSE = "<<MSE<< '\n';
                MultiLayerPerceptron::interrupted = 0; //reset MultiLayerPerceptron::interrupted
                return MSE;
            }
        }
    }
    return MSE;
}

void MultiLayerPerceptron::signalHandler(int signum)
{
    if (signum == SIGINT)
    {
        interrupted = 1;
    }
}

std::vector<Eigen::VectorXf> MultiLayerPerceptron::predict(const std::vector<Eigen::VectorXf>& data)
{
    std::vector<Eigen::VectorXf> temp;
    for (const auto& vec: data)
    {
//        temp.push_back((-(this->run(vec).array().inverse() - 1.0f).log()).matrix());
        temp.push_back(this->run(vec));
    }
    return temp;
}

std::vector<Eigen::VectorXf> MultiLayerPerceptron::sigmoid(const std::vector<Eigen::VectorXf>& x)
{
    // Create a new vector to store the results
    std::vector<Eigen::VectorXf> result;
    result.reserve(x.size());
    
    for (auto& vec: x)
    {
        result.push_back(vec.unaryExpr([](float element)
        {
           // Apply the sigmoid function to each element
           return 1.0f / (1.0f + std::exp(-element));
        }));
    }
        
    return result;
}

float MultiLayerPerceptron::expression_evaluator(float w_k, float eta, float d_ij, float value, const Eigen::VectorXf& params)
{
    std::stack<float> stack;
    size_t const_count = 0;
    bool is_prefix = (expression_type == "prefix");
    for (int i = (is_prefix ? (pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < pieces.size())); (is_prefix ? (i--) : (i++)))
    {
        std::string token = MultiLayerPerceptron::__tokens_dict[pieces[i]];
        if (std::find(MultiLayerPerceptron::__operators_float.begin(), MultiLayerPerceptron::__operators_float.end(), pieces[i]) == MultiLayerPerceptron::__operators_float.end()) // leaf
        {
            if (token == "const")
            {
                stack.push(params(const_count++));
            }
            else if (token == "w_k")
            {
                stack.push(w_k);
            }
            else if (token == "eta")
            {
                stack.push(eta);
            }
            else if (token == "d_ij")
            {
                stack.push(d_ij);
            }
            else if (token == "value")
            {
                stack.push(value);
            }
        }
        else if (std::find(MultiLayerPerceptron::__unary_operators_float.begin(), MultiLayerPerceptron::__unary_operators_float.end(), pieces[i]) != MultiLayerPerceptron::__unary_operators_float.end()) // Unary operator
        {
            if (token == "cos")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(cos(temp));
            }
            else if (token == "exp")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(exp(temp));
            }
            else if (token == "sqrt")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(sqrt(temp));
            }
            else if (token == "sin")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(sin(temp));
            }
            else if (token == "asin" || token == "arcsin")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(asin(temp));
            }
            else if (token == "log" || token == "ln")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(log(temp));
            }
            else if (token == "tanh")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(tanh(temp));
            }
            else if (token == "acos" || token == "arccos")
            {
                float temp = stack.top();
                stack.pop();
                stack.push(acos(temp));
            }
            else if (token == "~") //unary minus
            {
                float temp = stack.top();
                stack.pop();
                stack.push(-temp);
            }
        }
        else // binary operator
        {
            float left_operand = stack.top();
            stack.pop();
            float right_operand = stack.top();
            stack.pop();
            if (token == "+")
            {
                stack.push(((expression_type == "postfix") ? (right_operand + left_operand) : (left_operand + right_operand)));
            }
            else if (token == "-")
            {
                stack.push(((expression_type == "postfix") ? (right_operand - left_operand) : (left_operand - right_operand)));
            }
            else if (token == "*")
            {
                stack.push(((expression_type == "postfix") ? (right_operand * left_operand) : (left_operand * right_operand)));
            }
            else if (token == "/")
            {
                stack.push(((expression_type == "postfix") ? (right_operand / left_operand) : (left_operand / right_operand)));
            }
            else if (token == "^")
            {
                stack.push(((expression_type == "postfix") ? (/*right_operand.pow(left_operand) */pow(right_operand, left_operand)) : (/*left_operand.pow(right_operand)*/pow(left_operand, right_operand))));
            }
        }
    }
    return stack.top();
}
