#include "MLP_Vec.h"
#include <cassert>
#include <stack>
#define FLUSHTHETOILET std::flush
//#define INTERACTIVE_TRAIN true

//Source: https://github.com/LinkedInLearning/training-neural-networks-in-cpp-4404365

// Return a new Perceptron object with the specified number of inputs
Perceptron::Perceptron(int inputs, float bias, const std::string& output_type)
{
    this->bias = bias;
    this->weights.resize(inputs);

    // Use Eigen's random number generation to initialize the weights
    this->weights = Eigen::VectorXf::Random(inputs);
    this->velocities = Eigen::VectorXf::Random(inputs);
    this->v = Eigen::VectorXf::Zero(inputs); //2nd moment vector for Adam
    this->m = Eigen::VectorXf::Zero(inputs); //1st moment vector for Adam
    this->gradients = Eigen::VectorXf::Zero(inputs);
    this->expt_grad_squared = Eigen::VectorXf::Zero(inputs);
    this->expt_weight_squared = Eigen::VectorXf::Zero(inputs);
    this->output_type = output_type;
}

// Run the perceptron. x is a vector with the input values.
float Perceptron::run(const Eigen::VectorXf& x)
{
    assert(x.size() == this->weights.size());
    float dot_product = x.dot(this->weights) + bias;
    return (output_type != "sigmoid") ? dot_product : sigmoid(dot_product);
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
    return 1.0f / (1.0f + exp(-x));
}

float Perceptron::scale_between(float unscaled_num, float min, float max, float max_allowed, float min_allowed)
{
    return (max_allowed - min_allowed) *
    (unscaled_num - min) / (max - min) + min_allowed;
}

// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(const std::vector<int>& layers, const std::deque<std::string>& layer_types, float bias, float eta, float theta, float gamma, const std::string& weight_update, const std::string& expression_type, float epsilon, float beta_1, float beta_2, float lambda)
{
    // Set up the signal handler
    signal(SIGINT, signalHandler);
    
    std::call_once(initialization_flag, [&]()
    {
        this->__unary_operators = {"cos", "exp", "sqrt", "sin", "asin", "ln", "tanh", "acos", "~"};
        this->__binary_operators = {"+", "-", "*", "/", "^"};
        this->__operators.clear();
        for (std::string& i: this->__unary_operators)
        {
            this->__operators.push_back(i);
        }
        for (std::string& i: this->__binary_operators)
        {
            this->__operators.push_back(i);
        }
//        puts("this->__unary_operators");
//        for (const std::string& i: this->__unary_operators) {std::cout << i << ' ';}puts("");
//        puts("this->__binary_operators");
//        for (const std::string& i: this->__binary_operators) {std::cout << i << ' ';}puts("");
    });

    this->layers = layers;
    this->layer_types = layer_types;
    if (!this->layer_types.size())
    {
        this->layer_types.resize(this->layers.size() - 1, "none");
    }
    this->layer_types.push_front("none");
    
    assert(this->layers.size() == this->layer_types.size());
    this->bias = bias;
    this->eta = eta;
    this->theta = theta;
    this->weight_update = weight_update;
    this->expression_type = expression_type;
    this->epsilon = epsilon;
    this->gamma = gamma;
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->lambda = lambda;
    this->t = 1; //used in Adam

    size_t mlp_sz = this->layers.size();

    for (int i = 0; i < mlp_sz; i++) //for each layer
    {
        this->values.emplace_back(Eigen::VectorXf::Zero(layers[i])); //add vector of values
        this->network.emplace_back(); //add vector of neurons, first layer in the empty input layer
        this->d.emplace_back(Eigen::VectorXf::Zero(layers[i]));
        if (this->weight_update == "SR")
        {
            this->d_nest.emplace_back(Eigen::VectorXf::Zero(layers[i]));
        }
        if (i > 0)  //network[0] is the input layer, so it has no neurons, i.e., it's empty
        {
            for (int j = 0; j < this->layers[i]; j++)
            {
                this->network[i].emplace_back(this->layers[i-1], this->bias, this->layer_types[i]);
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

void MultiLayerPerceptron::reset_params()
{
    this->values[0] = Eigen::VectorXf::Zero(layers[0]);
    this->d[0] = this->values[0];
    if (this->weight_update == "SR")
    {
        this->d_nest[0] = this->d[0];
    }
    for (int i = 1; i < network.size(); i++)
    { //first layer is the input layer so they're no neurons there
        for (int j = 0; j < layers[i]; j++)
        { //for each neuron
            network[i][j].weights = Eigen::VectorXf::Random(network[i][j].weights.size());
            network[i][j].bias = this->bias;
            network[i][j].velocities = Eigen::VectorXf::Random(network[i][j].velocities.size());
            network[i][j].v = Eigen::VectorXf::Zero(network[i][j].v.size()); //2nd moment vector for Adam
            network[i][j].m = Eigen::VectorXf::Zero(network[i][j].m.size()); //1st moment vector for Adam
            network[i][j].gradients = Eigen::VectorXf::Zero(network[i][j].gradients.size());
            network[i][j].expt_grad_squared = Eigen::VectorXf::Zero(network[i][j].expt_grad_squared.size());
            network[i][j].expt_weight_squared = Eigen::VectorXf::Zero(network[i][j].expt_weight_squared.size());
            this->values[i][j] = 0.0f;
            this->d[i][j] = 0.0f;
            this->d_nest[i][j] = 0.0f;
        }
    }
    this->t = 0;
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
    this->values[0] = x; //First layer of values are the inputs!
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
    
//       Sigmoid: delta_k = d MSE(y_k, o_k) / d w_k
//                        = d MSE(y_k, sigmoid(x_k*w_k + b_k)) / d w_k
//                        = d ((1/n) * \sum_{i = 0}^{n-1} (y_{k_{i}} - sigmoid(x_k*w_k + b_k)_{k_{i}})^2) / d w_k
//                        = (1/n) * d (\sum_{i = 0}^{n-1} (y_{k_{i}} - sigmoid(x_k*w_k + b_k)_{k_{i}})^2) / d w_k
//                        = d (y_{k} - sigmoid(x_k*w_k + b_k))^2) / d w_k
//                        = 2 * (y_{k} - o_{k}) * d (- o_{k}) / d w_k
//                        = -2 * (y_k - o_k) * o_k * (1 - o_k)
//                        \propto o_k * (1 - o_k) * (y_k - o_k) * x_k
    
//        Linear: delta_k = d MSE(y_k, o_k) / d w_k
//                        = d (y_{k} - (x_k*w_k + b_k))^2) / d w_k
//                        = 2 * (y_{k} - (x_k*w_k + b_k)) * d (- (x_k*w_k + b_k)) / d w_k
//                        = -2 * (y_k - (x_k*w_k + b_k)) * x_k

    //Calculate the error terms for the output layer
    for (int i = 0; i < this->layers.back(); i++) //each neuron in the last layer 
    {
        float o_k = this->values.back()[i];
        this->d.back()[i] = ((this->network.back()[i].output_type == "sigmoid") ? (o_k * (1 - o_k) * (y[i] - o_k)) : (y[i] - o_k));//* x.sum();
        // /Users/edwardfinkelstein/Desktop/Machine\ Learning/\(The\ Morgan\ Kaufmann\ Series\ in\ Data\ Management\ Systems\)\ Ian\ H.\ Witten\,\ Eibe\ Frank\,\ Mark\ A.\ Hall\ -\ Data\ Mining_\ Practical\ Machine\ Learning\ Tools\ and\ Techniques\,\ Third\ Edition-Morgan\ Kaufmann\ \(2011\).pdf, page 273
    }
    
    if (this->weight_update == "SR")
    {
        this->d_nest.back() = this->d.back();
    }
    // STEP 4: Calculate the error term of each unit on each layer -> backpropagation
    for (int i = network.size()-2; i > 0; i--) //for each layer (starting from the one before the output layer and ending at and including the layer right before the input layer)
    {
        for (int h = 0; h < layers[i]; h++) //for each neuron in layer i
        {
            assert(layers[i] == network[i].size());
            float fwd_error = 0.0f, fwd_error_nest = 0.0f;
            //fwd_error = \sum_{k \in \mathrm{outs}} w_{kh} \delta_k
            for (int k = 0; k < layers[i+1]; k++) //for each neuron in layer i+1
            {
                if (this->weight_update == "NAG")
                {
                    fwd_error += (network[i+1][k].weights[h] - this->theta*this->network[i+1][k].velocities[h])*d[i+1][k]; //step 2: measure the gradient
                }
                else if (this->weight_update == "SR")
                {
                    fwd_error_nest += (network[i+1][k].weights[h] - this->theta*this->network[i+1][k].velocities[h])*d[i+1][k];
                    fwd_error += network[i+1][k].weights[h]*d[i+1][k];
                }
                else
                {
                    fwd_error += network[i+1][k].weights[h]*d[i+1][k];
                }
            }
            //\delta_h = o_h*(1-o_h)*fwd_error
            if (this->weight_update == "SR")
            {
                float deriv = 1.0f;
                if (this->network[i][h].output_type == "sigmoid")
                {
                    deriv = this->values[i][h] * (1 - this->values[i][h]);
                }
                
                this->d[i][h] = deriv * fwd_error;
                this->d_nest[i][h] = deriv * fwd_error_nest;
            }
            else
            {
                float deriv = 1.0f;
                if (this->network[i][h].output_type == "sigmoid")
                {
                    deriv = this->values[i][h] * (1 - this->values[i][h]);
                }
                this->d[i][h] = deriv * fwd_error;
            }
        }
    }
    
    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (int i = 1; i < network.size(); i++) //for each layer
    {
        for (int j = 0; j < layers[i]; j++) //for each neuron
        {
            for (int k = 0; k < layers[i-1]; k++) //weights (number of neurons in previous layer i-1
            {
                if (this->weight_update == "basic")
                {
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] + this->eta*this->d[i][j]*this->values[i-1][k];
                }
                else if (this->weight_update == "heavy ball" || this->weight_update == "NAG")
                {
                    this->network[i][j].velocities[k] = this->theta*this->network[i][j].velocities[k] + this->eta*this->d[i][j]*this->values[i-1][k]; //step 1 & then 3 in NAG: compute momentum
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] + this->network[i][j].velocities[k];
                    
                    
                    //Effectively:
                    //this->network[i][j].weights[k] = this->network[i][j].weights[k] + this->theta*this->network[i][j].velocities[k] + this->eta*this->d[i][j]*this->values[i-1][k];
                }
                else if (this->weight_update == "SR") //symbolic regression
                {
                    this->network[i][j].velocities[k] = this->theta*this->network[i][j].velocities[k] + this->eta*this->d[i][j]*this->values[i-1][k]; //step 1 & then 3 in NAG: compute momentum
                    float g_t_k = this->d[i][j] * this->values[i-1][k];
                    this->network[i][j].gradients[k] = this->network[i][j].gradients[k] + g_t_k * g_t_k;
                    this->network[i][j].expt_grad_squared[k] = this->gamma*this->network[i][j].expt_grad_squared[k] + (1-this->gamma)*g_t_k*g_t_k;
                    float delta_w_t_k = (-this->eta / sqrt(this->network[i][j].expt_grad_squared[k] + this->epsilon)) * g_t_k;
                    this->network[i][j].expt_weight_squared[k] = this->gamma*this->network[i][j].expt_weight_squared[k] + (1-this->gamma)*delta_w_t_k*delta_w_t_k;
                    float delta_w_t_k_ada_delta = -sqrt((this->network[i][j].expt_weight_squared[k] + this->epsilon) / (this->network[i][j].expt_grad_squared[k] + this->epsilon) ) * g_t_k;
                    this->network[i][j].m[k] = this->beta_1*this->network[i][j].m[k] + (1-this->beta_1)*g_t_k;
                    this->network[i][j].v[k] = this->beta_2*this->network[i][j].v[k] + (1-this->beta_2)*g_t_k*g_t_k;
                    float m_t_k_hat = this->network[i][j].m[k]/(1-pow(this->beta_1, t));
                    float v_t_k_hat = this->network[i][j].v[k]/(1-pow(this->beta_2, t));
                    
                    this->network[i][j].weights[k] = this->expression_evaluator(this->network[i][j].weights[k], this->d[i][j], this->values[i-1][k], this->d_nest[i][j], this->network[i][j].velocities[k], this->network[i][j].gradients[k], g_t_k, this->network[i][j].expt_grad_squared[k], delta_w_t_k, this->network[i][j].expt_weight_squared[k], delta_w_t_k_ada_delta, this->network[i][j].m[k], this->network[i][j].v[k], m_t_k_hat, v_t_k_hat);
                    
//                    float expression_evaluator(float w_k = 0.0f, float d_ij = 0.0f, float value = 0.0f, float d_ij_nest = 0.0f, float velocity_k = 0.0f, float gradient_k = 0.0f, float g_t_k = 0.0f, float expt_grad_squared_k = 0.0f, float delta_w_t_k = 0.0f, float expt_weight_squared_k = 0.0f, float delta_w_t_k_ada_delta = 0.0f, float m_t_k = 0.0f, float v_t_k = 0.0f, float m_t_k_hat = 0.0f, float v_t_k_hat = 0.0f, const Eigen::VectorXf& params = {});
                }
                else if (this->weight_update == "AdaGrad")
                {
                    float g_t_k = this->d[i][j] * this->values[i-1][k];
                    this->network[i][j].gradients[k] = this->network[i][j].gradients[k] + g_t_k * g_t_k;
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] + (this->eta / sqrt(this->network[i][j].gradients[k] + this->epsilon)) * g_t_k;
                }
                else if (this->weight_update == "RMSProp")
                {
                    float g_t_k = this->d[i][j] * this->values[i-1][k];
                    this->network[i][j].expt_grad_squared[k] = this->gamma*this->network[i][j].expt_grad_squared[k] + (1-this->gamma)*g_t_k*g_t_k;
                    float delta_w_t_k = (this->eta / sqrt(this->network[i][j].expt_grad_squared[k] + this->epsilon)) * g_t_k;
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] + delta_w_t_k;
                }
                else if (this->weight_update == "AdaDelta") //As described in "Recent Advances in Stochastic Gradient Descent in Deep Learning"
                {
                    float g_t_k = this->d[i][j] * this->values[i-1][k];
                    this->network[i][j].expt_grad_squared[k] = this->gamma*this->network[i][j].expt_grad_squared[k] + (1-this->gamma)*g_t_k*g_t_k;
                    float delta_w_t_k = -sqrt((this->network[i][j].expt_weight_squared[k] + this->epsilon) / (this->network[i][j].expt_grad_squared[k] + this->epsilon) ) * g_t_k;
                    this->network[i][j].expt_weight_squared[k] = this->gamma*this->network[i][j].expt_weight_squared[k] + (1-this->gamma)*delta_w_t_k*delta_w_t_k;
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] - delta_w_t_k;
                }
                else if (this->weight_update == "Adam")
                {
                    float g_t_k = this->d[i][j] * this->values[i-1][k];
                    this->network[i][j].m[k] = this->beta_1*this->network[i][j].m[k] + (1-this->beta_1)*g_t_k;
                    this->network[i][j].v[k] = this->beta_2*this->network[i][j].v[k] + (1-this->beta_2)*g_t_k*g_t_k;
//                    std::cout << "this->t = " << this->t << '\n';
                    float m_t_k_hat = this->network[i][j].m[k]/(1-pow(this->beta_1, t));
                    float v_t_k_hat = this->network[i][j].v[k]/(1-pow(this->beta_2, t));
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] + (this->eta * m_t_k_hat) / (sqrt(v_t_k_hat) + this->epsilon);
                }
                else if (this->weight_update == "AdamW")
                {
                    float g_t_k = this->d[i][j] * this->values[i-1][k];
                    this->network[i][j].m[k] = this->beta_1*this->network[i][j].m[k] + (1-this->beta_1)*g_t_k;
                    this->network[i][j].v[k] = this->beta_2*this->network[i][j].v[k] + (1-this->beta_2)*g_t_k*g_t_k;
                    float m_t_k_hat = this->network[i][j].m[k]/(1-pow(this->beta_1, t));
                    float v_t_k_hat = this->network[i][j].v[k]/(1-pow(this->beta_2, t));
                    this->network[i][j].weights[k] = this->network[i][j].weights[k] + this->eta*((this->network[i][j].weights[k]*this->lambda) +
                    (m_t_k_hat / (sqrt(v_t_k_hat) + this->epsilon))); //see https://arxiv.org/html/2405.13698v1 for some discussion on how to set the weight decay lambda
                }

            }
            //bias: https://stackoverflow.com/a/13342725/18255427
            this->network[i][j].bias += this->eta*this->d[i][j];
            //https://online.stat.psu.edu/stat501/lesson/1/1.2
        }
    }
    return MSE;
}

float MultiLayerPerceptron::train(const std::vector<Eigen::VectorXf>& x_train, const std::vector<Eigen::VectorXf>& y_train, unsigned long num_epochs)
{
    if (x_train.size() != y_train.size())
    {
        throw std::runtime_error("# of x_train rows != # of y_train rows");
    }
    
//    puts("Press ctrl-c to continue");
    float MSE = 0.0f;
    unsigned long int num_rows = x_train.size();
    assert(num_rows);
    for (unsigned long epoch = 0; ((num_epochs != 0) ? (epoch < (num_epochs - 1)) : true); epoch++)
    {
        MSE = 0.0f;
        for (unsigned long i = 0; i < num_rows; i++)
        {
            MSE += this->bp(x_train[i], y_train[i]);
        }
        if (std::isinf(MSE) || std::isnan(MSE))
        {
            return MSE;
        }
        #ifdef INTERACTIVE_TRAIN
            std::cout << "Epoch " << epoch << " MSE = " << MSE/num_rows << '\r' << FLUSHTHETOILET;
            if (MultiLayerPerceptron::interrupted)
            {
                std::cout << "\nInterrupted by Ctrl-C. Exiting loop.\n";
                MultiLayerPerceptron::interrupted = 0; //reset MultiLayerPerceptron::interrupted
                break;
            }
        #endif // INTERACTIVE_TRAIN
        this->t++; //increase t by 1 for Adam
    }
    //On the last epoch, we update the MSE
    MSE = 0.0;
    for (unsigned long i = 0; i < num_rows; i++)
    {
        MSE += this->bp(x_train[i], y_train[i]);
        if (std::isinf(MSE) || std::isnan(MSE))
        {
            return MSE;
        }
    }
    MSE /= num_rows;
    
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

float MultiLayerPerceptron::expression_evaluator(float w_k, float d_ij, float value, float d_ij_nest, float velocity_k, float gradient_k, float g_t_k, float expt_grad_squared_k, float delta_w_t_k, float expt_weight_squared_k, float delta_w_t_k_ada_delta, float m_t_k, float v_t_k, float m_t_k_hat, float v_t_k_hat, const Eigen::VectorXf& params)
{
    std::stack<float> stack;
    bool is_prefix = (expression_type == "prefix");
    
    for (int i = (is_prefix ? (pieces.size() - 1) : 0); (is_prefix ? (i >= 0) : (i < pieces.size())); (is_prefix ? (i--) : (i++)))
    {
        std::string token = pieces[i];
        if (std::find(this->__operators.begin(), this->__operators.end(), pieces[i]) == this->__operators.end()) // leaf
        {
//            {"w_k", "eta", "theta", "gamma", "epsilon", "beta_1", "beta_2", "d_ij", "value", "d_ij_nest", "velocity_k", "gradient_k", "g_t_k", "expt_grad_squared_k", "delta_w_t_k", "expt_weight_squared_k", "delta_w_t_k_ada_delta", "m_t_k", "v_t_k", "m_t_k_hat", "v_t_k_hat", "prev_w_k", "t"}
            if (token == "w_k")
            {
                stack.push(w_k);
            }
            else if (token == "eta")
            {
                stack.push(this->eta);
            }
            else if (token == "theta")
            {
                stack.push(this->theta);
            }
            else if (token == "gamma")
            {
                stack.push(this->gamma);
            }
            else if (token == "epsilon")
            {
                stack.push(this->epsilon);
            }
            else if (token == "beta_1")
            {
                stack.push(this->beta_1);
            }
            else if (token == "beta_2")
            {
                stack.push(this->beta_2);
            }
            else if (token == "d_ij")
            {
                stack.push(d_ij);
            }
            else if (token == "value")
            {
                stack.push(value);
            }
            else if (token == "d_ij_nest")
            {
                stack.push(d_ij_nest);
            }
            else if (token == "velocity_k")
            {
                stack.push(velocity_k);
            }
            else if (token == "gradient_k")
            {
                stack.push(gradient_k);
            }
            else if (token == "g_t_k")
            {
                stack.push(g_t_k);
            }
            else if (token == "expt_grad_squared_k")
            {
                stack.push(expt_grad_squared_k);
            }
            else if (token == "delta_w_t_k")
            {
                stack.push(delta_w_t_k);
            }
            else if (token == "expt_weight_squared_k")
            {
                stack.push(expt_weight_squared_k);
            }
            else if (token == "delta_w_t_k_ada_delta")
            {
                stack.push(delta_w_t_k_ada_delta);
            }
            else if (token == "m_t_k")
            {
                stack.push(m_t_k);
            }
            else if (token == "v_t_k")
            {
                stack.push(v_t_k);
            }
            else if (token == "m_t_k_hat")
            {
                stack.push(m_t_k_hat);
            }
            else if (token == "v_t_k_hat")
            {
                stack.push(v_t_k_hat);
            }
            else if (token == "t")
            {
                stack.push(this->t);
            }
            else
            {
                throw std::runtime_error(std::string("Error in MultiLayerPerceptron::expression_evaluator, trying to push token ")+token+", token.size() = "+std::to_string(token.size()));
            }
        }
        else if (std::find(this->__unary_operators.begin(), this->__unary_operators.end(), pieces[i]) != this->__unary_operators.end()) // Unary operator
        {
            assert((stack.size() >= 1 && "Wait what, stack.size() >= 1 failed 😨"));
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
            else
            {
                throw std::runtime_error(std::string("Error in MultiLayerPerceptron::expression_evaluator, trying to push token ")+token+", token.size() = "+std::to_string(token.size()));
            }
        }
        else // binary operator
        {
            float left_operand = stack.top();
            stack.pop();
            if (stack.size() == 0)
            {
                std::runtime_error(std::string("Error in MultiLayerPerceptron::expression_evaluator, trying to access element that doesn't exist, token = ")+token+", token.size() = "+std::to_string(token.size()));
            }
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
            else
            {
                throw std::runtime_error(std::string("Error in MultiLayerPerceptron::expression_evaluator, trying to push token ")+token+", token.size() = "+std::to_string(token.size()));
            }
        }
//        std::cout << "i = " << i << ", pieces.size() = " << pieces.size() << '\n';
//        std::cout << "pieces[" << i << "] = " << token << '\n';
//        std::cout << "stack.top() =" << stack.top();
    }
    assert(stack.size() > 0);
//    printf("stack.top() = "); std::cout << stack.top() << '\n';
    return stack.top();
}
