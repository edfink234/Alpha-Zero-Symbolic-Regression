#include "MLP.h"
#include <cassert>

//open /Users/edwardfinkelstein/LinkedIn/Ex_Files_Neural_Networks/Exercise\ Files/03_03/NeuralNetworks/*cpp /Users/edwardfinkelstein/LinkedIn/Ex_Files_Neural_Networks/Exercise\ Files/03_03/NeuralNetworks/*h
//cd /Users/edwardfinkelstein/LinkedIn/Ex_Files_Neural_Networks/Exercise\ Files/03_03/NeuralNetworks/

// Return a new Perceptron object with the specified number of inputs
Perceptron::Perceptron(int inputs, float bias)
{
    this->bias = bias;
    this->weights.resize(inputs);

    // Use Eigen's random number generation to initialize the weights
    this->weights = Eigen::VectorXf::Random(inputs);
}

// Run the perceptron. x is a vector with the input values.
float Perceptron::run(const Eigen::VectorXf& x)
{
	return sigmoid(x.dot(weights) + bias);
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

// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(std::vector<int> layers, float bias, float eta)
{
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;

    for (int i = 0; i < this->layers.size(); i++) //for each layer
    {
        this->values.emplace_back(Eigen::VectorXf::Zero(layers[i])); //add vector of values
        this->network.emplace_back(); //add vector of neurons
        this->d.emplace_back(Eigen::VectorXf::Zero(layers[i]));
        if (i > 0)  //network[0] is the input layer, so it has no neurons, i.e., it's empty
        {
            for (int j = 0; j < this->layers[i]; j++)
            {
                this->network[i].emplace_back(this->layers[i-1], this->bias);
            }
        }
    }
}


void MultiLayerPerceptron::set_weights(std::vector<Eigen::MatrixXf>&& w_init) 
{
    // Write all the weights into the neural network.
    // w_init is a vector of vectors of vectors of floats. 
    for (int i = 1; i < network.size(); i++){ //first layer is the input layer so they're no neurons there
        for (int j = 0; j < layers[i]; j++) { //for each neuron
            network[i][j].set_weights(w_init[i-1].row(j));
        }
    }
}

void MultiLayerPerceptron::reset_weights() 
{
    static std::random_device rd;  // Obtain a random number from hardware
    static std::mt19937 gen; // Seed the generator
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
//                = 2 * (y_{k} - o_{k}) * d (- sigmoid(x_k*w_k + b_k)) / d w_k
//                = -2 * (y_k - o_k) * o_k * (1 - o_k)
//                \propto o_k * (1 - o_k) * (y_k - o_k)

    for (int i = 0; i < this->layers.back(); i++)
    {
        float o_k = this->values.back()[i];
        this->d.back()[i] = o_k * (1 - o_k) * (y[i] - o_k);
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
                this->network[i][j].weights[k] += this->eta*this->d[i][j]*this->values[i-1][k];
            }
            //bias
            this->network[i][j].bias += this->eta*this->d[i][j]*bias;
        }
    }
    

    return MSE;
}

