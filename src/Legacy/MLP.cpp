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

Eigen::VectorXf PerceptronLayer::sigmoid(const Eigen::VectorXf& x)
{
    return 1.0f / (1.0f + (-x.array()).exp());
}

PerceptronLayer::PerceptronLayer(int inputs, int num_neurons, std::vector<float>&& bias_vec) : weights(num_neurons, inputs), biases(num_neurons)
{
    if (bias_vec.size() != biases.size())
    {
        biases = Eigen::VectorXf::Random(num_neurons);
    }
    else
    {
        for (int i = 0; i < bias_vec.size(); i++)
        {
            biases(i) = bias_vec[i];
        }
    }
    
    this->weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(num_neurons, inputs); //Random num_neurons x inputs matrix in the range (-1,1)
}

// Run the perceptron. x is a vector with the input values.
Eigen::VectorXf PerceptronLayer::run(const Eigen::VectorXf& x)
{
//    printf("weights.rows() = %lu, weights.cols() = %lu\n", weights.rows(), weights.cols());
//    printf("x.rows() = %lu, x.cols() = %lu\n", x.rows(), x.cols());
//    printf("biases.rows() = %lu, biases.cols() = %lu\n", biases.rows(), biases.cols());

    return sigmoid(weights * x + biases);
}

// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(std::vector<int> layers, float bias, float eta)
{
    // Set up the signal handler
    signal(SIGINT, signalHandler);
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;

    for (int i = 0; i < this->layers.size(); i++) //for each layer
    {
        this->values.emplace_back(Eigen::VectorXf::Zero(layers[i])); //add vector of values
        this->d.emplace_back(Eigen::VectorXf::Zero(layers[i]));
        if (i > 0)  //network[0] is the input layer, so it has no neurons, i.e., it's empty
        {
//            for (int j = 0; j < this->layers[i]; j++) //add `this->layers[i]` Perceptrons to `this->network[i]`
//            {
//                this->network[i].emplace_back(this->layers[i-1], this->bias);
//            }
            this->network.emplace_back(this->layers[i-1], this->layers[i], std::vector<float>(bias, this->layers[i]));
        }
        else
        {
            this->network.emplace_back(0, 0); //dummy, input layer
        }
    }
}

void MultiLayerPerceptron::set_weights(std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&& w_init)
{
    // Write all the weights into the neural network.
    // w_init is a vector of vectors of vectors of floats. 
    size_t num_cols;
    for (int i = 1; i < network.size(); i++) //For each layer
    { //first layer is the input layer so they're no neurons there
//        for (int j = 0; j < layers[i]; j++) 
//        { //for each neuron
//            network[i][j].set_weights(w_init[i-1].row(j));
//        }
        num_cols = w_init[i-1].cols();
        network[i].weights = w_init[i-1].leftCols(num_cols - 1);
        network[i].biases = w_init[i-1].rightCols(1);
    }
}

void MultiLayerPerceptron::reset_weights() 
{
//    static std::random_device rd;  // Obtain a random number from hardware
//    static std::mt19937 gen; // Seed the generator
//    static std::uniform_real_distribution<> distr; // Define the range
    // Write all the weights into the neural network.
    // w_init is a vector of vectors of vectors of floats.
    for (int i = 1; i < network.size(); i++)
    { //first layer is the input layer so they're no neurons there
//        for (int j = 0; j < layers[i]; j++)
//        { //for each neuron
//            network[i][j].weights = Eigen::VectorXf::Random(network[i][j].weights.size());
//            network[i][j].bias = distr(gen);
//        }
        network[i].weights = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Random(layers[i], layers[i-1]);
        network[i].biases = Eigen::VectorXf::Random(layers[i]);
    }
}

void MultiLayerPerceptron::print_weights() 
{
    std::cout << '\n';
    for (int i = 1; i < network.size(); i++)
    { //first layer is the input layer so they're no neurons there
//        for (int j = 0; j < layers[i]; j++) 
//        {
//            std::cout << "Layer " << i+1 << " Neuron " << j+1 << ": ";
//            for (auto &it: network[i][j].weights)
//            {
//                std::cout << it <<"   ";
//            }
//            std::cout << network[i][j].bias << '\n';
//        }
        std::cout << "Layer " << i << " weights:\n" << network[i].weights << '\n';
        std::cout << "Layer " << i << " biases:\n" << network[i].biases << '\n';
    }
    std::cout << '\n';
}
//
Eigen::VectorXf MultiLayerPerceptron::run(const Eigen::VectorXf& x) 
{
    // Run an input forward through the neural network.
    // x is a vector with the input values.
    //Returns: vector with output values, i.e., the last element in the values vector
    this->values[0] = x; 
    for (int i = 1; i < network.size(); i++) //for each layer
    {
//        for (int j = 0; j < this->layers[i]; j++) //for each neuron
//        {
//            this->values[i][j] = this->network[i][j].run(this->values[i-1]);
//            //run the previous layer output values through this neuron `this->network[i][j]`
//        }
//        printf("this->values[i-1].rows() = %lu, this->values[i-1].cols() = %lu\n", this->values[i-1].rows(), this->values[i-1].cols());

        this->values[i] = this->network[i].run(this->values[i-1]);
    }
    return this->values.back();
}

///*/
// MSE = 1/n * \sum_{i = 0}^{n-1} (y_i - o_i)^2
// */
float MultiLayerPerceptron::mse(const Eigen::VectorXf& x, const Eigen::VectorXf& y)
{
    if (x.size() != y.size())
    {
        throw std::invalid_argument("Vectors must be of the same size");
    }
    return (x - y).squaredNorm() / x.size();
}
//
//// Run a single (x,y) pair with the backpropagation algorithm.
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

//    for (int i = 0; i < this->layers.back(); i++)
//    {
//        float o_k = this->values.back()[i];
//        this->d.back()[i] = o_k * (1 - o_k) * (y[i] - o_k);
//    }
    
    auto& val = this->values.back(); // For readability and performance
    this->d.back() = val.array() * (1 - val.array()) * (y - val).array();
    
//     STEP 4: Calculate the error term of each unit on each layer
    for (int i = network.size()-2; i > 0; i--) //for each layer (starting from the one before the output layer and ending at and including the layer right before the input layer)
    {
//        for (int h = 0; h < layers[i]; h++) //for each neuron in layer i
//        {
//            assert(layers[i] == network[i].size());
//            float fwd_error = 0.0f;
//            //fwd_error = \sum_{k \in \mathrm{outs}} w_{kh} \delta_k
//            for (int k = 0; k < layers[i+1]; k++) //for each neuron in layer i+1
//            {
//                fwd_error += network[i+1][k].weights[h]*d[i+1][k];
//            }
//            
//            //\delta_h = o_h*(1-o_h)*fwd_error
//            this->d[i][h] = this->values[i][h] * (1 - this->values[i][h]) * fwd_error;
//        }
        
//        printf("layers[i+1] = %d\n",layers[i+1]);
//        printf("network[i+1].weights.rows() = %lu, network[i+1].weights.cols() = %lu\n", network[i+1].weights.rows(), network[i+1].weights.cols());
//        printf("d[i+1].rows() = %lu, d[i+1].cols() = %lu\n", d[i+1].rows(), d[i+1].cols());
//        printf("this->values[i].rows() = %lu, this->values[i].cols() = %lu\n", this->values[i].rows(), this->values[i].cols());
//        printf("(Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(this->values[i].rows()) - this->values[i]).rows() = %lu, (Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(this->values[i].rows()) - this->values[i]).cols() = %lu\n", (Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(this->values[i].rows(), this->values[i].cols()) - this->values[i]).rows(), (Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(this->values[i].rows(), this->values[i].cols()) - this->values[i]).cols());
    this->d[i] = this->values[i] * (Eigen::Matrix<float, 1, Eigen::Dynamic>::Ones(this->values[i].rows(), this->values[i].cols()) - this->values[i]).transpose() * (d[i+1].transpose() * network[i+1].weights);
    }
    
//    puts("here?");
    
    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (int i = 1; i < network.size(); i++) //for each layer
    {
//        for (int j = 0; j < layers[i]; j++)
//        {
//            for (int k = 0; k < layers[i-1]; k++) //weights
//            {
//                this->network[i][j].weights[k] += this->eta*this->d[i][j]*this->values[i-1][k];
//            }
//            //bias
//            this->network[i][j].bias += this->eta*this->d[i][j]*bias;
//        }
        
//        printf("this->values[i-1].rows() = %lu, this->values[i-1].cols() = %lu\n", this->values[i-1].rows(), this->values[i-1].cols());
//        printf("this->d[i].rows() = %lu, this->d[i].cols() = %lu\n", this->d[i].rows(), this->d[i].cols());

        this->network[i].weights += this->eta*this->d[i]*this->values[i-1];
//        this->network[i].biases += this->eta*this->d[i]*bias;
        
    }
//
    return MSE;
}


float MultiLayerPerceptron::train(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x_train, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& y_train, unsigned long num_epochs)
{
    if (x_train.rows() != y_train.rows())
    {
        throw std::runtime_error("# of x_train rows != # of y_train rows");
    }
    puts("Press ctrl-c to continue");
    float MSE;
    unsigned long int num_rows = x_train.rows();
    for (unsigned long epoch = 0; epoch < num_epochs; epoch++)
    {
        MSE = 0.0;
        for (unsigned long i = 0; i < num_rows; i++)
        {
            MSE += this->bp(x_train.row(i), y_train.row(i));
            
        }
        MSE /= num_rows;

        if (epoch % 100 == 0)
        {
            std::cout<<"MSE = "<<MSE<< '\r' << std::flush;
        }
        if (MultiLayerPerceptron::interrupted)
        {
            std::cout << "\nInterrupted by Ctrl-C. Exiting loop.\n";
            return MSE;
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
