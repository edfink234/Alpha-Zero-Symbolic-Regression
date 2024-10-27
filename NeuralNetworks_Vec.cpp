// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//
//g++ -Wall -std=c++20 -o NeuralNetworks_Vec NeuralNetworks_Vec.cpp MLP_Vec.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/username/LBFGSpp -ffast-math -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native

#include <iostream>
#include <memory>
#include <float.h>
#include "MLP_Vec.h"
#include <csignal>
#include <algorithm>
#include <deque>

using Clock = std::chrono::high_resolution_clock;

float example_func(const Eigen::VectorXf& x)
{
    return (30.0f*x[0]*x[0])/((10.0f-x[0])*x[1]*x[1]) + x[0]*x[0]*x[0]*x[0] - (4.0f*x[0]*x[0]*x[0])/5.0f + x[1]*x[1]/2.0f - 2.0f*x[1] + (8.0f / (2.0f + x[0]*x[0] + x[1]*x[1])) + (x[1]*x[1]*x[1])/2.0f - x[0];
}

float example_func_1(const Eigen::VectorXf& x)
{
    return std::pow((x[0]*x[1]*x[2]*x[3]*x[4])/(4*x[5]*std::pow(sin(x[6]/2),2)),2);
}

float example_func_2(const Eigen::VectorXf& x)
{
    return x[0]*x[0]*x[0]*(x[0]-1.0f) + x[1]*(x[1]/2.0f - 1.0f);
}

Eigen::MatrixXf createLinspaceMatrix(int rows, int cols, std::vector<float> min_vec, std::vector<float> max_vec)
{
    assert((cols == static_cast<int>(min_vec.size())) && (cols == static_cast<int>(max_vec.size())));
    Eigen::MatrixXf mat(rows, cols);
    for (int col = 0; col < cols; ++col)
    {
        for (int row = 0; row < rows; ++row)
        {
            mat(row, col) = min_vec[col] + (max_vec[col] - min_vec[col]) * row / (rows - 1);
        }
    }
    return mat;
}

//Returns the number of seconds since `start_time`
template <typename T>
double timeElapsedSince(T start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
}

std::ostream& operator<<(std::ostream& out, const std::vector<Eigen::VectorXf>& data)
{
    for (const auto& vec: data)
    {
        for (size_t i = 0; i < vec.size(); i++)
        {
            out << vec[i] << ' ';
        }
        out << '\n';
    }
    return out;
}

int main()
{

    std::cout << "\n\n--------Logic Gate Example----------------\n\n";

    std::unique_ptr<Perceptron> p = std::make_unique<Perceptron>(2 /*inputs*/, 1.0f /*bias*/, "sigmoid" /*is_output*/); //2 inputs
    
    /*
     AND
     ===
     
     10*A + 10*B - 15

     A | B | output | sigmoid(output)
     0 | 0 | -15    | 1/(1+exp(15)) = 3e-7
     1 | 0 | -5     | 1/(1+exp(5)) = 6.7e-3
     0 | 1 | -5     | 1/(1+exp(5)) = 6.7e-3
     1 | 1 | 5      | 1/(1+exp(-5)) = 0.993
     
     */

    Eigen::VectorXf x(3);
    x << 10, 10, -15;
    p->set_weights(x); //AND
    
    std::cout << "AND Gate: "<<'\n';
    x.resize(2);
    x << 0, 0;
    std::cout<<p->run(x)<<'\n';
    x << 0, 1;
    std::cout<<p->run(x)<<'\n';
    x << 1, 0;
    std::cout<<p->run(x)<<'\n';
    x << 1, 1;
    std::cout<<p->run(x)<<'\n';

    /*
     OR
     ===

     2*A + 2*B - 1

     A | B | output | sigmoid(output)
     0 | 0 | -1     | 1/(1+exp(1)) = 0.26
     1 | 0 | 1      | 1/(1+exp(-1)) = 0.73
     0 | 1 | 1      | 1/(1+exp(-1)) = 0.73
     1 | 1 | 3      | 1/(1+exp(-3)) = 0.9526
    */
    
    x.resize(3);
    x << 2, 2, -1;
    p->set_weights(x); //OR
    std::cout << "OR Gate: "<<'\n';
    x.resize(2);
    x << 0, 0;
    std::cout<<p->run(x)<<'\n';
    x << 0, 1;
    std::cout<<p->run(x)<<'\n';
    x << 1, 0;
    std::cout<<p->run(x)<<'\n';
    x << 1, 1;
    std::cout<<p->run(x)<<'\n';

    /*
     NAND
     ===

     -10*A + -10*B + 15

     A | B | output   | sigmoid(output)
     0 | 0 | 15       | 1/(1+exp(-15)) = 1
     1 | 0 | 5        | 1/(1+exp(-1)) = 0.993307
     0 | 1 | 5        | 1/(1+exp(-1)) = 0.993307
     1 | 1 | -20      | 1/(1+exp(-3)) = 0.00669285
    */
    
    x.resize(3);
    x << -10, -10, 15;
    p->set_weights(x); //NAND
    std::cout << "NAND Gate: "<<'\n';
    x.resize(2);
    x << 0, 0;
    std::cout<<p->run(x)<<'\n';
    x << 0, 1;
    std::cout<<p->run(x)<<'\n';
    x << 1, 0;
    std::cout<<p->run(x)<<'\n';
    x << 1, 1;
    std::cout<<p->run(x)<<'\n';

    /*
     NOR
     ===

     -2*A + -2*B + 1

     A | B | output | sigmoid(output)
     0 | 0 | 1      | 1/(1+exp(-1)) = 0.73
     1 | 0 | -1     | 1/(1+exp(1)) = 0.26
     0 | 1 | -1     | 1/(1+exp(-1)) = 0.26
     1 | 1 | -3     | 1/(1+exp(3)) = 0.9526
    */
    
    x.resize(3);
    x << -2, -2, 1;
    p->set_weights(x); //NOR
    std::cout << "NOR Gate: "<<'\n';
    x.resize(2);
    x << 0, 0;
    std::cout<<p->run(x)<<'\n';
    x << 0, 1;
    std::cout<<p->run(x)<<'\n';
    x << 1, 0;
    std::cout<<p->run(x)<<'\n';
    x << 1, 1;
    std::cout<<p->run(x)<<'\n';

    std::cout<<"\n\n--------Hardcoded XOR Example----------------\n\n";
    MultiLayerPerceptron mlp = MultiLayerPerceptron({2,2,1} /*number of neurons in each layer*/, std::deque<std::string>(2, "sigmoid") /*layer types*/, 1.0f /*bias*/, 0.5f /*eta*/);  //mlp: 2 inputs in first layer, 2 neurons in second layer, 1 neuron in the output (third) layer
    Eigen::MatrixXf matrix1(2, 3);
    matrix1 << -10, -10, 15,
                15, 15, -10;
    Eigen::MatrixXf matrix2(1, 3);
    matrix2 << 10, 10, -15;
    // Create the vector of vectors of Eigen::MatrixXf
    std::vector<Eigen::MatrixXf> weights;
    weights.emplace_back(matrix1);
    weights.emplace_back(matrix2);

    // Call set_weights() with the weights
    mlp.set_weights(std::move(weights));

    std::cout << "Hard-coded weights:\n";
    mlp.print_weights();

    std::cout<<"XOR:"<<'\n';
    x << 0, 0;
    std::cout<<mlp.run(x)<<'\n';
    x << 0, 1;
    std::cout<<mlp.run(x)<<'\n';
    x << 1, 0;
    std::cout<<mlp.run(x)<<'\n';
    x << 1, 1;
    std::cout<<mlp.run(x)<<'\n';
    
//    /*
//    XOR
//    ===
//
//     sigmoid(-10*A - 10*B + 15)*10 + sigmoid(15*A + 15*B - 10)*10 - 15
//
//     A | B | output | sigmoid(output)
//     0 | 0 |  -5    | 0.0067
//     1 | 0 |   5    | 0.9923558641717396
//     0 | 1 |   5    | 0.9923558641717396
//     1 | 1 |   -5   | 0.007
//
//     */
    
    std::cout << "Resetting weights:\n";
//    mlp = MultiLayerPerceptron({2,5,6,1});
    mlp.reset_weights();
    mlp.print_weights();
    
    std::cout<<"Training Neural Network as an XOR Gate...\n";
    puts("Press ctrl-c to continue");
    float MSE;
    Eigen::VectorXf x_train(2), y_train(1);
    std::vector<Eigen::VectorXf> x_train_data, y_train_data;
    x_train_data.reserve(4);
    y_train_data.reserve(4);
    
    auto start_time = Clock::now();
    x_train << 0, 0;
    y_train << 0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 0, 1;
    y_train << 1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1, 0;
    y_train << 1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1, 1;
    y_train << 0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    mlp.train(x_train_data, y_train_data, 1000000);
    printf("Time elapsed = %lf\n", timeElapsedSince(start_time));
    std::cout<<"\n\nTrained weights (Compare to hard-coded weights):\n\n";
    mlp.print_weights();
    std::cout<<"XOR:"<<'\n';
    x << 0, 0;
    std::cout<<mlp.run(x)<<'\n';
    x << 0, 1;
    std::cout<<mlp.run(x)<<'\n';
    x << 1, 0;
    std::cout<<mlp.run(x)<<'\n';
    x << 1, 1;
    std::cout<<mlp.run(x)<<'\n';
    
    //100000, 8, Feynman_3, 1.0f, 5.0f
    std::vector<Eigen::VectorXf> data = generateNNData(20, 3, example_func_2, -3.0f, 3.0f);
//    std::vector<Eigen::VectorXf> data = generateNNData(10, 8, example_func_1, 1.0f, 5.0f);
    std::cout << data << '\n';
    x_train_data = leftCols(data, data[0].size() - 1);
    y_train_data = rightCols(data, 1);
    
//    std::unique_ptr<MultiLayerPerceptron> srnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{2,10,5,5,1} /*number of neurons in each layer*/, std::deque<std::string>(4, "sigmoid") /*layer types*/, 1.0f /*bias*/, 0.001f /*eta*/, 0.9f /*theta*/, "NAG");
    std::unique_ptr<MultiLayerPerceptron> srnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{2,10,10,5,5,1} /*number of neurons in each layer*/, std::deque<std::string>{"sigmoid", "sigmoid", "none", "none", "none"} /*layer types*/, 1.0f /*bias*/, 1e-4f /*eta*/, 0.01f /*theta*/, 0.9f /*gamma*/, "NAG" /*weight update rule*/, "prefix" /*expression_type*/, 1e-8f /*epsilon*/);
    //https://stackoverflow.com/a/33980220/18255427 -> Set LR low to prevent nans from gradient blow up

    /*
     Observations
     ============
      - LR (eta) too high (> 1e-3) and all except last layer sigmoid -> all outputs similar
      - LR (eta) too high (> 1e-3) last several layers (like 3 or so) linear following sigmoid layers -> baNANas
     */
    
//    srnn->set_learning_rate(1e-3);
    MSE = srnn->train(x_train_data, y_train_data, 1000000);
    
    std::cout << "SR network MSE: " << MSE << '\n';
    std::cout << "SR Network Prediction: "
    << srnn->predict(x_train_data) << '\n';
    std::cout << "Actual SR Labels: "
    << y_train_data << '\n';
    
//    exit(1);
    //test code - Segment Display Recognition System
    int epochs = 10000;
    
    std::unique_ptr<MultiLayerPerceptron> sdrnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{7,7,1} /*number of neurons in each layer*/, std::deque<std::string>(2, "sigmoid") /*layer types*/, 1.0f /*bias*/, 0.001f /*eta*/, 0.9f /*theta*/, 0.01 /*gamma*/, "sigmoid" /*output_type*/ "NAG" /*weight update rule*/);
    x_train.resize(7);
    x_train_data.clear();
    y_train_data.clear();
    // Dataset for the 7 to 1 network
    
    x_train << 1,1,1,1,1,1,0; y_train << 0.05; //0 pattern
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 0,1,1,0,0,0,0; y_train << 0.15;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,0,1,1,0,1; y_train << 0.25; //2 pattern
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,0,0,1; y_train << 0.35;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 0,1,1,0,0,1,1; y_train << 0.45;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,0,1,1,0,1,1; y_train << 0.55;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,0,1,1,1,1,1; y_train << 0.65;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,0,0,0,0; y_train << 0.75;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,1,1,1; y_train << 0.85;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,0,1,1; y_train << 0.95;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);

    MSE = sdrnn->train(x_train_data, y_train_data, epochs);
    std::cout << '\n' << "7 to 1  network MSE: " << MSE << '\n';
    GetData(7, *sdrnn);
    
    // Dataset for the 7 to 10 network
    sdrnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{7,7,10} /*number of neurons in each layer*/, std::deque<std::string>(2, "sigmoid") /*layer types*/, 1.0f /*bias*/, 0.5f /*eta*/);
    y_train.resize(10);
    x_train_data.clear();
    y_train_data.clear();
    x_train << 1,1,1,1,1,1,0; y_train << 1,0,0,0,0,0,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 0,1,1,0,0,0,0; y_train << 0,1,0,0,0,0,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,0,1,1,0,1; y_train << 0,0,1,0,0,0,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,0,0,1; y_train << 0,0,0,1,0,0,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 0,1,1,0,0,1,1; y_train << 0,0,0,0,1,0,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,0,1,1,0,1,1; y_train << 0,0,0,0,0,1,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,0,1,1,1,1,1; y_train << 0,0,0,0,0,0,1,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,0,0,0,0; y_train << 0,0,0,0,0,0,0,1,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,1,1,1; y_train << 0,0,0,0,0,0,0,0,1,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,0,1,1; y_train << 0,0,0,0,0,0,0,0,0,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    MSE = sdrnn->train(x_train_data, y_train_data, epochs);
    std::cout << "7 to 10 network MSE: " << MSE << '\n';
    GetData(7, *sdrnn);

    // Dataset for the 7 to 7 network
    sdrnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{7,7,7} /*number of neurons in each layer*/, std::deque<std::string>(2, "sigmoid") /*layer types*/, 1.0f /*bias*/, 0.5f /*eta*/);
    y_train.resize(7);
    x_train_data.clear();
    y_train_data.clear();
    x_train << 1,1,1,1,1,1,0; y_train << 1,1,1,1,1,1,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 0,1,1,0,0,0,0; y_train << 0,1,1,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,0,1,1,0,1; y_train << 1,1,0,1,1,0,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,0,0,1; y_train << 1,1,1,1,0,0,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 0,1,1,0,0,1,1; y_train << 0,1,1,0,0,1,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,0,1,1,0,1,1; y_train << 1,0,1,1,0,1,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,0,1,1,1,1,1; y_train << 1,0,1,1,1,1,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,0,0,0,0; y_train << 1,1,1,0,0,0,0;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,1,1,1; y_train << 1,1,1,1,1,1,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    x_train << 1,1,1,1,0,1,1; y_train << 1,1,1,1,0,1,1;
    x_train_data.push_back(x_train);
    y_train_data.push_back(y_train);
    MSE = sdrnn->train(x_train_data, y_train_data, epochs);
    std::cout << "7 to 7  network MSE: " << MSE << '\n';
    GetData(7, *sdrnn);
}

// g++ -Wall -std=c++20 -o NeuralNetworks_Vec NeuralNetworks_Vec.cpp MLP_Vec.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -ffast-math -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native
