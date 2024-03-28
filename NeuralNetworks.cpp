// NeuralNetworks.cpp : This file contains the 'main' function. Program execution begins and ends there.//

#include <iostream>
#include <memory>
#include <float.h>
#include "MLP.h"
#include <csignal>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using VectorRowMajorXf = Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::RowMajor>;

// Define a flag to control the loop
volatile sig_atomic_t interrupted = 0;

template <typename T>
double timeElapsedSince(T start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_time).count()/1e9;
}

// Signal handler function to catch Ctrl-C
void signalHandler(int signum) 
{
    if (signum == SIGINT) 
    {
        interrupted = 1;
    }
}

int getNum(const Eigen::VectorXf& results)
{
    if (results.size() == 1)
    {
        return static_cast<int>(results[0]*10);
    }
    else if (results.size() == 10)
    {
        return std::distance(results.begin(), std::max_element(results.begin(), results.end()));
    }
    else
    {
        Eigen::Matrix<float, 10, 7> number_vecs
        {
            {1,1,1,1,1,1,0},
            {0,1,1,0,0,0,0},
            {1,1,0,1,1,0,1},
            {1,1,1,1,0,0,1},
            {0,1,1,0,0,1,1},
            {1,0,1,1,0,1,1},
            {1,0,1,1,1,1,1},
            {1,1,1,0,0,0,0},
            {1,1,1,1,1,1,1},
            {1,1,1,1,0,1,1},
        };
        
        int idx = 0;
        float mse_val = DBL_MAX, temp_mse_val;
        
        for (size_t i = 0; i < number_vecs.rows(); i++)
        {
            temp_mse_val = MultiLayerPerceptron::mse(results, number_vecs.row(i));
            if (temp_mse_val == 0)
            {
                return i;
            }
            else if (temp_mse_val < mse_val)
            {
                idx = i;
                mse_val = temp_mse_val;
            }
        }
        return idx;
    }
}

/*
    ===========================
    Segment Display Recognition
    ===========================
 
                a
             |-----|
            f|  g  |b
             |-----|
            e|     |c
             |-----|
                d

    Possible Patterns
    -----------------
 
    {1,1,1,1,1,1,0} //0 pattern
 
            a
         |-----|
        f|     |b
         |     |
        e|     |c
         |-----|
            d
 
 
    {0,1,1,0,0,0,0} //1 pattern
 
               |
               |b
               |
               |c
               |
 
    {1,1,0,1,1,0,1} //2 pattern
            
            a
         |-----|
            g  |b
         |-----|
        e|
         |-----|
            d
 
    {1,1,1,1,0,0,1} //3 pattern
 
            a
         |-----|
            g  |b
         |-----|
               |c
         |-----|
            d
 
    {0,1,1,0,0,1,1} //4 pattern
            
         |     |
        f|  g  |b
         |-----|
               |c
               |

    {1,0,1,1,0,1,1} //5 pattern
 
            a
         |-----|
        f|  g
         |-----|
               |c
         |-----|
            d
 
    {1,0,1,1,1,1,1} //6 pattern
 
            a
         |-----|
        f|  g
         |-----|
        e|     |c
         |-----|
            d
 
    {1,1,1,0,0,0,0} //7 pattern
 
            a
         |-----|
               |b
               |
               |c
               |

    {1,1,1,1,1,1,1} //8 pattern
 
            a
         |-----|
        f|  g  |b
         |-----|
        e|     |c
         |-----|
            d
 
    {1,1,1,1,0,1,1} //9 pattern
 
            a
         |-----|
        f|  g  |b
         |-----|
               |c
         |-----|
            d
 */

void GetData(int numInputs, MultiLayerPerceptron& sdrnn)
{
    char ans;
    int numOutputs = sdrnn.layers.back();
    printf("Would you like to test your SDR neural network with %d inputs and %d outputs (y/n)? ", numInputs, numOutputs);
    std::cin >> ans;
    if (ans != 'y')
    {
        return;
    }
    while (ans == 'y')
    {
        puts("Time to enter your test case!");
        Eigen::VectorXf inputs(numInputs);
        for (int i = 0; i < numInputs; i++)
        {
            printf("Enter input %d of %d: ", i+1, numInputs);
            std::cin >> inputs[i];
        }
        printf("Your SDR neural network's prediction is %d\n", getNum(sdrnn.run(inputs)));
        printf("Would you like to test your SDR neural network with %d inputs and %d outputs again (y/n)? ", numInputs, numOutputs);
        std::cin >> ans;
    }
}


int main() {
    srand(time(NULL));
    rand();
    // Set up the signal handler
    signal(SIGINT, signalHandler);

    std::cout << "\n\n--------Logic Gate Example----------------\n\n";

    std::unique_ptr<Perceptron> p = std::make_unique<Perceptron>(2); //2 inputs
    
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
    MultiLayerPerceptron mlp = MultiLayerPerceptron({2,2,1});  //mlp: 2 inputs in first layer, 2 neurons in second layer, 1 neuron in the output (third) layer
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
    auto start_time = Clock::now();
    for (int i = 0; i < 1000000; i++)
    {
        MSE = 0.0;
        x_train << 0, 0;
        y_train << 0;
        MSE += mlp.bp(x_train, y_train);
        x_train << 0, 1;
        y_train << 1;
        MSE += mlp.bp(x_train, y_train);
        x_train << 1, 0;
        y_train << 1;
        MSE += mlp.bp(x_train, y_train);
        x_train << 1, 1;
        y_train << 0;
        MSE += mlp.bp(x_train, y_train);
        MSE = MSE / 4.0;
        if (i % 100 == 0)
        {
            std::cout<<"MSE = "<<MSE<< '\r' << std::flush;
        }
        
        // Check if interrupted flag is set, and break out of the loop if true
        if (interrupted)
        {
            std::cout << "\nInterrupted by Ctrl-C. Exiting loop.\n";
            break;
        }
    }
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

    //test code - Segment Display Recognition System
    int epochs = 1000;
    
    std::unique_ptr<MultiLayerPerceptron> sdrnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{7,7,1});
    x_train.resize(7);
    // Dataset for the 7 to 1 network
    for (int i = 0; i < epochs; i++)
    {
        MSE = 0.0;
        x_train << 1,1,1,1,1,1,0; y_train << 0.05;
        MSE += sdrnn->bp(x_train, y_train); //0 pattern
        x_train << 0,1,1,0,0,0,0; y_train << 0.15;
        MSE += sdrnn->bp(x_train, y_train); //1 pattern
        x_train << 1,1,0,1,1,0,1; y_train << 0.25;
        MSE += sdrnn->bp(x_train, y_train); //2 pattern
        x_train << 1,1,1,1,0,0,1; y_train << 0.35;
        MSE += sdrnn->bp(x_train, y_train); //3 pattern
        x_train << 0,1,1,0,0,1,1; y_train << 0.45;
        MSE += sdrnn->bp(x_train, y_train); //4 pattern
        x_train << 1,0,1,1,0,1,1; y_train << 0.55;
        MSE += sdrnn->bp(x_train, y_train); //5 pattern
        x_train << 1,0,1,1,1,1,1; y_train << 0.65;
        MSE += sdrnn->bp(x_train, y_train); //6 pattern
        x_train << 1,1,1,0,0,0,0; y_train << 0.75;
        MSE += sdrnn->bp(x_train, y_train); //7 pattern
        x_train << 1,1,1,1,1,1,1; y_train << 0.85;
        MSE += sdrnn->bp(x_train, y_train); //8 pattern
        x_train << 1,1,1,1,0,1,1; y_train << 0.95;
        MSE += sdrnn->bp(x_train, y_train); //9 pattern
    }
    MSE /= 10.0;
    std::cout << '\n' << "7 to 1  network MSE: " << MSE << '\n';
    GetData(7, *sdrnn);
    
    // Dataset for the 7 to 10 network
    sdrnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{7,7,10});
    y_train.resize(10);
    for (int i = 0; i < epochs; i++)
    {
        MSE = 0.0;
        x_train << 1,1,1,1,1,1,0; y_train << 1,0,0,0,0,0,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //0 pattern
        x_train << 0,1,1,0,0,0,0; y_train << 0,1,0,0,0,0,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //1 pattern
        x_train << 1,1,0,1,1,0,1; y_train << 0,0,1,0,0,0,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //2 pattern
        x_train << 1,1,1,1,0,0,1; y_train << 0,0,0,1,0,0,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //3 pattern
        x_train << 0,1,1,0,0,1,1; y_train << 0,0,0,0,1,0,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //4 pattern
        x_train << 1,0,1,1,0,1,1; y_train << 0,0,0,0,0,1,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //5 pattern
        x_train << 1,0,1,1,1,1,1; y_train << 0,0,0,0,0,0,1,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //6 pattern
        x_train << 1,1,1,0,0,0,0; y_train << 0,0,0,0,0,0,0,1,0,0;
        MSE += sdrnn->bp(x_train, y_train); //7 pattern
        x_train << 1,1,1,1,1,1,1; y_train << 0,0,0,0,0,0,0,0,1,0;
        MSE += sdrnn->bp(x_train, y_train); //8 pattern
        x_train << 1,1,1,1,0,1,1; y_train << 0,0,0,0,0,0,0,0,0,1;
        MSE += sdrnn->bp(x_train, y_train); //9 pattern
    }
    MSE /= 10.0;
    std::cout << "7 to 10 network MSE: " << MSE << '\n';
    GetData(7, *sdrnn);

    // Dataset for the 7 to 7 network
    sdrnn = std::make_unique<MultiLayerPerceptron>(std::vector<int>{7,7,7});
    y_train.resize(7);
    for (int i = 0; i < epochs; i++)
    {
        MSE = 0.0;
        x_train << 1,1,1,1,1,1,0; y_train << 1,1,1,1,1,1,0;
        MSE += sdrnn->bp(x_train, y_train); //0 pattern
        x_train << 0,1,1,0,0,0,0; y_train << 0,1,1,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //1 pattern
        x_train << 1,1,0,1,1,0,1; y_train << 1,1,0,1,1,0,1;
        MSE += sdrnn->bp(x_train, y_train); //2 pattern
        x_train << 1,1,1,1,0,0,1; y_train << 1,1,1,1,0,0,1;
        MSE += sdrnn->bp(x_train, y_train); //3 pattern
        x_train << 0,1,1,0,0,1,1; y_train << 0,1,1,0,0,1,1;
        MSE += sdrnn->bp(x_train, y_train); //4 pattern
        x_train << 1,0,1,1,0,1,1; y_train << 1,0,1,1,0,1,1;
        MSE += sdrnn->bp(x_train, y_train); //5 pattern
        x_train << 1,0,1,1,1,1,1; y_train << 1,0,1,1,1,1,1;
        MSE += sdrnn->bp(x_train, y_train); //6 pattern
        x_train << 1,1,1,0,0,0,0; y_train << 1,1,1,0,0,0,0;
        MSE += sdrnn->bp(x_train, y_train); //7 pattern
        x_train << 1,1,1,1,1,1,1; y_train << 1,1,1,1,1,1,1;
        MSE += sdrnn->bp(x_train, y_train); //8 pattern
        x_train << 1,1,1,1,0,1,1; y_train << 1,1,1,1,0,1,1;
        MSE += sdrnn->bp(x_train, y_train); //9 pattern
    }
    MSE /= 10.0;
    std::cout << "7 to 7  network MSE: " << MSE << '\n' << '\n';
    GetData(7, *sdrnn);
    
}
