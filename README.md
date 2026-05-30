# Searching the Space of Feed-Forward Neural-Network Weight Update Rules with Fixed Depth Symbolic Regression

## Prerequisites

Make sure you have the following prerequisites installed before compiling the script:

- [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) library
- [LBFGS++](https://github.com/yixuan/LBFGSpp) library
- [Boost](https://www.boost.org/) library

## Compilation

Use the provided compilation command to build the executable:

```bash
g++ -Wall -std=c++20 -o NeuralNetworks_VecSR NeuralNetworks_VecSR.cpp MLP_Vec.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/edwardfinkelstein/LBFGSpp -ftree-vectorize -L/opt/homebrew/Cellar/boost/1.84.0 -I/opt/homebrew/Cellar/boost/1.84.0/include -march=native
```

## Usage

After compiling, run the executable:

```bash
./NeuralNetworks_VecSR
```

## Description of Code files
 - `NeuralNetworks_VecSR.cpp`: Implements the genetic evolution of candidate symbolic-regression weight-udpdate rules
 - `MLP Vec.h` and `MLP Vec.cpp`: Header and source files for the Multilayer-perceptron class and weight-update rule logic
 - `RunTestsNeuralNetworksVecSRPart1.py`: Conducts the hyper-parameter grid-sweep for the established weight-update rules considered (Gradient-Descent, Heavy-Ball, Nesterov-Accelerated Gradient Descent, AdaGrad, RMSProp, AdaDelta, Adam, and AdamW)
 - `RunTestsNeuralNetworksVecSRPart2.py`: Executes the 30 benchmarks, comprising 10 symbolic-regression benchmark-expressions for 3 simple neural-network architectures
 - `extract_best_mses.py`: Post-processing file to extract the best mean-squared error values obtained from the best-established weight-update rule hyperparameter combination for each of the 30 benchmarks.
 - `visualize_tree.py`: Helper script to produce expression trees of the established weight-update rules considered (Gradient-Descent, Heavy-Ball, Nesterov-Accelerated Gradient Descent, AdaGrad, RMSProp, AdaDelta, Adam, and AdamW), for example.


## License

This project is licensed under the [MIT License](LICENSE).

