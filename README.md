# Generalized Prefix and Postfix Faultless, Fixed-Depth Grammars in Symbolic Regression


## Prerequisites

Make sure you have the following prerequisites installed before compiling the script:

- [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) library
- [LBFGS++](https://github.com/yixuan/LBFGSpp) library

## Compilation

Use the provided compilation command to build the executable:

```bash
g++ -std=c++20 -o PrefixPostfixSR PrefixPostfixSR.cpp -O2 -I/opt/homebrew/opt/eigen/include/eigen3 -I/Users/username/LBFGSpp -ffast-math -ftree-vectorize
```

## Usage

After compiling, run the executable:

```bash
./PrefixPostfixSR
```

## License

This project is licensed under the [MIT License](LICENSE).

