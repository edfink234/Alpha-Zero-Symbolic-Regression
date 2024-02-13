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

## Hardware Specifications

The benchmarks were run on a MacBook Pro with an M1 Core and approximately 16 GB of usable RAM, namely, `sysctl -a | grep hw.memsize` gives:

```
hw.memsize: 17179869184
hw.memsize_usable: 16383606784
```

The macOS system version information is obtained with `sw_vers`:

```
ProductName:		macOS
ProductVersion:		14.2.1
BuildVersion:		23C71
```



## License

This project is licensed under the [MIT License](LICENSE).

