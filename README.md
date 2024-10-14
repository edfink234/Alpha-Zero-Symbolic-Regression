# Solving the 2D Advection-Diffusion Equation using Fixed-Depth Symbolic Regression and Symbolic Differentiation without Expression Trees


## Prerequisites

Make sure you have the following prerequisites installed before compiling the script:

- [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) library
- [LBFGS++](https://github.com/yixuan/LBFGSpp) library
- [Boost](https://www.boost.org/) library

## Compilation

Use the provided compilation command to build the executable:

```bash
g++ -Wall -std=c++20 -o PrefixPostfixMultiThreadDiffSimplifySR PrefixPostfixMultiThreadDiffSimplifySR.cpp -O2 \
    -I<path_to_eigen_include> -I<path_to_LBFGSpp_include> -L<path_to_boost_lib> -I<path_to_boost_include> \
    -ffast-math -ftree-vectorize -march=native
```

## Usage

After compiling, run the executable:

```bash
./PrefixPostfixMultiThreadDiffSimplifySR
```

## Hardware Specifications

The tests were run on a MacBook Pro with an M1 Core and approximately 16 GB of usable RAM, namely, `sysctl -a | grep hw.memsize` gives:

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

