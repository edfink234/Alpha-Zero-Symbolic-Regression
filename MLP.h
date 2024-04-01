#ifndef MLP_H
#define MLP_H
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>
#include <Eigen/Core>
#include <random>

class Perceptron
{
	public:
        Eigen::VectorXf weights;
		float bias;
		Perceptron(int inputs, float bias=1.0);
        float run(const Eigen::VectorXf& x);
		void set_weights(const Eigen::VectorXf& w_init);
		float sigmoid(float x);
};

class PerceptronLayer
{
    public:
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> weights;
        Eigen::VectorXf biases;
        PerceptronLayer(int inputs, int num_neurons, std::vector<float>&& bias_vec = {});
        Eigen::VectorXf run(const Eigen::VectorXf& x);
        Eigen::VectorXf sigmoid(const Eigen::VectorXf& x);
};


class MultiLayerPerceptron 
{
	public:
		MultiLayerPerceptron(std::vector<int> layers, float bias=1.0f, float eta = 0.5f);
		void set_weights(std::vector<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>&& w_init);
//        void reset_weights();
		void print_weights();
        Eigen::VectorXf run(const Eigen::VectorXf& x);
//        static float mse(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
//		float bp(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
//        float bp(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& y);
//        float train(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& x_train, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& y_train, unsigned long num_epochs = 10);
        static void signalHandler(int signum);
//
		std::vector<int> layers; //# of neurons per layer including the input layer (in which case layers[0] refers to the number of inputs)
		float bias;
		float eta; //learning rate
		std::vector<PerceptronLayer> network; //the actual network
        std::vector<Eigen::VectorXf> values; //holds output values of the neurons
		std::vector<Eigen::VectorXf> d; //contains error terms for neurons: one error term for each neuron of each layer
        static volatile sig_atomic_t inline interrupted = 0;
};

#endif
