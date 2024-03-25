#ifndef MLP_H
#define MLP_H
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>
#include <initializer_list>
#include <Eigen/Core>

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

class MultiLayerPerceptron 
{
	public:
		MultiLayerPerceptron(std::vector<int> layers, float bias=1.0f, float eta = 0.5f);
//		void set_weights(std::vector<std::vector<std::vector<float> > > w_init);
//        void reset_weights();
//		void print_weights();
//		std::vector<float> run(std::vector<float> x);
//        static float mse(const std::vector<float> &y, const std::vector<float>& o);
//		float bp(std::vector<float> x, std::vector<float> y);
//		
		std::vector<int> layers; //# of neurons per layer including the input layer (in which case layers[0] refers to the number of inputs)
		float bias;
		float eta; //learning rate
		std::vector<std::vector<Perceptron> > network; //the actual network
        std::vector<Eigen::VectorXf> values; //holds output values of the neurons
		std::vector<std::vector<float> > d; //contains error terms for neurons: one error term for each neuron of each layer
};

#endif
