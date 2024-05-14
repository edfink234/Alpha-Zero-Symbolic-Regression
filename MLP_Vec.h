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
#include <random>
#include <float.h>

class Perceptron
{
    public:
        Eigen::VectorXf weights;
        std::string output_type; 
        float bias;
        bool is_output;
        Perceptron(int inputs, float bias=1.0f, bool is_output = false, std::string&& output_type = "none");
        float run(const Eigen::VectorXf& x);
        void set_weights(const Eigen::VectorXf& w_init);
        static float sigmoid(float x);
        float scale_between(float unscaled_num, float min = 0.0f, float max = 1.0f, float max_allowed = FLT_MAX, float min_allowed = FLT_MIN);
};

class MultiLayerPerceptron
{
    public:
        static std::unordered_map<float, std::string> inline __tokens_dict; //Converts number to string
        static std::vector<float> inline __operators_float;
        static std::vector<float> inline __unary_operators_float;
        std::vector<float> pieces;

        MultiLayerPerceptron(std::vector<int> layers, float bias=1.0f, float eta = 0.5f, std::string&& output_type = "sigmoid", const std::string& expression_type = "prefix", const std::string& weight_update = "basic");
        void set_weights(std::vector<Eigen::MatrixXf>&& w_init);
        void reset_weights();
        void print_weights();
        Eigen::VectorXf run(const Eigen::VectorXf& x);
        static float mse(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
        float bp(const Eigen::VectorXf& x, const Eigen::VectorXf& y);
        float train(const std::vector<Eigen::VectorXf>& x_train, const std::vector<Eigen::VectorXf>& y_train, const unsigned long num_epochs = 0, bool interactive = true);
        std::vector<int> layers; //# of neurons per layer including the input layer (in which case layers[0] refers to the number of inputs)
        static void signalHandler(int signum);
        void set_learning_rate(float eta) {this->eta = eta;}
        std::vector<Eigen::VectorXf> predict(const std::vector<Eigen::VectorXf>&);
        static std::vector<Eigen::VectorXf> sigmoid(const std::vector<Eigen::VectorXf>&);
        float expression_evaluator(float w_k = 0.0f, float eta = 0.0f, float d_ij = 0.0f, float value = 0.0f, const Eigen::VectorXf& params = {});
        
    private:
        float bias;
        float eta; //learning rate
        std::vector<std::vector<Perceptron> > network; //the actual network
        std::vector<Eigen::VectorXf> values; //holds output values of the neurons
        std::vector<Eigen::VectorXf> d; //contains error terms for neurons: one error term for each neuron of each layer
        std::string output_type;
        static volatile sig_atomic_t inline interrupted = 0;
        std::string expression_type;
        std::string weight_update;
};

#endif
