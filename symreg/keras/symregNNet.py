import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from keras2tikzClone.model_to_tex import gen_tikz_from_model
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

"""
NeuralNet for the game of Symbolic Regression.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloNNet by SourKream and Surag Nair.
"""
class symregNNet():
    def __init__(self, game, args):
        # game params
        self.board = game.getBoardSize()  # a number
        self.action_size = game.getActionSize()
        self.args = args

        if self.args['useNN']:
            # Neural Net
            self.input_boards = Input(shape=(self.board, 1))  # s: batch_size x board x 1

            # Adjust the convolutional layers for 1D data
            h_conv1 = Activation('relu')(BatchNormalization(axis=-1)(Conv1D(args.num_channels, 3, padding='same')(self.input_boards)))  # batch_size x board x num_channels
            h_conv1_flat = Flatten()(h_conv1)
            s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv1_flat))))  # batch_size x 1024
            self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc1)   # batch_size x self.action_size
            self.v = Dense(1, activation='softmax', name='v')(s_fc1)                    # batch_size x 1

            self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
            self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
            
            print(gen_tikz_from_model(self.model))
            self.model.summary()
            
            #Speed pretty slow, probably because of the size of the network, and each epoch takes longer with increasing iterations. The policy estimate did not improve overall, but the value error went from 1 (maximum possible) to 0.8, so some improvement. Score generally improved with time, see Model_History_Multi_Dimension_NN.png and best_multi_dim_expression_NN.txt
        else:
#            self.clf_value = make_pipeline(StandardScaler(), SVR(gamma='auto', epsilon=0.1, kernel = 'rbf', degree = 3)) #Speed ok but gets worse as data increases, R^2 sometimes ok sometimes bad, Score generally improves with time, see Model_History_Multi_Dimension_SVR.png and best_multi_dim_expression_SVR.txt. Found exact expression at iteration 106
#            self.clf_value = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, loss = "squared_error", learning_rate = "invscaling", eta0 = 0.01)) #Speed good, R^2 ok, Score improves with time, see Model_History_Multi_Dimension_SGDRegressor.png and best_multi_dim_expression_SGDRegressor.txt. Found exact expression at iteration 173
            self.clf_value = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3, n_jobs = None)) #fast to train but slow to predict (though speed seems to improve with time), very good R^2, score improves over time, see Model_History_Multi_Dimension_KNeighborsRegressor.png and best_multi_dim_expression_KNeighborsRegressor.txt, Recovered exact expression at iteration 55. Arena game Score 1.0/1.0 achieved at iteration 69. TODO: Can set njobs to -1 to run in parallel with maximum number of processors available, see:  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html!
#            self.clf_value = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel = DotProduct() + WhiteKernel(), random_state=0)) #impractically slow, ok R^2 that improves with increasing number of iterations, Score simprove slowly, see Model_History_Multi_Dimension_GaussianProcessRegressor.png and best_multi_dim_expression_GaussianProcessRegressor.txt
#            self.clf_value = make_pipeline(StandardScaler(), LinearSVR(dual="auto", random_state=0, tol=1e-5, max_iter = 10000)) #Speed ok but gets worse as data increases, bad R^2 that doesn't seem to improve (policy estimation constantly masked), score doesn't seem to improve well/fluctuates, see Model_History_Multi_Dimension_LinearSVR.png and best_multi_dim_expression_LinearSVR.txt
#            self.clf_value = make_pipeline(StandardScaler(), PLSRegression()) #Speed good, R^2 starts off ok and seems to improve, Score seems to improve, see Model_History_Multi_Dimension_PLSRegression.png and  best_multi_dim_expression_PLSRegression.txt
#            self.clf_value = make_pipeline(StandardScaler(), DecisionTreeRegressor()) #Speed good, R^2 nearly perfect, Score ok, see Model_History_Multi_Dimension_DecisionTreeRegressor.png and best_multi_dim_expression_DecisionTreeRegressor.txt
#            self.clf_value = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth = 3)) #Speed good, R^2 good, Score ok, see Model_History_Multi_Dimension_DecisionTreeRegressor_max_depth_3.png and best_multi_dim_expression_DecisionTreeRegressor_max_depth_3.txt
#            self.clf_value = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators = 10, max_depth = 3)) #Very slow at first but speed seems to increase slowly with the number of iterations, R^2 ok and gets better with time, score improves slowly with time, see Model_History_Multi_Dimension_RandomForestRegressor_n_estimators_10.png and best_multi_dim_expression_RandomForestRegressor_n_estimators_10. Recovered exact expression at iteration 12.
#            self.clf_value = make_pipeline(StandardScaler(), AdaBoostRegressor(n_estimators = 10)) #Very slow at first but speed seems to increase slowly with the number of iterations, R^2 ok and gets better with time, score improves slowly with time, see Model_History_Multi_Dimension_AdaBoostRegressor_n_estimators_10.png and Model_History_Multi_Dimension_AdaBoostRegressor_n_estimators_10.txt
#            self.clf_value = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators = 10)) #Slow but not as slow as RandomForestRegressor or AdaBoostRegressor and doesn't necessarily get slower with more training data (in fact it seems to speed up), R^2 ok and gets better with time, Score ok and seems to increase over time significantly (but not gradually, rather abruptly, but stable), see Model_History_Multi_Dimension_GradientBoostingRegressor_n_estimators_10.png and best_multi_dim_expression_GradientBoostingRegressor_n_estimators_10.txt
#            self.clf_value = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators = 100)) #similar situation as above but slower to converge, see Model_History_Multi_Dimension_GradientBoostingRegressor_n_estimators_100.png and best_multi_dim_expression_GradientBoostingRegressor_n_estimators_100.txt
            self.clf_policy = []
            for i in range(self.action_size):
#                self.clf_policy.append(make_pipeline(StandardScaler(), SVR(gamma='auto', epsilon=0.1, kernel = 'rbf', degree = 3)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3, loss = "squared_error", learning_rate = "invscaling", eta0 = 0.01)))
                self.clf_policy.append(make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3, n_jobs = None)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel = DotProduct() + WhiteKernel(), random_state=0)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), LinearSVR(dual="auto", random_state=0, tol=1e-5, max_iter = 10000)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), PLSRegression()))
#                self.clf_policy.append(make_pipeline(StandardScaler(), DecisionTreeRegressor()))
#                self.clf_policy.append(make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth = 3)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators = 10, max_depth = 3)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), AdaBoostRegressor(n_estimators = 10)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators = 10)))
#                self.clf_policy.append(make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators = 100)))
#
#mv Model_History_Multi_Dimension.png Model_History_Multi_Dimension_NN.png
#mv best_multi_dim_expression.txt best_multi_dim_expression_NN.txt
