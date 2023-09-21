import os
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

from .symregNNet import symregNNet as onnet

"""
NeuralNet wrapper class for the TicTacToeNNet.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on (copy-pasted from) the NNet by SourKream and Surag Nair.
"""

args = dotdict({
    'useNN' : False,
    'lr': 0.01,
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = onnet(game, args)
        self.board = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.trained = False

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
#        print("input_boards, target_pis, target_vs =",input_boards, target_pis, target_vs, sep = "\n======\n")
        input_boards = pad_sequences(input_boards, padding='post') #np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        if args['useNN']:
            self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        else:
            self.example_length = len(input_boards[0])
            self.nnet.clf_value.fit(input_boards, target_vs) #learning value function
            print("Value R^2 score =",self.nnet.clf_value.score(input_boards, target_vs))
            assert(len(self.nnet.clf_policy) == self.action_size)
            for i in range(self.action_size):
                self.nnet.clf_policy[i].fit(input_boards, target_pis[:,i])
                print(f"Policy {i} R^2 Score =",self.nnet.clf_policy[i].score(input_boards, target_pis[:,i]))

#            # Initialize the policy and value models
#            self.nnet.clf_value.fit(input_boards, target_vs)  # Initialize the value function model
#
#            # Get predictions from the current value model
#            predicted_vs = self.nnet.clf_value.predict(input_boards)
#            print("Value R^2 score =",self.nnet.clf_value.score(input_boards, target_vs))
#
#            # Initialize empty lists to store data for policy model updates
#            input_boards_policy = []
#            target_pis_policy = []
#
#            # Generate data for policy model updates using the value model's predictions
#            for i in range(self.action_size):
#                input_boards_policy.append(input_boards)
#                target_pis_policy.append(target_pis[:, i] + predicted_vs)
#
#            # Update the policy models
#            for i in range(self.action_size):
#                self.nnet.clf_policy[i].fit(input_boards_policy[i], target_pis_policy[i])
#                print(f"Policy {i} R^2 Score =",self.nnet.clf_policy[i].score(input_boards, target_pis_policy[i]))
#            # Get predictions from the updated policy models
#            predicted_pis = [self.nnet.clf_policy[i].predict(input_boards) for i in range(self.action_size)]
#
#            # Update the value model using the policy model's predictions
#            self.nnet.clf_value.fit(input_boards, target_vs + sum(predicted_pis))
        
        #(1.087411495459077, 0.40460654772803967, 2.604881885431175, 0.4046065477280398, 2.724222366673192): mean = 1.4451457686039046, var = 1.0548884709892918
        #(1.887357057178798, 1.7082022595628767, 1.6186210489487627, 1.087411495459078, 1.8275157108414692): mean = 1.625821514398197, var = 0.08116103427740798
        
        #(3.14352726376052, 2.7652405406897542, 2.4581126451405684, 2.466750465608598, 1.7737176969099686): mean = 2.521469722421882, var = 0.20250424090805136
        #(3.114643407056879, 3.2386341425429404, 2.9983956706047525, 2.0664457931297287e-31, 3.1445277675119865): mean = 2.4992401975433114, var = 1.5674356630882627
            
        if not self.trained:
            self.trained = True

    def predict(self, board):
        """
        board: np array with board
        """
        
        # run
        if args['useNN']:
            pi, v = self.nnet.model.predict(board, verbose=False) #pi is the NN predicted probability of selecting each possible action given the state s
        elif self.trained:
#            print("BEFORE: len(board) == self.example_length is {}, len(board) = {}, self.example_length = {}".format(len(board) == self.example_length, len(board), self.example_length))

            
            board_length = len(board)
            
            diff = board_length - self.example_length
            #Below, we are making board have the same length as self.example_length, i.e., the
            #length the supervised learning models were trained on.
            board = board[:-diff] if diff > 0 else (board + [0]*(self.example_length-len(board))) if diff < 0 else board

            v = self.nnet.clf_value.predict([board])
            pi = [np.empty(self.action_size)]
            for i in range(self.action_size):
                pi[0][i] = self.nnet.clf_policy[i].predict([board])
        else:
            v = [np.random.random(1)]
            pi = [np.random.random(self.action_size)]
            
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        if args['useNN']:
            filename = filename.split(".")[0] + ".h5"

            filepath = os.path.join(folder, filename)
            if not os.path.exists(folder):
                print("Checkpoint Directory does not exist! Making directory {}".format(folder))
                os.mkdir(folder)
            else:
                print(f"Checkpoint Directory {filepath} exists!")
            self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        if args['useNN']:
            filename = filename.split(".")[0] + ".h5"

            # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
            filepath = os.path.join(folder, filename)
            if not os.path.exists(filepath):
                raise ValueError("No model in path '{}'".format(filepath))
            self.nnet.model.load_weights(filepath)



