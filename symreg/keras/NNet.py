import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

import argparse
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
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        if args['useNN']:
            self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
        else:
            self.nnet.clf_value.fit(input_boards, target_vs) #learning value function
            print("Value R^2 score =",self.nnet.clf_value.score(input_boards, target_vs))
            assert(len(self.nnet.clf_policy) == self.action_size)
            self.nnet.clf_value.fit(input_boards, target_vs)
            for i in range(self.action_size):
                self.nnet.clf_policy[i].fit(input_boards, target_pis[:,i])
                print(f"Policy {i} R^2 Score =",self.nnet.clf_policy[i].score(input_boards, target_pis[:,i]))
        if not self.trained:
            self.trained = True

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :]

        # run
        if args['useNN']:
            pi, v = self.nnet.model.predict(board, verbose=False) #pi is the NN predicted probability of selecting each possible action given the state s
        elif self.trained:
            v = self.nnet.clf_value.predict(board)
            pi = [np.empty(self.action_size)]
            for i in range(self.action_size):
                pi[0][i] = self.nnet.clf_policy[i].predict(board)
        else:
            v = [np.random.random(1)]
            pi = [np.random.random(self.action_size)]
            
#        print("pi[0], v[0] =", type(pi[0]), type(v[0]))
        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
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

