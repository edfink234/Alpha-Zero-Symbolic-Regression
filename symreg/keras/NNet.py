import os
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet
import threading

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
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 100,
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
        input_boards = pad_sequences(input_boards, padding='post', maxlen = self.board if args['useNN'] else None)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        with threading.Lock():
            if args['useNN']:
                self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)
            else:
                self.example_length = len(input_boards[0])
                self.nnet.clf_value.fit(input_boards, target_vs) #learning value function
                for i in range(self.action_size):
                    self.nnet.clf_policy[i].fit(input_boards, target_pis[:,i])
            
        if not self.trained:
            self.trained = True

    def predict(self, board):
        
        # run
        if args['useNN']:
            pi, v = self.nnet.model.predict(pad_sequences([board], padding='post', maxlen = self.board), verbose=False) #pi is the NN predicted probability of selecting each possible action given the state s
        elif self.trained:
            board_length = len(board)
            
            diff = board_length - self.example_length
            #Below, we are making board have the same length as self.example_length, i.e., the
            #length the supervised learning models were trained on.
            board = board[:-diff] if diff > 0 else (board + [0]*(self.example_length-len(board))) if diff < 0 else board

            v = self.nnet.clf_value.predict([board])
#            pi = [np.empty(self.action_size)]
#            for i in range(self.action_size):
#                pi[0][i] = self.nnet.clf_policy[i].predict([board])
        else:
            v = [np.random.random(1)]
#            pi = [np.random.random(self.action_size)]
            
        return v[0]#pi[0], v[0]

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



