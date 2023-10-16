import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from sympy import latex
from copy import copy

import numpy as np
from tqdm import tqdm

from Arena_symreg import Arena
from MCTS_symbreg import MCTS
from symreg.SymRegLogic import Board

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
#    scores = []
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.iteration_numbers = []
        self.unique_expression_counts = []
        self.scores = []

    def executeEpisode(self):
        """
        This function executes one episode of self-play
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        episodeStep = 0
#        print("New iteration started")
        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            r = self.game.getGameEnded(board)
            pi = self.mcts.getActionProb(board, temp=temp) #Here, self.args['numMCTSSims'] games are simulated from the state board
            trainExamples.append(([copy(board), copy(pi)]))
            action = np.random.choice(len(pi), p=pi)
            board = self.game.getNextState(board, action)
            r = self.game.getGameEnded(board)
            if r != -1:
                #get length of last board
                expression_length = len(trainExamples[-1][0]) #board of last "trainExample"
                #append reward to each trainExample
                for i in range(len(trainExamples)):
                    trainExamples[i].append(r)
                    trainExamples[i][0].extend([0]*(expression_length-len(trainExamples[i][0])))
                return trainExamples

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if score >= self.args.best_score
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
#            if i == 51:
#                raise KeyboardInterrupt
            log.info(f'Starting Iter #{i}, # of search expressions = {Board.expression_dict_len}...')
            self.iteration_numbers.append(i)
            self.unique_expression_counts.append(Board.expression_dict_len)
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                self.mcts.iteration_number = i 
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    iterationTrainExamples += self.executeEpisode()
                # save the iteration examples to the history
                
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
#            self.saveTrainExamples(i - 1)
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            
            self.nnet.train(trainExamples) #Machine learning model is getting trained on the saved training examples
            self.mcts.nnet = self.nnet

#            log.info('PITTING AGAINST PREVIOUS VERSION')
#            arena = Arena(lambda x: np.argmax(self.mcts.getActionProb(x, temp=0)), self.game)
#            score = arena.playGames(self.args.arenaCompare)
#
#            log.info(f'Score / Ideal: {score} / {self.args.arenaCompare}')
#            
#            self.scores.append(score)
#            
#            if score < self.args.bestScore:
#                log.info('REJECTING NEW MODEL')
#                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
#            else:
#                log.info('ACCEPTING NEW MODEL')
#                self.args.bestScore = score
#                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
#                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
#            
        print(f"Best expression: {Board.best_expression}")
        print(f"Best expression latex: {latex(Board.best_expression)}")
        print(f"Best loss: {Board.best_loss:.3f}")

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f, protocol = 4).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        print(modelFile)
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
