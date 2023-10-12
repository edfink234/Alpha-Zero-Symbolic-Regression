import logging
import coloredlogs
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from Coach_symreg import Coach
from symreg.SymRegGame import SymRegGame as Game
from symreg.SymRegLogic import Board
from symreg.keras.NNet import NNetWrapper as nn
from utils import *
from time import time
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100000,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        # This means that the temperature 'temp' will be 1 for 15 Monte-Carlo simulations and then will be 0
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 1,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': {50: 1, 100: 1, 150: 1}, #Controls the exploration/exploitation trade-off.

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp/','checkpoint_0.pth.tar'),
    'load_model_file': ('temp','temp.h5'),
    'numItersForTrainExamplesHistory': 10000,
    'bestScore': 0

})

def main():
    try:
        log.info('Loading data...')
        X_data = 2 * np.random.randn(100, 5)
        Y_data = 2.5382 * np.cos(X_data[:, 3]) + X_data[:, 0] ** 2 - 0.5 #2.5382 cos(x_3) + x_0^2 - 0.5
        
        Y_data = Y_data.reshape(-1,1)
        Board.data = np.hstack((X_data,Y_data))
        
        log.info('Loading %s...', Game.__name__)
        g = Game(3, "prefix") #(1.)
        #TODO: Add a feature to make the expression size variable?

        log.info('Loading %s...', nn.__name__)
        nnet = nn(g) #(2.)

        if args.load_model:
            log.info('Loading checkpoint "%s/%s"...', args.load_model_file[0], args.load_model_file[1])
            nnet.load_checkpoint(args.load_model_file[0], args.load_model_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')

        c = Coach(g, nnet, args) #(3.)

        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the learning process 🎉')
        start = time()
        c.learn()
    except KeyboardInterrupt:
        print(f"Time taken = {time()-start}")
        print(f"best expression = {Board.best_expression}")
        print(f"best loss = {Board.best_loss}")
        save = input("Save best expression and loss? (y/n): ")
        if 'y' in save.lower():
            with open("best_multi_dim_expression.txt", "w") as f:
                f.write(f"{Board.best_expression}\n")
                f.write(f"{Board.best_loss}\n")
                print("Saved!")
            print("Saving Model History")
            plt.plot(range(1, 1+len(Coach.scores)), Coach.scores, label = "Training History")
            plt.xlabel("Iteration number")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig("Model_History_Multi_Dimension_KNN_c_1.png", dpi = 5*96)
            plt.close()
            print("Model History Saved")
            plt.plot(c.iteration_numbers, c.unique_expression_counts, label = "Exploration curve")
            plt.xlabel("Iteration number")
            plt.ylabel("Number of unique expressions visited")
            plt.legend()
            plt.savefig("Exploration_Curve_prefix_kNN_c_100_no_P.png", dpi = 5*96)
            plt.close()

if __name__ == "__main__":
    main()
