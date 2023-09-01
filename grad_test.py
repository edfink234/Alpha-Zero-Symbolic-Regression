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

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 10,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 0,        
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 30,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 1,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp/','checkpoint_0.pth.tar'),
    'load_model_file': ('temp','temp.h5'),
    'numItersForTrainExamplesHistory': 20,
    'bestScore': 0

})

def main():
    try:
        log.info('Loading data...')
                
        X_data = (np.linspace(1,10,1000)**2).reshape((1000,1))
        Y_data = np.gradient(X_data.flatten()).reshape((len(X_data),1))
        Board.data = np.hstack((X_data,Y_data))

        log.info('Loading %s...', Game.__name__)
        g = Game(10) #(1.)

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
        c.learn()
    except KeyboardInterrupt:
        print(f"best expression = {Board.best_expression}")
        print(f"best loss = {Board.best_loss}")
        save = input("Save best expression and loss? (y/n): ")
        if 'y' in save.lower():
            with open("best_expression.txt", "w") as f:
                f.write(f"{Board.best_expression}\n")
                f.write(f"{Board.best_loss}\n")
                x_vals = X_data
                x = sp.symbols('x0')
                best_func = sp.lambdify(x, Board.best_expression)
                y_vals = best_func(x_vals.flatten())
                
                plt.plot(x_vals, y_vals, label = "$"+sp.latex(Board.best_expression)+"$")
                plt.plot(Board.data[:,0], Board.data[:,1], label = "Original data")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.title("Best Expression Gradient Test")
                plt.savefig("Grad_Test_Best_Expression.png", dpi = 5*96)
                print("Saved!")

if __name__ == "__main__":
    main()
