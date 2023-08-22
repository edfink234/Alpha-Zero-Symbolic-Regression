import logging
import coloredlogs
import numpy as np

from Coach_symreg import Coach
from symreg.SymRegGame import SymRegGame as Game
from symreg.SymRegLogic import Board
from symreg.keras.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 5000,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 5,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp/','checkpoint_0.pth.tar'),
    'load_model_file': ('temp/','temp.h5'),
    'numItersForTrainExamplesHistory': 20,

})

def main():
    try:
        log.info('Loading %s...', Game.__name__)
        g = Game(6)

        log.info('Loading %s...', nn.__name__)
        nnet = nn(g)

        if args.load_model:
            log.info('Loading checkpoint "%s/%s"...', args.load_model_file[0], args.load_model_file[1])
            nnet.load_checkpoint(args.load_model_file[0], args.load_model_file[1])
        else:
            log.warning('Not loading a checkpoint!')

        log.info('Loading the Coach...')
        
        Board.data = np.loadtxt("symreg/test_data.txt")
        c = Coach(g, nnet, args)

        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the learning process 🎉')
        c.learn()
    except KeyboardInterrupt:
        save = input("Save best expression and loss? (y/n): ")
        if 'y' in save.lower():
            with open("best_expression.txt", "w") as f:
                f.write(f"{Board.best_expression}\n")
                f.write(f"{Board.best_loss}\n")
                print("Saved!")
    #TODO: Plot best expression and save


if __name__ == "__main__":
    main()
