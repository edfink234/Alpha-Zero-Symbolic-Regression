import logging
import math
from copy import copy
import numpy as np
from symreg.SymRegLogic import Board
from visualize_tree import *

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.iteration_number = 0
        self.Qsa = {}  # stores Q values, i.e., estimations of expected cumulative rewards, for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited (s = state, a = action)
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
    
    def get_max_reward(self, s, a):
        state_action_pair = s + bytes([a])
        max_r_val = 0
        for s in self.Es:
            if state_action_pair in s and 0 <= self.Es[s] <= 1 and self.Es[s] > max_r_val:
                max_r_val = self.Es[s]
        return max_r_val

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard!

        Returns:
            probs: a policy vector where the probability of the ith possible action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(copy(canonicalBoard)) #self.search mutates what you pass in, so we use a copy
        s = self.game.stringRepresentation(canonicalBoard) #state
        counts = [self.Nsa.get((s,a), 0) for a in range(self.game.getActionSize())]
        if temp == 0:
            if not any(counts): #if all counts are 0, then we need to prevent an invalid state!
                counts = self.game.getValidMoves(canonicalBoard)
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if not counts_sum: #if all counts are 0, then we need to prevent an invalid state!
            counts = self.game.getValidMoves(canonicalBoard)
            counts_sum = float(sum(counts))
        probs = [x / counts_sum if counts_sum else 1/self.game.getActionSize() for x in counts]
        
        if not np.isclose(sum(probs),1):
            print("Sum probs =",sum(probs))
        
        
        return probs
    
    def cpuct(self):
        for i in self.args.cpuct:
            if i <= self.iteration_number:
                return self.args.cpuct[i]
        return list(self.args.cpuct.values())[-1]
#        return math.exp(Board.best_loss)

    def search(self, canonicalBoard: list):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return value is 0 <= v <= 1

        """
        s = self.game.stringRepresentation(canonicalBoard)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard)
        if self.Es[s] != -1:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves -> This is where certain moves are made impossible! (valids is an array of 1's and 0's, so even if the NN predicts a move to be perfectly valid 
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
#                    u = self.Qsa[(s, a)] + self.cpuct() * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)]) #AlphaGoZero formula, see page 26 of https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf
                    
                    u = self.Qsa[(s, a)] + self.cpuct() * self.Ps[s][a] * ( (self.Ns[s])**(0.25) / math.sqrt(self.Nsa[(s, a)])) #see page 4 of https://arxiv.org/pdf/1902.05213.pdf
                else:
                    u = self.cpuct() * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act
        try:
            assert((y := ((x:= getPNdepth([self.game.b._Board__tokens_dict[i] for i in canonicalBoard]))[0] <= self.game.n)) and (True if y else not x[1]))
        except AssertionError:
            print("AssertionError :",x)
        next_s = self.game.getNextState(canonicalBoard, a)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
