from __future__ import print_function
import numpy as np
import pickle

def display(board):
    n = board.shape[0]
    
    for y in range(n):
        print (y,"|",end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("b ",end="")
            elif piece == 1: print("W ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("   -----------------------")
    
class Arena():
    def __init__(self, player1, player2, game):
        # player1 and player2 are functions which take in board, return action
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def playGame(self, verbose=False):
        # execute one game and return winner
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                print("Turn ", str(it), "Player ", str(curPlayer))
                display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)
            # if valids[action] == 0:
            #     for i in range(len(valids)):
            #         if valids[i] >0:
            #             print(i/8,i%8)

            assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            print("Turn ", str(it), "Player ", str(curPlayer))
            display(board)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):


    	num = int(num/2)
    	oneWon = 0
    	twoWon = 0
    	for _ in range(num):
    		if self.playGame(verbose=verbose)==1:
    			oneWon+=1
    		else:
    			twoWon+=1
    	self.player1, self.player2 = self.player2, self.player1
    	for _ in range(num):
    		if self.playGame(verbose=verbose)==-1:
    			oneWon+=1
    		else:
    			twoWon+=1
    	return oneWon, twoWon

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.n), int(i%self.game.n))
        a = input()
        print(a)
        
        x,y = int(a[0]),int(a[1])
        a = self.game.n * x + y if x!= -1 else self.game.n ** 2

        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

if __name__ == "__main__":
    from OthelloGame import OthelloGame
    curGame = OthelloGame(6)
    p1 = RandomPlayer(curGame)
    p2 = GreedyOthelloPlayer(curGame)
    # p2 = HumanOthelloPlayer(curGame)

    p1,p2 = p1, p2

    arena = Arena(p1.play, p2.play, curGame)
    for _ in range(20):
        winner =  arena.playGame(verbose=False)
        print(winner)
        
    
