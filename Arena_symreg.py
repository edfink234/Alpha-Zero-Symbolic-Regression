import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                win (1) or loss (-1)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player1]
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board) == -1:
            it += 1
            if verbose:
                assert self.display
                self.display(board)
            action = players[0](board) #Function that returns action with maximum prob given the board

            valids = self.game.getValidMoves(board)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board = self.game.getNextState(board, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board)))
            self.display(board)
        return self.game.getGameEnded(board)

    def playGames(self, num, verbose=False):
        """
        Plays num games

        Returns:
            wins: games won
            losses: games lost
            draws:  games neither won nor lost
        """
        
        score = 0
        for _ in tqdm(range(num), desc="Arena.playGames"):
            gameResult = self.playGame(verbose=verbose)
            score += gameResult

        return score
