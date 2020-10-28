"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    contX = 0
    contO = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                contX += 1
            elif board[i][j] == O:
                contO += 1
    
    # The X always starts the game.
    # So, if the number of X and O are equal, it's time for X to play
    if contX == contO:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Possible actions are the empty spots on the board
    possibleActions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possibleActions.add((i, j))
    
    return possibleActions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # The original board should be left unmodified
    newBoard = deepcopy(board)
    # Make the move on the copied board
    newBoard[action[0]][action[1]] = player(board)
    # Returns the copied board
    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for i in range(3):
        if board[i][0] == X and board[i][1] == X and board[i][2] == X:
            return X
        if board[i][0] == O and board[i][1] == O and board[i][2] == O:
            return O
    
    # Check columns
    for j in range(3):
        if board[0][j] == X and board[1][j] == X and board[2][j] == X:
            return X
        if board[0][j] == O and board[1][j] == O and board[2][j] == O:
            return O
    
    # Check one diagonal
    if board[0][0] == X and board[1][1] == X and board[2][2] == X:
        return X
    if board[0][0] == O and board[1][1] == O and board[2][2] == O:
        return O
    
    # Check the other diagonal
    if board[2][0] == X and board[1][1] == X and board[0][2] == X:
        return X
    if board[2][0] == O and board[1][1] == O and board[0][2] == O:
        return O
    
    # If there is no winner
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    isThereWinner = winner(board)
    if isThereWinner is None:
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    return False
    
    # If there is a winner or if there is no empty space, then the game is over
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    isThereWinner = winner(board)
    if isThereWinner is None:
        return 0
    elif isThereWinner is X:
        return 1
    elif isThereWinner is O:
        return -1


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # The best move that the AI can do
    bestMove = None
    playerToGo = player(board)
    # If AI is X, AI should maximize the score
    if playerToGo == X:
        previousValue = -math.inf
        for action in actions(board):
            v = minValue(result(board, action))
            if previousValue < v:
                # Maximize the score
                previousValue = v
                bestMove = action
    # If AI is O, AI should minimize the score
    elif playerToGo == O:
        previousValue = math.inf
        for action in actions(board):
            v = maxValue(result(board, action))
            if previousValue > v:
                # Minimize the score
                previousValue = v
                bestMove = action
    return bestMove


def maxValue(board):
    """
    Returns the highest value of minValue(result(s,a))
    """
    v = -math.inf
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = max(v, minValue(result(board, action)))
    return v


def minValue(board):
    """
    Returns the lowest value of maxValue(result(s,a))
    """
    v = math.inf
    if terminal(board):
        return utility(board)
    for action in actions(board):
        v = min(v, maxValue(result(board, action)))
    return v