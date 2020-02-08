import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import array
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
import pandas as pd
import h5py

class TicTac():
    def __init__(self, alpha, gamma, epsilon, mode, size):
        self.size = size
        self.state = np.full((self.size*self.size), 0)
        self.player1 = QPlayer('X', self.state, self.size, True)
        self.mode = mode

        if self.mode == 'Easy':
            self.player2 = QPlayer('O', self.state, self.size, False)
        elif self.mode == 'Hard':
            self.player2 = QPlayer('O', self.state, self.size, True)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.winner = None
        self.Xwin = 0
        self.Owin = 0
        self.Draw = 0
        self.results = pd.DataFrame()
        self.number_games = 0
        self.learn = True
        self.tossWin=None
        self.current_player = None
        self.turn = None
        #self.current_player, self.turn = self.toss()

    def restart(self):
        self.number_games = self.number_games + 1
        # if(self.number_games == 1000):
        #      self.epsilon = 0
        self.state = np.full((self.size*self.size), 0)
        self.winner = None
        self.current_player, self.turn = self.toss()
        self.current_player.toss += 1
        self.tossWin = self.turn
        self.player1.prev_state = self.state
        self.player2.prev_state = self.state

    def toss(self):
        if self.tossWin is None:
            return self.player1, 'X'
        elif self.tossWin == 'X':
            return self.player2, 'O'
        elif self.tossWin == 'O':
            return self.player1, 'X'


    def checkWinner(self):

        #check columns
        columnFlag = True

        for i in range(self.size):
            columnFlag = True
            if(self.state[i] != 0):
                flag = self.state[i]
                for j in range(i, i+(self.size * self.size), self.size):
                    if(self.state[j] != flag):
                        columnFlag = False
                        break
                if(columnFlag==True):
                    return flag

        # check rows
        rowFlag = True

        for i in range(0, self.size * self.size, self.size):
            rowFlag = True
            if (self.state[i] != 0):
                flag = self.state[i]
                for j in range(i, i+self.size):
                    if (self.state[j] != flag):
                        rowFlag = False
                        break
                if (rowFlag == True):
                    return flag

        #check Diagnal
        diagnalFlag = True
        if(self.state[0] != 0):
            flag = self.state[0]
            for i in range(0, self.size*self.size, self.size+1):
                if(self.state[i] != flag):
                    diagnalFlag = False
            if(diagnalFlag == True):
                return flag

        diagnalFlag = True
        if (self.state[self.size-1] != 0):
            flag = self.state[self.size-1]
            for i in range(self.size-1, (self.size * self.size)-1, self.size-1):
                if (self.state[i] != flag):
                    diagnalFlag = False
            if (diagnalFlag == True):
                return flag


        #check for draw
        count = 0
        for i in range(len(self.state)):
            #print(self.state[i])
            if (self.state[i] == 0):
                return None
            else:
                count += 1
        if count == len(self.state):
            winner='Draw'
            return winner

    def updateGameResults(self, win):
        if (win == 1):
            self.winner = 'X'
            self.Xwin = self.Xwin + 1
            self.results = self.results.append({'epoch': self.number_games, 'result': self.winner, 'toss': self.tossWin}, ignore_index=True)
            #print(self.winner)
            # for j in range(0, self.size * self.size, self.size):
            #     print(self.state[j:j + self.size])
            #     print('\n')
        elif (win == -1):
            self.winner = 'O'
            self.Owin = self.Owin + 1
            self.results = self.results.append({'epoch': self.number_games, 'result': self.winner, 'toss': self.tossWin}, ignore_index=True)
        elif (win == 'Draw'):
            self.winner = 'Draw'
            self.Draw = self.Draw + 1
            self.results = self.results.append({'epoch': self.number_games, 'result': self.winner, 'toss': self.tossWin}, ignore_index=True)

    def plotResults(self):
        epochs = self.number_games
        bins = np.arange(1, epochs / 25) * 25
        self.results['game_bins'] = np.digitize(self.results['epoch'], bins, right=True)
        #print(self.results)
        counts = self.results.groupby(['game_bins', 'result']).epoch.count().unstack()
        #toss = self.results.groupby(['game_bins', 'toss']).epoch.count().unstack()
        counts.fillna(0, inplace=True)
        #toss.fillna(0, inplace=True)
        ax = counts.plot(kind='bar', stacked=True)
        ax.set_xlabel("Count of Games in Bins of 25s")
        ax.set_ylabel("Counts of Draws / Player O win / Player X win")
        if self.mode=='Easy':
            title2= 'Player X is intelligent and Player O is random'
        else:
            title2 = 'Player X is intelligent and Player O is intelligent'
        ax.set_title('Distribution of Results Vs Count of Games ('+ str(self.size) + 'X' + str(self.size) +') Played - '+ title2)
        plt.show()



class QPlayer():
    def __init__(self, symbol, state, size, intelligent):
        self.symbol = symbol
        self.prev_state = state
        self.intelligent = intelligent
        self.toss = 0
        self.model = createModel(size)
        self.target_model=createModel(size)
        if(self.symbol == 'X'):
            self.tag = 1
        else:
            self.tag = -1

    def make_move(self, game, next_state):
        self.prev_state = game.state
        if game.winner is None:
            game.state = next_state
            win = game.checkWinner()
            game.updateGameResults(win)

        #change turn
        if game.turn == 'X':
            game.turn = 'O'
            game.current_player = game.player2
        else:
            game.turn = 'X'
            game.current_player = game.player1

#create deep learning model
def createModel(size):
    model=Sequential()
    model.add(Dense(size*size*3, input_dim=size*size, activation='relu'))
    model.add(Dense(size*size*2, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
    model.summary()
    return model

#rewards
def getReward(game):
    if game.winner == game.current_player.symbol:
        reward = 10
    elif game.winner is None:
        reward = 0
    elif game.winner == 'Draw':
        reward = 5
    else:
        reward = -10
    return reward

#predict score for moves using DQN
def predictScores(movesDict, model, size):
    scoreDict={}
    for move in movesDict:
        score = model.predict(movesDict[move].reshape(1, size*size))
        scoreDict[move] = score
    return scoreDict

#find legal move for opponent
def opponetMoves(game, move):
    if (game.current_player.tag == 1):
        tag = -1
    else:
        tag = 1
    stateDict = {}
    for i in range(move.shape[0]):
        if move[i] == 0:
            new_state = move.copy()
            new_state[i] = tag
            stateDict[(i)] = new_state
    return stateDict

# make move such that it maximizes the minimum score of next state
def optimal_intelligent_move(game, movesDict):
    best_moves = []
    min_score = -float('inf')

    for move in movesDict:
        #find all legal moves for opponent
        opponent_moves = opponetMoves(game, movesDict[move])
        if(len(opponent_moves) > 0):
            #predict score of all opponent moves
            opponent_scores = predictScores(opponent_moves, game.current_player.model, game.size)
            #find minimum score move
            temp_min_opponent = min(opponent_scores.items(), key=lambda x: x[1])[1]
            #find move with maximum minimum score
            if temp_min_opponent > min_score:
                min_score = temp_min_opponent
                best_moves = [move]
            elif temp_min_opponent == min_score:
                best_moves.append(move)
        else:
            next_state = optimal_move(game, movesDict)
            return next_state
    return movesDict[random.choice(best_moves)]

#choose move with highest score from all legal moves
def optimal_move(game, movesDict):
    next_state_scores = predictScores(movesDict, game.current_player.model, game.size)
    max_move = max(next_state_scores.items(), key=lambda x: x[1])[0]
    return movesDict[max_move]

#choose random move from all leagl moves
def random_move(game):
    if game.winner is None:
        movesDict=legalMoves(game)
        next_state = random.choice(list(movesDict.items()))[1]
    else:
        next_state = None
    game.current_player.make_move(game, next_state)

#find leagl moves for a given state
def legalMoves(game):
    stateDict = {}
    for i in range(game.state.shape[0]):
        #for j in range(state.shape[1]):
        if game.state[i] == 0:
            new_state = game.state.copy()
            new_state[i] = game.current_player.tag
            stateDict[(i)] = new_state
    return stateDict


def learn_move(game):
    if game.winner is None:
        movesDict=legalMoves(game)
        #make random move
        if (random.random() < game.epsilon):
            next_state = random.choice(list(movesDict.items()))[1]
        else:
            next_state = optimal_intelligent_move(game, movesDict)
    else:
        next_state = None
    game.current_player.make_move(game, next_state)


def deepLearn(game):
    #if player has played before
    if(game.current_player.tag in game.state):
        #find the previous q-value
        prev_qValue = game.current_player.model.predict(game.current_player.prev_state.reshape(1, game.size*game.size))

        #if game has not ended yet
        if game.winner is None:
            #find best action using DQN
            best_action_next = optimal_move(game, legalMoves(game))
            #find target q value using target model
            next_qValue = game.current_player.target_model.predict(best_action_next.reshape(1, game.size*game.size))
        else:
            next_qValue = 0

        # reward is received if game ended after last move of any player
        reward = getReward(game)
        temp = game.gamma * next_qValue + reward
        target = np.array(prev_qValue + game.alpha * (temp - prev_qValue))
        X_train = game.current_player.prev_state.reshape(1, game.size*game.size)
        #learn weights
        game.current_player.model.fit(X_train, target, epochs=10, verbose=0)


#choose what steps to be taken for intelligent and non-intelligent players
def play_game(game):

    if game.current_player.intelligent:
        deepLearn(game)
        learn_move(game)
    else:
        random_move(game)




def play(game, episodes):
    for epoch in range(episodes):
        game.restart()
        if(epoch % game.size == 0):
            #copy weights to target model
            game.player1.target_model.set_weights(game.player1.model.get_weights())
            game.player2.target_model.set_weights(game.player2.model.get_weights())
        #continue till game is not finished
        while game.winner is None:

            #first player turn
            play_game(game)
            #if game has ended
            if game.winner is not None:
                break

            #second player turn
            play_game(game)

        play_game(game)
        play_game(game)
        play_game(game)
        play_game(game)
    game.plotResults()
    #s = 'model_values' + game.player1.symbol + '.h5'
    #game.player1.model.save(s)



def validateInput(gameMode):
    if gameMode=='Easy' or gameMode=='Hard':
        # input variables
        alpha = 0.1
        gamma = 0.5
        epsilon = 0.1
        iterations = 50
        game = TicTac(alpha, gamma, epsilon, gameMode, gameSize)
        play(game, iterations)
    else:
    	print('Enter Game Mode - Easy or Hard')
        


#get input from user
x, y=input("Please enter game size and game mode(Easy or Hard): ").split()
gameSize = int(x)
gameMode = str(y)
validateInput(gameMode)