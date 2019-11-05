#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:45:29 2019

@author: bala
"""

import random
import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
from keras import optimizers
import copy
from sklearn.metrics import mean_squared_error as skMSE


class SimpleNNagent():
    def __init__(self, env):
        self.trainX = []
        self.trainY = []
        self.epsilon = 0.3
        self.discount = 0.99
        self.sLow = env.observation_space.low
        self.sHigh = env.observation_space.high
        self.nActions = env.action_space.n
        self.buildModel(env.observation_space.shape[0], env.action_space.n)
        
    def nState(self, state):
        return np.divide(state-self.sLow,
                         (self.sHigh-self.sLow))
        
    def buildModel(self,iSize, oSize):
#        self.model = Sequential()
#        self.model.add(Dense(128, input_dim=iSize, activation='relu'))
#        self.model.add(Dense(52, activation='relu'))
#        self.model.add(Dense(oSize, activation='linear'))
#        self.model.compile(loss='mse', optimizer='sgd') # Adam()
        
#        self.model = Sequential()
#        self.model.add(Dense(34, input_dim=iSize, activation='relu'))
#        self.model.add(Dense(31, activation='relu'))
#        self.model.add(Dense(21, activation='relu'))
#        self.model.add(Dense(19, activation='relu'))
#        self.model.add(Dense(10, activation='relu'))
#        self.model.add(Dense(4, activation='relu'))
#        self.model.add(Dense(oSize, activation='linear'))
#        self.model.compile(loss='mse', optimizer='sgd') # Adam()
        
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=iSize, activation='relu'))
        self.model.add(Dense(oSize, activation='linear'))
        sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer= sgd) # Adam()
        
    def trainModel(self):
        self.model.fit(np.asarray(self.trainX),
                       np.asarray(self.trainY),
                       epochs=2)
        
    def EpsilonGreedyPolicy(self):
        if random.random() < self.epsilon:
            # choose random
            action = random.randint(0,self.nActions-1)
        else:
            #ChooseMax
            #Handle multiple max
            action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
            
        return action
    
    def newGame(self):
        self.trainX = []
        self.trainY = []
    
    def getTrainAction(self,state):
        self.qValues = self.model.predict(np.reshape(self.nState(state),(-1,2)))[0]
        action = self.EpsilonGreedyPolicy()
        return action    
    
    def buildTrainData(self, currState, nextState, reward, done, action):
        qVal = self.model.predict(np.reshape(self.nState(nextState),(-1,2)))[0]
        qMax = np.max(qVal)
        Y = copy.deepcopy(self.qValues)
        if done:
            y = reward
        else:
            y = reward + self.discount * qMax
        #check if replaced prpoerly, 1 epoh loss should be mpr, initial loss has to be more
        #check if values are referenced rather rhan copy
        
        Y[action] = y
        self.trainX.append(self.nState(currState))
        self.trainY.append(Y)
        return skMSE(Y,self.qValues)
    
    def getReward(self, currState, nextState, action, reward, episodeMaxDist, step):
        
# =============================================================================
#         # Reward 1
#         if nextState[0] >= 0.5 or  nextState[0] > episodeMaxDist:
#             reward += 5 
#         else:
#             reward = nextState[0] + 0.5
# =============================================================================
            
# =============================================================================
#         # Reward 2
#         if nextState[0] >= 0.5:
#             reward += 5
#         else:
#             reward = nextState[0] + 0.5
# =============================================================================
        
# =============================================================================
#         # Reward 3
#         # No change
# =============================================================================
        
# =============================================================================
#         # Reward 4
#         sign = np.array([-1.0,0.0,1.0])
#         if nextState[1]*sign[action] >= 0:
#             reward = nextState[0] + 0.5
#         else:
#             reward = nextState[0] - 0.5
# =============================================================================
        
# =============================================================================
#         # Reward 5
#         sign = np.array([-1.0,0.0,1.0])
#         if currState[1]*sign[action] >= 0:
#             reward = nextState[0] + 0.5
#         else:
#             reward = nextState[0] - 0.5
# =============================================================================
        
# =============================================================================
#         # Reward 6
#         sign = np.array([-1.0,0.0,1.0])
#         if currState[1]*sign[action] >= 0:
#             reward = 1
#         else:
#             reward = -1
# =============================================================================
        
# =============================================================================
#         # Reward 7
#         sign = np.array([-1.0,0.0,1.0])
#         if currState[1]*sign[action] >= 0:
#             reward = 1
#         else:
#             reward = -1
#         reward = (0.999**step) * reward
# =============================================================================
        
# =============================================================================
#         # Reward 8
#         sign = np.array([-1.0,0.0,1.0])
#         if currState[1]*sign[action] >= 0:
#             reward = 1 * (0.99**step) 
#         else:
#             reward = -1 * (1.01**step) 
# =============================================================================
            
        # Reward 9
        sign = np.array([-1.0,0.0,1.0])
        if currState[1]*sign[action] >= 0:
            reward = nextState[0] + 1 * (0.99**step)
        else:
            reward = nextState[0] - -1 * (1.01**step)
        
        return reward
        
    