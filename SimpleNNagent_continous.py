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


class SimpleNNagent_Continous():
    def __init__(self, env):
        self.trainX = []
        self.trainY = []
        self.replayMemory = []
        self.epsilon = 1.0
        self.minEpsilon = 0.01
        self.epsilonDecay = 0.997
        self.discount = 0.95
        self.learningRate = 0.002
        self.batchSize = 128
        self.sLow = env.observation_space.low
        self.sHigh = env.observation_space.high
        self.nActions = 10
        self.model = self.buildModel(env.observation_space.shape[0],self.nActions)
        self.actionContinous = np.linspace(-1, 1, num=10, endpoint=True)
        
    def toDiscreteAction(self,action):
        return np.argmin(np.absolute(self.actionContinous - action))
    
    def toContAction(self,action):
        return self.actionContinous[action]
        
    def nState(self, state):
#        return np.divide(state-self.sLow,
#                         (self.sHigh-self.sLow))
        return state
        
    def buildModel(self,iSize, oSize):
# =============================================================================
#         model = Sequential()
#         model.add(Dense(128, input_dim=iSize, activation='relu'))
#         model.add(Dense(52, activation='relu'))
#         model.add(Dense(oSize, activation='linear'))
#         model.compile(loss='mse', optimizer='sgd') # Adam()
# =============================================================================
        
# =============================================================================
#         model = Sequential()
#         model.add(Dense(34, input_dim=iSize, activation='relu'))
#         model.add(Dense(31, activation='relu'))
#         model.add(Dense(21, activation='relu'))
#         model.add(Dense(19, activation='relu'))
#         model.add(Dense(10, activation='relu'))
#         model.add(Dense(4, activation='relu'))
#         model.add(Dense(oSize, activation='linear'))
#         model.compile(loss='mse', optimizer='sgd') # Adam()
# =============================================================================
        
# =============================================================================
#         model = Sequential()
#         model.add(Dense(50, input_dim=iSize, activation='relu'))
#         model.add(Dense(oSize, activation='linear'))
#         sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
#         model.compile(loss='mse', optimizer= sgd) # Adam()
# =============================================================================
            
        model = Sequential()
        model.add(Dense(64, input_dim=iSize, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(oSize, activation='linear'))
#        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learningRate))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learningRate))
        return model
        
    def trainModel(self):
        self.model.fit(np.asarray(self.trainX),
                       np.asarray(self.trainY),
#                       epochs=2,
                       verbose = 0)
        
    def miniBatchTrainModel(self):
        self.trainX = []
        self.trainY = []
        loss = 0
        for sarsa in self.replayMemory:
            loss+=self.buildTrainData(sarsa[0], sarsa[1], sarsa[2], sarsa[3], sarsa[4])       
        self.model.fit(np.asarray(self.trainX),
                       np.asarray(self.trainY),
                       epochs=1)
        return loss
        
    def EpsilonGreedyPolicy(self,state):
        if random.random() <= self.epsilon:
            # choose random
            action = random.randint(0,self.nActions-1)
        else:
            #ChooseMax
            #Handle multiple max
            self.qValues = self.model.predict(np.reshape(self.nState(state),(-1,2)))[0]
            action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
        return action
    
    def newGame(self):
        self.trainX = []
        self.trainY = []
#        self.replayMemory = []
    
    def getTrainAction(self,state):
        action = self.EpsilonGreedyPolicy(state)
        return action    
    
    def getAction(self,state):
        self.qValues = self.model.predict(np.reshape(self.nState(state),(-1,2)))[0]
        action = np.random.choice(
                            np.where(self.qValues == np.max(self.qValues))[0]
                            )
        return action
    
    def buildReplayMemory(self, currState, nextState, reward, done, action):
#        if len(self.replayMemory)> self.batchSize:
#            self.replayMemory.pop()
        self.replayMemory.append([currState, nextState, reward, done, action])
    
    def buildMiniBatchTrainData(self):
        c = []
        n = []
        r = []
        d = []
        a = []
        if len(self.replayMemory)>self.batchSize:
            minibatch = random.sample(self.replayMemory, self.batchSize)
        else:
            minibatch = self.replayMemory
        for ndx,[currState, nextState, reward, done, action] in enumerate(minibatch):
#        for ndx,val in enumerate(choices):
#            [currState, nextState, reward, done, action] = self.replayMemory[val]
            c.append(currState)
            n.append(nextState)
            r.append(reward)
            d.append(done)
            a.append([ndx, action])
        c = np.asanyarray(c)
        n = np.asanyarray(n)
        r = np.asanyarray(r)
        d = np.asanyarray(d)
        a = np.asanyarray(a)
        a = a.T
        qVal_n = self.model.predict(np.reshape(self.nState(n),(-1,2)))
        qMax_n = np.max(qVal_n, axis  = 1)
        qVal_c = self.model.predict(np.reshape(self.nState(c),(-1,2)))
        Y = copy.deepcopy(qVal_c)
        y = np.zeros(r.shape)
        ndx = np.where(d == True)
        y[ndx] = r[ndx]
        ndx = np.where(d == False)
        y[ndx] = r[ndx] + self.discount * qMax_n[ndx]
        Y[a[0],a[1]] = y
        self.trainX = c
        self.trainY = Y
        return skMSE(Y,qVal_c)
        
    def buildTrainData(self, currState, nextState, reward, done, action):
        states = np.asarray([currState, nextState])
        q = self.model.predict(np.reshape(self.nState(states),(-1,2)))
        self.qValues = q[0]
        qVal = q[1]
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
    
    def getReward(self, currState, nextState, action, reward, maxDist, step, done):
        
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
            
# =============================================================================
#         # Reward 9
#         sign = np.array([-1.0,0.0,1.0])
#         if currState[1]*sign[action] >= 0:
#             reward = nextState[0] + 1 * (0.99**step)
#         else:
#             reward = nextState[0] - -1 * (1.01**step)
# =============================================================================
        
# =============================================================================
#         # Reward 10
#         sign = np.array([-1.0,0.0,1.0])
#         if currState[1]*sign[action] >= 0:
#             reward = 1
#         else:
#             reward = -1
#         reward = (0.8**step) * reward
#         if nextState[0] >=0.5:
#             reward+= 100
# =============================================================================
        
# =============================================================================
#         # Reward 11
#         if nextState[1] > currState[1] and nextState[1]>0 and currState[1]>0:
#             reward += 15
#         elif nextState[1] < currState[1] and nextState[1]<=0 and currState[1]<=0:
#             reward +=15
#         if done:
#             reward = reward + 1000
#         else:
#             reward=reward-10
# =============================================================================
        
# =============================================================================
#         # Reward 12
#         reward = nextState[0]
#         if nextState[0] >= 0.5:
#             reward += 5000
#         elif nextState[0] > maxDist:
#             reward += 5
# =============================================================================
        
# =============================================================================
#         # Reward 13
#         sign = np.array([-1.0,0.0,1.0])
#         if currState[1]*sign[action] >= 0:
#             reward = 1
#         else:
#             reward = -1
#         if currState[0]>=0.5:
#             reward += 1000
#         reward = (0.999**step) * reward
# =============================================================================
        
# =============================================================================
#         # Reward 14
#         reward = currState[0]+0.5
#         if nextState[0]>-0.5:
#             reward+=1
# =============================================================================
        
        # Reward 15
        if nextState[1] > currState[1] and nextState[1]>0 and currState[1]>0:
            reward += 15
        elif nextState[1] < currState[1] and nextState[1]<=0 and currState[1]<=0:
            reward +=15
        if step >=199:
            reward = reward + 1000
        else:
            reward=reward-10
        if nextState[0]>= 0.5:
            reward += 1000
#        reward = (0.8**step)*reward
            
        return reward
        
    