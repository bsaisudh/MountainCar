#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:51:36 2019

@author: bala
"""

import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17

import SimpleNNagent as sNN
import PolicyVisualization as pv

env = gym.make('MountainCar-v0')
env.seed(42);

# Print some info about the environment
print("State space (gym calls it observation space)")
print(env.observation_space)
print("\nAction space")
print(env.action_space)

# Parameters
NUM_STEPS = 200
NUM_EPISODES = 10000
LEN_EPISODE = 200
reward_history = []
loss_history = []
max_dist = []
final_position = []

agent = sNN.SimpleNNagent(env)

# Run for NUM_EPISODES
for episode in range(NUM_EPISODES):
    agent.newGame()
    episode_reward = 0
    episode_loss = 0
    episode_maxDist = -0.4
    curr_state = env.reset()
    
    for step in range(LEN_EPISODE):
        # Comment to stop rendering the environment
        # If you don't render, you can speed things up
        if episode % 50 == 0:
            env.render()
        
        # Randomly sample an action from the action space
        # Should really be your exploration/exploitation policy
        action = agent.getTrainAction(curr_state)

        # Step forward and receive next state and reward
        # done flag is set when the episode ends: either goal is reached or
        #       200 steps are done
        next_state, reward, done, _ = env.step(action)
        
        if next_state[0] >= 0.5 or  next_state[0] > episode_maxDist:
            reward += 1 
        else:
            reward = next_state[0] + 0.5
        
        
        
        loss = agent.buildTrainData(curr_state, next_state, reward, done, action)

        # This is where your NN/GP code should go
        # Create target vector
        # Train the network/GP
        agent.trainModel()
        agent.newGame()

        # Record history
        episode_reward += reward
        episode_loss += loss
        if next_state[0] > episode_maxDist:
            episode_maxDist = next_state[0]

        # Current state for next step
        curr_state = next_state
        
        if done:
            if curr_state[0] >=0.5:
                agent.epsilon *= 0.95
            # Record history
            reward_history.append(episode_reward)
            loss_history.append(episode_loss)
            max_dist.append(episode_maxDist)
            final_position.append(curr_state[0])
            # You may want to plot periodically instead of after every episode
            # Otherwise, things will slow
            if episode % 25 == 0:
                fig = plt.figure(1)
                plt.clf()
                plt.xlim([0,NUM_EPISODES])
                plt.plot(reward_history,'ro')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Reward Per Episode')
                plt.pause(0.01)
                fig.canvas.draw()
                
                fig = plt.figure(2)
                plt.clf()
                plt.xlim([0,NUM_EPISODES])
                plt.plot(loss_history,'bo')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Reward Per Episode')
                plt.pause(0.01)
                fig.canvas.draw()
            break
    break
    
    pv.ploicyViz(agent)
            
