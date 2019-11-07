#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:51:36 2019

@author: bala
"""

import matplotlib.pyplot as plt
import gym          # Tested on version gym v. 0.14.0 and python v. 3.17
import time
import numpy as np

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
NUM_EPISODES = 5000
LEN_EPISODE = 200
reward_history = []
loss_history = []
max_dist = []
final_position = []

agent = sNN.SimpleNNagent(env)


# =============================================================================
# 
# max_position = -.4
# positions = np.ndarray([0,2])
# rewards = []
# successful = []
# for episode in range(1000):
#     print(f"episode: {episode}")
#     running_reward = 0
#     env.reset()
#     done = False
#     for i in range(200):
#         state, reward, done, _ = env.step(np.random.randint(0,3))
#         # Give a reward for reaching a new maximum position
#         if state[0] > max_position:
#             max_position = state[0]
#             positions = np.append(positions, [[episode, max_position]], axis=0)
#             running_reward += 10
#         else:
#             running_reward += reward
#         if done: 
#             if state[0] >= 0.5:
#                 successful.append(episode)
#             rewards.append(running_reward)
#             break
# 
# print('Furthest Position: {}'.format(max_position))
# plt.figure(1, figsize=[10,5])
# plt.subplot(211)
# plt.plot(positions[:,0], positions[:,1])
# plt.xlabel('Episode')
# plt.ylabel('Furthest Position')
# plt.subplot(212)
# plt.plot(rewards)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.show()
# print('successful episodes: {}'.format(np.count_nonzero(successful)))
# 
# =============================================================================


# Parameters
steps = 200
state = env.reset()
agent.epsilon= 0.3
agent.discount = 0.99
loss_history = []
reward_history = []
episodes = 1000
max_position = -0.4
agent.learningRate = 0.001
successes = 0
position = []

for episode in trange(episodes):
    episode_loss = 0
    episode_reward = 0
    state = env.reset()

    for s in range(steps):
        if episode % 100 == 0 and episode > 0:
            env.render()
        
        action = agent.getTrainAction(state)
        
        state_1, reward, done, _ = env.step(action)
        
        # Adjust reward based on car position
        reward = state_1[0] + 0.5
        
        # Keep track of max position
        if state_1[0] > max_position:
            max_position = state_1[0]
            writer.add_scalar('data/max_position', max_position, episode)
        
        # Adjust reward for task completion
        if state_1[0] >= 0.5:
            reward += 1
        
        # Find max Q for t+1 state
        Q1 = policy(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
        maxQ1, _ = torch.max(Q1, -1)
        
        # Create target Q value for training the policy
        Q_target = Q.clone()
        Q_target = Variable(Q_target)
        Q_target[action] = reward + torch.mul(maxQ1.detach(), gamma)
        
        # Calculate loss
        loss = loss_fn(Q, Q_target)
        
        # Update policy
        policy.zero_grad()
        loss.backward()
        optimizer.step()

        episode_loss += loss.item()
        episode_reward += reward
        
        if done:
            if state_1[0] >= 0.5:
                # On successful epsisodes, adjust the following parameters

                # Adjust epsilon
                epsilon *= .95
                writer.add_scalar('data/epsilon', epsilon, episode)

                # Adjust learning rate
                scheduler.step()
                #optimizer.param_groups[0]['lr'] = max(optimizer.param_groups[0]['lr'], 1.0e-4)
                writer.add_scalar('data/learning_rate', optimizer.param_groups[0]['lr'], episode)

                # Record successful episode
                successes += 1
                writer.add_scalar('data/cumulative_success', successes, episode)
                writer.add_scalar('data/success', 1, episode)
            
            elif state_1[0] < 0.5:
                writer.add_scalar('data/success', 0, episode)
            
            # Record history
            loss_history.append(episode_loss)
            reward_history.append(episode_reward)
            writer.add_scalar('data/episode_loss', episode_loss, episode)
            writer.add_scalar('data/episode_reward', episode_reward, episode)
            weights = np.sum(np.abs(policy.l2.weight.data.numpy()))+np.sum(np.abs(policy.l1.weight.data.numpy()))
            writer.add_scalar('data/weights', weights, episode)
            writer.add_scalar('data/position', state_1[0], episode)
            position.append(state_1[0])

            break
        else:
            state = state_1
            
writer.close()
print('successful episodes: {:d} - {:.4f}%'.format(successes, successes/episodes*100))
