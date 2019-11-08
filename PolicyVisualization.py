#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 08:35:51 2019

@author: bala
"""
import numpy as np
import pandas as pd

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def ploicyViz(agent):


    X = np.random.uniform(-1.2, 0.6, 10000)
    Y = np.random.uniform(-0.07, 0.07, 10000)
    Z = []
    for i in range(len(X)):
        q = agent.model.predict(np.reshape(np.asarray([X[i],Y[i]]),(-1,2)))[0]
        temp = np.random.choice(
                                np.where(q == np.max(q))[0]
                                )
        z = temp
        Z.append(z)
    Z = pd.Series(Z)
    colors = {0:'blue',1:'lime',2:'red'}
    colors = Z.apply(lambda x:colors[x])
    labels = ['Left','Right','Nothing']
    
    
    
    
    fig = plt.figure(5, figsize=[7,7])
    ax = fig.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X,Y, c=Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    recs = []
    try:
        for i in range(0,3):
             recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    except:
         try:
              for i in range(0,2):
                 recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
         except:
             for i in range(0,1):
                 recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt.legend(recs,labels,loc=4,ncol=3)
#    fig.savefig('Policy - Modified.png')
    plt.show()
    
    
def ploicyViz_Cont(agent, action_cont):


    X = np.random.uniform(-1.2, 0.6, 10000)
    Y = np.random.uniform(-0.07, 0.07, 10000)
    Z = []
    for i in range(len(X)):
        q = agent.model.predict(np.reshape(np.asarray([X[i],Y[i]]),(-1,2)))[0]
        temp = np.random.choice(
                                np.where(q == np.max(q))[0]
                                )
        z = temp
        Z.append(z)
    
    Z_dir = []
    for act in Z:
        if action_cont[act] > 0:
            Z_dir.append(2)
        if action_cont[act] < 0:
            Z_dir.append(1)
        if action_cont[act] == 0:
            Z_dir.append(0)
        
    Z = pd.Series(Z_dir)
    colors = {0:'blue',1:'lime',2:'red'}
    colors = Z.apply(lambda x:colors[x])
    labels = ['Left','Right','Nothing']
    
    
    
    
    fig = plt.figure(5, figsize=[7,7])
    ax = fig.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X,Y, c=Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    recs = []
    try:
        for i in range(0,3):
             recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    except:
         try:
              for i in range(0,2):
                 recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
         except:
             for i in range(0,1):
                 recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))
    plt.legend(recs,labels,loc=4,ncol=3)
#    fig.savefig('Policy - Modified.png')
    plt.show()

