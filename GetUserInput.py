#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 23:31:48 2019

@author: bala
"""

from time import sleep

def getUsernput(waitTime):
    print(f'Please provide input in {waitTime} seconds! (Hit Ctrl-C to start)')
    try:
        discreteInterval = int(waitTime/0.05)
        for i in range(0,discreteInterval):
            sleep(0.05) # could use a backward counter to be preeety :)
        print('No input is given.')
    except KeyboardInterrupt:
        x = input('Input x:')
        print(f'Given Input : {x}')
        return x