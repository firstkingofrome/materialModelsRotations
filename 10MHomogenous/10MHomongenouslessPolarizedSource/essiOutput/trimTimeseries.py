#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:57:50 2019

@author: eeckert
"""
import os
import h5py
print("this program trims the number of timesteps in the hdf5 container (if you want to say exclude the surface waves or something)")

container = os.listdir('.')
container = [i for i in container if('.essi' in i)] #make sure it ends in .essi
startTime=0
stopTime=20
for i in container:
    #load the container
    currentContainer = h5py.File(i,'r+')
    dt = currentContainer['timestep'][0]
    if(startTime>0):startStep = int(startTime/dt)
    else:startStep = 0
    stopStep = int(stopTime/dt)
    currentContainer['cycle start, end'][0],currentContainer['cycle start, end'][1] = startStep,stopStep
    #assign the new velocity data sets for this container
    data = currentContainer['vel_0 ijk layout'][startStep:stopStep][:]
    del currentContainer['vel_0 ijk layout']
    currentContainer['vel_0 ijk layout'] = data
    
    data = currentContainer['vel_1 ijk layout'][startStep:stopStep][:]
    del currentContainer['vel_1 ijk layout']
    currentContainer['vel_1 ijk layout'] = data
    
    data = currentContainer['vel_2 ijk layout'][startStep:stopStep][:]
    del currentContainer['vel_2 ijk layout']
    currentContainer['vel_2 ijk layout'] = data
    
    currentContainer.close()