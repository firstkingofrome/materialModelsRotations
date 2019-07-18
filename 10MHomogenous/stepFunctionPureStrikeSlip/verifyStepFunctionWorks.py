#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:50:05 2019

@author: eeckert
"""

import numpy as np
import matplotlib.pyplot as plt

print("gaussian")
x = np.arange(0,10,0.001)
gaussian = 1*np.exp((-1.0*(x-5.0)**2)/(2*1**2))
plt.plot(x,gaussian)
#clear it

#convolve with rectangle
print("rectangle")
rect = np.zeros(shape=x.shape)
rect[:] = 1.0
rect[0:5] = 0
rect[-5:-1] = 0
plt.plot(x,rect)
plt.close()
#convolve against rectangle
print("gaussian convolved with rectangle")
plt.plot(x,np.convolve(gaussian,rect,mode='same'))
#convolve rectangle on rectangle
rect2 = np.zeros(shape=x.shape)
rect2[:] = 2.0
rect2[0:5] = 0.0
rect2[-5:-1] = 0.0
print("rectange on rectangle")
plt.plot(x,np.convolve(rect,rect2,mode='same'))