#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:59:45 2019

@author: eeckert
"""


import sys
import os
import subprocess
import multiprocessing
import pySW4 as sw4
import obspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.signal

"""
def spectralIntegrate(data):
    data = np.fft.fft(data)
    data = data*2j#*w
    #taper the data using a  hamming taper
    hammingTaper= np.arange(0,ntaper,1,dtype=np.float32)[::-1]
    hammingTaper= 0.50 + 0.50 * np.cos(np.pi * (hammingTaper - 1) / (ntaper - 1))  # Hamming taper
    data[0:ntaper] = data[0:ntaper]*hammingTaper
    data[-ntaper-1:-1] = data[-ntaper-1:-1]*hammingTaper[::-1]
    #inverse fft
    data = np.fft.ifft(data)
    return data
"""
def spectralIntegrand(data,dt,t,ntaper=20):
    integrand = np.array([-1,1,-1], dtype=np.float32)
    return scipy.signal.fftconvolve(data,integrand,'same')


#https://docs.scipy.org/doc/scipy-1.1.0/reference/signal.html
def spectralDerriv(data,dt,t,ntaper=20):
    #detrend
    data=scipy.signal.detrend(data)
    derriv = np.array([1,-1,1], dtype=np.float32)
    #window with hamming, then deriv.
    #window = scipy.signal.hanning(ntaper)
    #data = scipy.signal.convolve(data,window,'same')
    data = scipy.signal.fftconvolve(data,derriv,'same')
    return data

def numpyIntegrand(data,delta=0.007048872079363485):
    return scipy.integrate.cumtrapz(data, dx=delta,initial=0)
    #return np.concatenate([np.array([0], dtype=ret.dtype), ret])



def numpyCentralDifference(data,delta=0.007048872079363485):
    #detrend data
    #data=scipy.signal.detrend(data)
    #difference
    data = np.gradient(data,delta)
    return data



"""
#ntaper is the number of time steps in the taper
def spectralDerriv(data,dt,t,ntaper=20):
    data = data[1:]
    data = scipy.fftpack.fft(data)
    df=1/(dt*(data.shape[0]-1))
    #this is so that I skip over the 0 frequency
    w=np.arange(1,((data.shape[0]/2)),1,dtype=np.float32)
    #the fouier transform starts at 0 goes all of the way to the highest postive frequency and then goes from the lowset negetive frequency
    w = df*(2*np.pi)*((w+1))
    w = np.append(w,w)
    #data = data[::-1]
    ### mirror cc
    #data = data.conj()
    #data = np.roll(data,int(data.shape[0]/2))
    #halfWay = int(data.shape[0]/2)
    ###flip it around 
    #taper the data using a  hamming taper
    #hamming = scipy.signal.hamming(data.shape[0]-2,False)
    data[1:] = data[1:]*2j*w #do not run over the 0 frequency term

    hamming = [0.50 + 0.50 * np.cos(np.pi * (i-1) / (ntaper - 1)) for i in range(1,ntaper+1,1)]   # Hamming taper
    hamming = hamming[::-1]

    #foreward
    
    data[1:ntaper+1] = data[1:ntaper+1]*hamming
    #backward
    data = data[::-1]
    data[1:ntaper+1] = data[1:ntaper+1]*hamming[::-1]
    data = data[::-1] #revert to foreward facing directions

    #some how the 0 frequency is getting messed up
    #data[0:int(data.shape[0]/2)] = data[0:int(data.shape[0]/2)].conj()
    #data[-1:-ntaper]=0
    data = scipy.fftpack.ifft(data)
    return data
#ntaper is the number of time steps in the taper
"""

def getPowerSpectra(data,timeStep,freq_range=(0,15)):
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, timeStep)
    idx = np.argsort(freqs)
    #only take the indexes within my postive frequency range
    idx = [i for i in idx if(freqs[i]>freq_range[0] and freqs[i]<freq_range[1])]
    plt.figure(figsize=(5,5))
    plt.plot(freqs[idx], ps[idx])  
    pass

dt = 0.001
t= np.arange(0,20,dt)
t[0:100] = 0 #give it some 0 padding
ft = np.cos(t)
plt.plot(t,ft)
plt.plot(t,spectralDerriv(ft,dt,t))
plt.plot(t,-1*np.sin(t))

### integrand


#this is apparently implemented in scipy!
#this does not work on non periodic data like you have
#plt.plot(t,scipy.fftpack.diff(x=ft,order=1,period=25*dt))
print("now check with a pure odd function")

"""
oddt = np.sin(t)
plt.plot(t,oddt)
plt.plot(t[:-1],spectralDerriv(oddt))
"""



#scipy.fftpack.diff(ft,25*dt)
### try with an actual seismic record since that seems to be the hangup
tr  = obspy.read("500_500.x")
tr = tr[0] #play with first trace
dt = tr.stats["delta"]
npts = tr.stats["npts"]
t = np.arange(0,npts*dt,dt)
plt.plot(t,tr.data)
#plt.plot(t,spectralDerriv(tr.data,dt,t))
plt.plot(t,numpyCentralDifference(tr.data))
tr.differentiate()
plt.plot(t,tr.data)
### now try differentiating with the spectral method

"""
### now try integrating with the spectral method
"""

"""
#Nice plot code
samplingFrequency=10.0
nfft = data.shape[0]
eta=data
etaHann=np.hanning(nfft)*eta
EtaSpectrum=abs(np.fft.fft(eta))
EtaSpectrum=EtaSpectrum*2/nfft # convert to amplitude
EtaSpectrumHann=abs(np.fft.fft(etaHann))
EtaSpectrumHann=EtaSpectrumHann*2*2/nfft # also correct for Hann filter
frequencies=np.linspace(0,samplingFrequency,nfft)
plt.plot(frequencies[0:200],EtaSpectrumHann[0:200],'b.-',label='Hann filtered')
plt.plot(frequencies[0:200],EtaSpectrum[0:200],'c.-',label='rectangular')
plt.xlim(-0.01,2)
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.grid()
plt.legend(loc=0)
plt.show()
"""



