#import pandas as pd
import h5py
#import sys
import os
import subprocess
import multiprocessing
import numpy as np
import scipy
import scipy.integrate
from mayavi import mlab
mlab.options.offscreen = True #dont display anything
gridSpacing = 10
#load up youre hdf5 container and get the velocity data set, try plotting that!
#load up the relevent container
path = "./processed/frames/1X3X6.SS_ESSI_SMALL.cycle=0000.essi/"
currentContainer = './sw4output/1X3X6.SS_ESSI_SMALL.cycle=0000.essi'
hdf5File = h5py.File(currentContainer,'r')
surface = hdf5File['vel_0 ijk layout'][:,:,:,0]
"""
generate a meshgrid so that this is correctly spaced
"""
x,y = np.linspace(0,surface.shape[1]*gridSpacing,surface.shape[1]),np.linspace(1,surface.shape[2]*gridSpacing,surface.shape[1])
x,y = np.meshgrid(x,y) #deal with the different sw4 coordinates
x,y = x.T,y.T


### interesting stuff happens at t=900 seconds
vmin,vmax = np.min(surface[:]),np.max(surface[:])   
for t in range(surface.shape[0]):

    mlab.surf(surface[t],warp_scale='auto',vmin=vmin,vmax=vmax)
    #mlab.surf(x,y,surface[t],warp_scale='auto')
    mlab.savefig(filename=path+'{0:04d}'.format(t)+".png")
    


    
    
    
    
"""
from numpy import linspace, meshgrid, array, sin, cos, pi, abs
from scipy.special import sph_harm 
from mayavi import mlab

theta_1d = linspace(0,   pi,  91) 
phi_1d   = linspace(0, 2*pi, 181)

theta_2d, phi_2d = meshgrid(theta_1d, phi_1d)
xyz_2d = array([sin(theta_2d) * sin(phi_2d),
                sin(theta_2d) * cos(phi_2d),
                cos(theta_2d)]) 
l=3
m=0

Y_lm = sph_harm(m,l, phi_2d, theta_2d)
r = abs(Y_lm.real)*xyz_2d
    
mlab.figure(size=(700,830))
mlab.mesh(r[0], r[1], r[2], scalars=Y_lm.real, colormap="cool")
mlab.view(azimuth=0, elevation=75, distance=2.4, roll=-50)
mlab.savefig("Y_%i_%i.jpg" % (l,m))
mlab.show()
"""