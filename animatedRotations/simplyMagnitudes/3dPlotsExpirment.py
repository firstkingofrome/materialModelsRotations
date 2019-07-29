#import pandas as pd
import h5py
#import sys
import os
import subprocess
import multiprocessing
import numpy as np
import scipy
import scipy.integrate
from mayavi.mlab import * ### figure out what this is doing
from mayavi import mlab
mlab.options.offscreen = True #dont display anything
gridSpacing = 10
NCPUS = 16
normalize=True
#load up youre hdf5 container and get the velocity data set, try plotting that!
#load up the relevent container
path = "./processed/frames/1X3X6.SS_ESSI_LARGE.cycle=0000.essi/"
#currentContainer = './sw4output/1X3X6.SS_ESSI_LARGE.cycle=0000.essi'
currentContainer = './sw4output/1X3X6.SS_ESSI_SMALL.cycle=0000.essi'
FPS=10
rootDir = os.getcwd()
hdf5File = h5py.File(currentContainer,'r')
#get the sw4 x surface
surface = hdf5File['vel_2 ijk layout'][:,:,:,0]
#get the sw4 y surface

#get the sw4 z surface

"""
generate a meshgrid so that this is correctly spaced
"""
x,y = np.linspace(0,surface.shape[1]*gridSpacing,surface.shape[1]),np.linspace(1,surface.shape[2]*gridSpacing,surface.shape[1])
x,y = np.meshgrid(x,y) #deal with the different sw4 coordinates
x,y = x.T,y.T


### interesting stuff happens at t=900 seconds
vmin,vmax = np.min(surface[:]),np.max(surface[:])   
for t in range(surface.shape[0]):
    
    ### expirment by computing a single vector field
    mlab.imshow(surface[t],vmin=vmin,vmax=vmax)

    #mlab.surf(surface[t],warp_scale='auto',vmin=vmin,vmax=vmax)
    #mlab.surf(x,y,surface[t])
    mlab.savefig(filename=path+'{0:04d}'.format(t)+".png")
    #clear it
    mlab.clf()
    

### assemble all of these frames with ffmpeg
#call ffmpeg
os.chdir(path)
videoName = "X-plane"+".mp4"
#try and remove the file (incase one already exists)
if(os.path.isfile(videoName)):os.remove(videoName)
print("/data/software/sl-7.x86_64/module/tools/ffmpeg/bin/ffmpeg -framerate "+str(FPS)+" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "+videoName)
#subprocess.call(("/data/software/sl-7.x86_64/modules/tools/ffmpeg/bin/ffmpeg -framerate "+str(parameters["FPS"])+" -pattern_type glob -i '*.png' -c:v h264 -pix_fmt yuv420p "+videoName), shell=True)
subprocess.call(("/data/home/eeckert/ffmpeg-4.1.4-amd64-static/ffmpeg -framerate "+str(FPS)+" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "+videoName), shell=True)
#go back to the normal working directory
subprocess.call(("cp "+videoName + " " +rootDir), shell=True)

#copy it to the root directory
os.chdir(rootDir)
    


    
    
    
    
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