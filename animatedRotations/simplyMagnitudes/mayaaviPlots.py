"""
Routines for loading up an essi output from sw4 and creating animated rotations
:author:
    Eric E. Eckert
:copyright:
    Eric E. Eckert eeckert A t nevada d_0_t uNr d-o-T edu
:license:
    BSD 2-Clause License
    
"""
#import pandas as pd
import h5py
#import sys
import os
import subprocess
import multiprocessing

import numpy as np
import scipy
import scipy.integrate
#import joblib

CPU_COUNT = multiprocessing.cpu_count()
### takes a 3d array of 1d time series, computes 1d fft and returns for each


class velocityField():
    def __init__(self,hdf5File,location=None):
        self.x = hdf5File['vel_0 ijk layout'][:]
        self.y = hdf5File['vel_1 ijk layout'][:]
        self.z = hdf5File['vel_2 ijk layout'][:]
        self.dx,self.dy,self.dz = np.zeros(shape=self.x.shape),np.zeros(shape=self.y.shape),np.zeros(shape=self.z.shape)
        self.ix,self.iy,self.iz = np.zeros(shape=self.x.shape),np.zeros(shape=self.y.shape),np.zeros(shape=self.z.shape)
        self.t = np.arange(0,hdf5File['timestep'][0]*self.x.shape[0],hdf5File['timestep'][0])  #time series so as to make the plotting easier
        self.dt= hdf5File['timestep'][0]
        self.orgin = hdf5File['ESSI xyz origin'][:]
        self.location = location
        self.gridSpcaing = hdf5File["ESSI xyz grid spacing"][0]
    
    #simple trapezoid numerical integration, upgrade to spectral methods later
    def integrateAll(self):
        #number        
        print("simple numerical method (trapezoid), need to upgrade to spectral routines!")
        self.ix[:] = scipy.integrate.cumtrapz(y=self.x[:,:,:,:],dx=self.dt,initial=0,axis=0)
        self.iy[:] = scipy.integrate.cumtrapz(y=self.y[:,:,:,:],dx=self.dt,initial=0,axis=0)
        self.iz[:] = scipy.integrate.cumtrapz(y=self.z[:,:,:,:],dx=self.dt,initial=0,axis=0)
    
    #simple numerical differentiation (by central difference, upgrade to spectral methods later)
    def differAll(self):
        print("WARN: simple numerical method, need to upgrade to spectral routines!")
        #number
        self.dx[:] = np.gradient(self.x[:,:,:,:],self.dt,axis=0)
        self.dy[:] = np.gradient(self.y[:,:,:,:],self.dt,axis=0)
        self.dz[:] = np.gradient(self.z[:,:,:,:],self.dt,axis=0)
        
#computes rotations using curl
class computeRotations():
    def __init__(self,hdf5File,location=None):
        self.vel = velocityField(hdf5File,location)
        #defualt plot parameters
    
    #@staticmethod
    def rotationAtan(self,record1,record2):
        return np.arctan((record1-record2)/self.vel.gridSpcaing)

    def computeSurfaceRotationsAtan(self):
        #same shape except for the time sereis which is one smaller
        print("only computes for surface layer, also make sure that you integrate youre vel field first!")
        shape = (self.vel.x.shape[0]-1,self.vel.x.shape[1],self.vel.x.shape[2],self.vel.x.shape[3])
        self.rotXaxis = np.zeros(shape,dtype=np.float32) 
        self.rotYaxis = np.zeros(shape,dtype=np.float32) 
        self.rotZaxis = np.zeros(shape,dtype=np.float32) 
        ### assume that this is a symeteric volume
        for i in range(self.vel.x[0,:,:,0].shape[0]-1):
            for j in range(self.vel.y[0,:,:,0].shape[1]-1):
                for t in range(self.vel.z[:,0,0,0].shape[0]-1):
                    self.rotXaxis[t,i,j] = self.rotationAtan(self.vel.x[t,i+1,j,0],self.vel.x[t,i,j,0])
                    self.rotYaxis[t,i,j] = self.rotationAtan(self.vel.y[t,i+1,j,0],self.vel.y[t,i,j,0])
                    self.rotZaxis[t,i,j] = self.rotationAtan(self.vel.z[t,i+1,j,0],self.vel.z[t,i,j,0])
        pass
    
    def computeRotationsAsCurl(self):
        #integrate velocity field against time to generate a displacement field
        self.vel.integrateAll()
        print("Computing rotations using defintion of CURL, Assumes velocity field as input")
        shape = (self.vel.x.shape[0],self.vel.x.shape[1],self.vel.x.shape[2],self.vel.x.shape[3])
        self.rotXaxis,xGrad = np.zeros(shape,dtype=np.float32),np.zeros((self.vel.x.shape[1],self.vel.x.shape[2],self.vel.x.shape[3]),dtype=np.float32)
        self.rotYaxis,yGrad = np.zeros(shape,dtype=np.float32),np.zeros((self.vel.x.shape[1],self.vel.x.shape[2],self.vel.x.shape[3]),dtype=np.float32)
        self.rotZaxis,zGrad = np.zeros(shape,dtype=np.float32),np.zeros((self.vel.x.shape[1],self.vel.x.shape[2],self.vel.x.shape[3]),dtype=np.float32)
        #to that point in the time series
        for t in range(shape[0]): #loop across all timesteps
            xGrad = np.gradient(self.vel.ix[t,:,:,:],rotCurl.vel.gridSpcaing,axis=0) #U1
            yGrad = np.gradient(self.vel.iy[t,:,:,:],rotCurl.vel.gridSpcaing,axis=0) #U2
            #zGrad = np.gradient(self.vel.z[t,:,:,:],rotCurl.vel.gridSpcaing,axis=0)
            #assign rotation values
            self.rotXaxis[t,:,:,:] = xGrad
            self.rotYaxis[t,:,:,:] = yGrad
            self.rotZaxis[t,:,:,:] = 0.5*(yGrad-xGrad)
        pass
    
### Creates frames for video the top one shows the velocity field solution while the bottom one shows the predicted rotations
###
def create3dVelPlot(velData,rotData,fname,title="none",vmin=0.0,vmax=4.0E-7,gridSpacing=10,DPI=200):    
    #debug, velData,rotData,fname,title,vmin,vmax,gridSpacing,DPI  = rotCurl.vel.x[t,:,:,0],rotCurl.rotXaxis[t,:,:,0],fname,'{0:.2f}'.format(t*rotCurl.vel.dt),0.0,4.0E-7,10,200
    #from mpl_toolkits import mplot3d
    ###get the 3d plot to work first

    x,y = np.linspace(0,velData.shape[0]*gridSpacing,velData.shape[0]),np.linspace(1,velData.shape[1]*gridSpacing,velData.shape[1])
    y,x = np.meshgrid(x,y) #deal with the different sw4 coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()

    #ax.view_init(0, 45)
    ax.set_title('surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(x,y, velData, rstride=1, cstride=1,
            cmap='viridis', edgecolor='none')
    #just save it
    #plt.title(title)
    fig.savefig(fname+title+".png")
    plt.close(fig) #close it


#use the parallel one instead its way faster
def createFrame(data,fname,title="none",vmin=0.0,vmax=4.0E-7,DPI=150):
    display = False
    DPI=DPI
    #data = np.copy(rotCurl.rotXaxis[1000,:,:,0])    
    #i am only doing this because it is the easiest way to get the data to display with the correct axis in pyplot
    data = np.flipud(data)
    #data = np.fliplr(data)
    fig, ax = plt.subplots()
    im = ax.imshow(data,cmap="jet",vmin=vmin,vmax=vmax)
    ax.invert_yaxis()
    plt.colorbar(im,ax=ax)
    plt.title(title)
    fig.savefig(fname,dpi=DPI)
    if(not(display)):plt.close(fig)
    return

#a parallel version of the above designed to work with a multiprocessing pool
def parallelCreateFrames(currentFrameArgs):
    ### you have to load mpl or it is not correctly pickeld from the main process
    #import matplotlib as mpl
    import matplotlib.pyplot as plt
    data,fname,title,vmin,vmax = currentFrameArgs
    DPI = 300
    #data = np.copy(rotCurl.rotXaxis[1000,:,:,0])    
    #i am only doing this because it is the easiest way to get the data to display with the correct axis in pyplot
    #data = np.flipud(data)
    #data = np.fliplr(data)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(data,cmap="jet",vmin=vmin,vmax=vmax)
    ax.invert_yaxis()
    plt.colorbar(im,ax=ax)
    plt.title(title)
    plt.savefig(fname,dpi=DPI,bbox_inches = "tight",format="png")
    plt.close()
    

def createTrace(data):
    header = obspy.core.trace.Stats(header=trace_header)
    trace=obspy.core.trace.Trace(data=data,header=header)
    # add the rest of the sac header
    #trace.meta["sac"]=metaDataTraces[0].meta["sac"]
    return trace



###compute power sepctra conveniance function
def plotPowerSpectra(data,timeStep,freq_range=(0,15)):
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, timeStep)
    idx = np.argsort(freqs)
    #only take the indexes within my postive frequency range
    idx = [i for i in idx if(freqs[i]>freq_range[0] and freqs[i]<freq_range[1])]
    plt.figure(figsize=(5,5))
    plt.plot(freqs[idx], ps[idx])  

### Main parameters
parameters = {
        "DATA_TYPE":"VEL",
        "MAX_CPU":35,
        "MODEL_DIR":"sw4output",
        "OUTPUTS_FROM_SCRIPT":"processed",
        "FPS":50,
        "LOCATION_INFORMATION":{"1X3X6.SS_ESSI_SMALL.cycle=0000.essi":{'x':(500,700),'y':(1500,1700),"depth":200},
        "1X3X6.SS_ESSI_100M.cycle=0000.essi":{'x':(500,700),'y':(1500,1700),"depth":200}}
        }
        

### Main
rootDir = os.getcwd()
sw4Outputs=[]
essi = []
hdf5File = ''
sw4Outputs = os.listdir("./"+parameters["MODEL_DIR"])
essi = [i for i in sw4Outputs if ('essi') in i]
### if multiprocessing bitches about this it is probably an issue with the paths, try using fully qualified complete paths
essi = ['./'+parameters["MODEL_DIR"]+'/'+i for i in essi]

#make a directory for the frames
os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames",exist_ok=True)

#generate a movie for each container
for location in parameters["LOCATION_INFORMATION"]:
    #load up the relevent container
    currentContainer = []
    #debug currentContainer=['./processed/1X3X6.SS_ESSI_SMALL.cycle=0000.essi'][0]
    currentContainer = [i for i in essi if (location.split('.')[1] == i.split('.')[2])][0]
    hdf5File = h5py.File(currentContainer,'r')
    #instantiate curl rotation objects
    rotCurl=computeRotations(hdf5File,location)
    #now compute rotations using the defintion of curl
    rotCurl.computeRotationsAsCurl()
    rmin,rmax = np.min(rotCurl.rotXaxis[:]),np.max(rotCurl.rotXaxis[:])    
    #make output directory
    os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames/"+str(location),exist_ok=True)
    #make sure this works in a non parallel way
    for t in range(rotCurl.rotXaxis.shape[0]):
        #make a frame for each time step
        #dont hard code this and make it parallel
        fname = "./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames/"+str(location) + "/"
        #zero pad these filenames to make it easier for ffmpeg
        #createFrame(rotCurl.rotXaxis[t,:,:,0],fname+'{0:04d}'.format(t)+".png",title='{0:.2f}'.format(t*rotCurl.vel.dt),vmin=rmin,vmax=rmax)
        break
        create3dVelPlot(rotCurl.vel.x[t,:,:,0],rotCurl.rotXaxis[t,:,:,0],fname,title='{0:04d}'.format(t),vmin=rmin,vmax=rmax,gridSpacing=10,DPI=200)
        #if(t==1500):break #stop at the 1200 timestep for debug
        
    break
    #call ffmpeg
    os.chdir("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames/"+str(location)+"/")
    videoName = location + "X-plane"+".mp4"
    #try and remove the file (incase one already exists)
    if(os.path.isfile(videoName)):os.remove(videoName)
    print("/data/software/sl-7.x86_64/module/tools/ffmpeg/bin/ffmpeg -framerate "+str(parameters["FPS"])+" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "+videoName)
    #subprocess.call(("/data/software/sl-7.x86_64/modules/tools/ffmpeg/bin/ffmpeg -framerate "+str(parameters["FPS"])+" -pattern_type glob -i '*.png' -c:v h264 -pix_fmt yuv420p "+videoName), shell=True)
    subprocess.call(("/data/home/eeckert/ffmpeg-4.1.4-amd64-static/ffmpeg -framerate "+str(parameters["FPS"])+" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "+videoName), shell=True)
    #go back to the normal working directory
    os.chdir(rootDir)
    break

    
    #create frames for animation-- X rotations
    #make sure that output directory exists to save all of this data too
    os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames/"+str(location),exist_ok=True)
    #create a single named frame for each timestep in the sequence
    outputFrameFnames = ['X' + str(i) + ".png" for i in range(rotCurl.rotXaxis.shape[0])]
    #append the rest of the reletive path to these files so that the parallel processing module works correctly
    outputFrameFnames = ["./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames/"+str(location) + "/" + i for i in outputFrameFnames]
    #create frames, make a copy of the rotation data so I dont muck with it too much!
    data = [rotCurl.rotXaxis[i,:,:,0] for i in range(rotCurl.rotXaxis.shape[0])]
    rmin,rmax = np.min(rotCurl.rotXaxis[:]),np.max(rotCurl.rotXaxis[:])    
    rmin = [rmin for i in range(rotCurl.rotXaxis.shape[0])] #this remains the same for every parallel function call
    rmax = [rmax for i in range(rotCurl.rotXaxis.shape[0])]
    title = [str(i) for i in range(rotCurl.rotXaxis.shape[0])]
    ### create a parallel processing pool and excecute the frame createion in parallel
    pool = multiprocessing.Pool(processes=parameters["MAX_CPU"])
    poolArgs = zip(data,outputFrameFnames,title,rmin,rmax)
    #data,fname,title="none",vmin=0.0,vmax=4.0E-7
    #excecute
    #IF MEMORY CONSUMPTION BECOMES A PROBLEM REGENINEER TO USE IMAP ALA: https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    pool.map(parallelCreateFrames,poolArgs);
    #pool.close() #semaphore wait
    #invoke ffmpeg to render the result    
    # using ffmpeg, convert list of png to mp4
    os.chdir("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames/"+str(location)+"/")
    videoName = location + "X-plane"+".mp4"
    #try and remove the file (incase one already exists)
    if(os.path.isfile(videoName)):os.remove(videoName)
    print("/data/software/sl-7.x86_64/module/tools/ffmpeg/bin/ffmpeg -framerate "+str(parameters["FPS"])+" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "+videoName)
    #subprocess.call(("/data/software/sl-7.x86_64/modules/tools/ffmpeg/bin/ffmpeg -framerate "+str(parameters["FPS"])+" -pattern_type glob -i '*.png' -c:v h264 -pix_fmt yuv420p "+videoName), shell=True)
    subprocess.call(("/data/software/sl-7.x86_64/modules/tools/ffmpeg/bin/ffmpeg -framerate "+str(parameters["FPS"])+" -pattern_type glob -i '*.png' -pix_fmt yuv420p "+videoName), shell=True)
    #go back to the normal working directory
    os.chdir(rootDir)
    """
    os.chdir(flat_mpath+"snap/png/") # change directory to png
    if os.path.isfile(flat_mpath+"snap/png/"+outfile)==True: #does the video already exist?
        print("if worked!")
        os.remove(outfile) #delete it if it does before making the movie 
    print("ffmpeg -framerate "+str(fps)+" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "+outfile)
    subprocess.call(("ffmpeg -framerate "+str(fps)+" -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p "+outfile), shell=True)
    """
    break
