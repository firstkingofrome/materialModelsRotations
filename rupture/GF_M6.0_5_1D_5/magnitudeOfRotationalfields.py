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
#import pySW4 as sw4
import obspy
#import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import scipy
#import joblib

CPU_COUNT = multiprocessing.cpu_count()
### takes a 3d array of 1d time series, computes 1d fft and returns for each
def parallelFFT(array):
    ### deal with this later
    print("not working right now!")
    return np.fft.rfft(array)

class velocityField():
    def __init__(self,hdf5File,location=None):
        self.x = hdf5File['vel_0 ijk layout'][:]
        self.y = hdf5File['vel_1 ijk layout'][:]
        self.z = hdf5File['vel_2 ijk layout'][:]
        self.t = np.arange(0,hdf5File['timestep'][0]*self.x.shape[0],hdf5File['timestep'][0])  #time series so as to make the plotting easier
        self.dt= hdf5File['timestep'][0]
        self.orgin = hdf5File['ESSI xyz origin'][:]
        self.location = location
        self.gridSpcaing = hdf5File["ESSI xyz grid spacing"][0]
    
    #simple trapezoid numerical integration, upgrade to spectral methods later
    def integrateAll(self):
        #number        
        print("simple numerical method (trapezoid), need to upgrade to spectral routines!")
        self.x[:] = scipy.integrate.cumtrapz(y=self.x[:,:,:,:],dx=self.dt,initial=0,axis=0)
        self.y[:] = scipy.integrate.cumtrapz(y=self.y[:,:,:,:],dx=self.dt,initial=0,axis=0)
        self.z[:] = scipy.integrate.cumtrapz(y=self.z[:,:,:,:],dx=self.dt,initial=0,axis=0)
    
    #simple numerical differentiation (by central difference, upgrade to spectral methods later)
    def differAll(self):
        print("WARN: simple numerical method, need to upgrade to spectral routines!")
        #number
        self.x[:] = np.gradient(self.x[:,:,:,:],self.dt,axis=0)
        self.y[:] = np.gradient(self.y[:,:,:,:],self.dt,axis=0)
        self.z[:] = np.gradient(self.z[:,:,:,:],self.dt,axis=0)
        
#computes rotations using curl
class computeRotations():
    def __init__(self,hdf5File,location=None):
        self.vel = velocityField(hdf5File,location)
        #defualt plot parameters
        self.plotArgs ={
            "data":self.vel.x,
            "t1":0.0,
            "t2":15.0,
            "z1":500,
            "z2":600,
            "dt": self.vel.dt,
            "npts":self.vel.t,
            "grid_spacing":10,
            "direction":'x',
            "zlabel":"1-D rocking rotation in x direction",
            "line_string":"line at 500",
            "name": "rotations in x direction ",
            "scale":1E8,
            "outpath":None
                    } 
    
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
        #now do so using curl definition
        #sigma1 = rotation around yaxis (rotyAxis), gradU3/by X2
        #sigma2 = rotation around xaxis (rotxAxis), grad U3/by X1
        #sigma3 = rotation around zaxis (rotzAxis), (1/2)(grad u2/by x1 - gradu1/x2)
        #self.rotXaxis = np.gradient(self.vel.x[:,:,:,:],self.vel.dt,axis=0)/(self.vel.gridSpcaing)
        #self.rotXaxis = scipy.integrate.cumtrapz(y=self.x[:,:,:,:],dx=self.vel.dt,initial=0,axis=0)/(self.vel.gridSpcaing)
        #loop through all of the time steps, compute the curl at each and every time step and save the reulst
        #to that point in the time series
        for t in range(shape[0]): #loop across all timesteps
            xGrad = np.gradient(self.vel.x[t,:,:,:],rotCurl.vel.gridSpcaing,axis=0) #U1
            yGrad = np.gradient(self.vel.y[t,:,:,:],rotCurl.vel.gridSpcaing,axis=0) #U2
            #zGrad = np.gradient(self.vel.z[t,:,:,:],rotCurl.vel.gridSpcaing,axis=0)
            #assign rotation values
            self.rotXaxis[t,:,:,:] = xGrad
            self.rotYaxis[t,:,:,:] = yGrad
            self.rotZaxis[t,:,:,:] = 0.5*(yGrad-xGrad)
        pass
    
    ### plots the requested rotation based on the current plot parameters
    def plotRotations(self,display=False):
        ### parameters
        args=self.plotArgs
        data = args["data"]
        t1,t2 = args["t1"],args["t2"]
        z1,z2 = args["z1"],args["z2"]
        timeStep = args["dt"]
        lineSpacing = args["grid_spacing"]
        zlabel = args["zlabel"]
        line_string=args["line_string"]
        name=args["name"]
        outpath = args["outpath"]
        scale = args["scale"]
        direction = args["direction"]
        ### end parameters
        time = np.linspace(0, (len(data[:,0,0]))*timeStep, len(data[:,0,0]))
        #print('Lengths: ', len(st_acc[0].data), len(st_vel[0].data), len(st_dis[0].data), len(time))
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25,8), dpi=200)
        axes.set_xlim(t1, t2)
        axes.set_xlabel('Time (sec)')
        axes.set_ylim(z1, z2)
        axes.set_yticks(range(z1,z2,lineSpacing))
        axes.set_ylabel(zlabel)
        axes.grid()
        axes.set_title("rotation in X-direction")
        z = z1 + lineSpacing
        #check plot direction
        if(direction=='x'):
            for line in range(data.shape[1]):
                axes.plot(time, z + scale*data[:,line,0,0], lw=1)
                axes.text(t2-5, z, str(round((np.abs(data[:,line,0,0])).max()*scale,6)) + '{:.2e}'.format(1/scale) +" rads/s", fontsize=10)
                z += lineSpacing
                if z > z2:
                    break
        #otherwise assume y
        else:    
            for line in range(data.shape[1]):
                axes.plot(time, z + scale*data[:,0,line,0], lw=1)
                axes.text(t2-5, z, str(round((np.abs(data[:,0,line,0])).max()*scale,6)) + '{:.2e}'.format(1/scale) +" rads/s", fontsize=10)
                z += lineSpacing
                if z > z2:
                    break
        axes.vlines(2,z1+1, z1+1.5, colors='k', lw=3)
        axes.vlines(4, z1+1, z1+2, colors='k', lw=3)
        axes.text(1, z1+0.8, '  ', fontsize=8, horizontalalignment='left', verticalalignment='top')
    
        line_string = line_string
        #fig.suptitle('Run: '+name+' component: '+st[0].stats['channel']+' '+'Line along '+line_string, fontsize=14)
        plt.autoscale(tight=True)
        fig.subplots_adjust(top=0.90)
        print(outpath+name+'.'+line_string+'.plot_wf.png')
        fig.savefig(outpath+name+'.'+line_string+'.plot_wf.png')
        if(not(display)):plt.close(fig)
        return 

### Creates frames for video the top one shows the velocity field solution while the bottom one shows the predicted rotations
###
        



def create3dVelPlot(velData,rotData,fname,title="none",vmin=0.0,vmax=4.0E-7,gridSpacing=10,DPI=200):    
    #debug, velData,rotData,fname,title,vmin,vmax,gridSpacing,DPI  = rotCurl.vel.x[t,:,:,0],rotCurl.rotXaxis[t,:,:,0],fname,'{0:.2f}'.format(t*rotCurl.vel.dt),0.0,4.0E-7,10,200
    #from mpl_toolkits import mplot3d
    ###get the 3d plot to work first

    x,y = np.linspace(0,velData.shape[0]*gridSpacing,velData.shape[0]),np.linspace(1,velData.shape[1]*gridSpacing,velData.shape[1])
    y,x = np.meshgrid(x,y) #deal with the different sw4 coordinates
    fig, ax = plt.subplots(projection='3d', dpi=DPI)
    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()
    ax.plot_surface(x,y, velData, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    #ax.view_init(0, 45)
    ax.set_title('surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #just save it
    #plt.title(title)
    fig.savefig(fname)
    #plt.close(fig) #close it
    """
    #first thing is first, just try to get a scatter out of this
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
    

    ax.plot_trisurf(x, y, z,
                cmap='viridis', edgecolor='none');
    
    
    ###now show the animation                
    
    
    sample from mpl documentation
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    """

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



#steal most of this by reading 1 trace from the sw4 simulation
metaDataTraces =obspy.read("./"+parameters["MODEL_DIR"]+"/*.x")
trace_header = {
    "starttime":metaDataTraces[0].stats["starttime"],
    "endtime":metaDataTraces[0].stats["endtime"],
    "sampling_rate":metaDataTraces[0].stats["sampling_rate"],
    "delta":metaDataTraces[0].stats["delta"], 
    "npts":metaDataTraces[0].stats["npts"],
    "calib":metaDataTraces[0].stats["calib"]
}



#generate a movie for each container
for location in parameters["LOCATION_INFORMATION"]:
    #load up the relevent container
    currentContainer = []
    #debug currentContainer=['./processed/1X3X6.SS_ESSI_SMALL.cycle=0000.essi'][0]
    currentContainer = [i for i in essi if (location.split('.')[1] == i.split('.')[2])][0]
    hdf5File = h5py.File(currentContainer,'r')
    #instantiate curl rotation objects
    rotAtan=computeRotations(hdf5File,location)
    rotCurl=computeRotations(hdf5File,location)
    rotAtan.vel.integrateAll()
    rotAtan.computeSurfaceRotationsAtan()
    """
    It would probably be a good idea to clean this up a little bit and have it output all of the data to an output directory instead of just a single
    transect
    """
    #compare predicated rotations from atan to curl (make sure they are similar)
    rotAtan.plotArgs['data'] = rotAtan.rotXaxis
    rotAtan.plotArgs['name'] = "Rotations computed with arctan"
    rotAtan.plotArgs['outpath'] = './' + parameters["OUTPUTS_FROM_SCRIPT"]
    rotAtan.plotArgs['scale'] = 1E7
    rotAtan.plotArgs["grid_spacing"] = 10
    rotAtan.plotRotations()
    #now compute rotations using the defintion of curl
    rotCurl.computeRotationsAsCurl()
    #plot
    rotCurl.plotArgs['data'] = rotCurl.rotXaxis
    rotAtan.plotArgs['name'] = "Rotations computed with CURL"
    rotCurl.plotArgs['outpath'] = './' + parameters["OUTPUTS_FROM_SCRIPT"]
    rotCurl.plotArgs['scale'] = 1E7
    rotCurl.plotArgs["grid_spacing"] = 10
    rotCurl.plotRotations()


    
