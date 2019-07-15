"""
Routines for loading up an essi output from sw4 and creating animated rotations
:author:
    Eric E. Eckert
:copyright:
    Eric E. Eckert eeckert A t nevada d_0_t uNr d-o-T edu
:license:
    BSD 2-Clause License
    
"""
import pandas as pd
import h5py
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
    
    #integrates all records in the volume with respect to time
    #taper indicates the number of time steps that I need to taper at the beginning and the end
    def integrateAll(self):
        print("integrate with obspy")
    
    #diff. all records in the volume
    #taper indicates the number of time steps that I need to taper at the beginning and the end
    def differAll(self):
        #compute derivetive using the obspy methods
        #fix up the time series (since you ditched the first point)
        #self.t = self.t[:-1]
        """
        self.x[:] = np.gradient(self.x[:,:,:,:],axis=0)
        self.y[:] = np.gradient(self.y[:,:,:,:],axis=0)
        self.z[:] = np.gradient(self.z[:,:,:,:],axis=0)
        """
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
        shape = (self.vel.x.shape[0],self.vel.x.shape[1],self.vel.x.shape[2],self.vel.x.shape[3])
        self.rotXaxis = np.zeros(shape,dtype=np.float32) 
        self.rotYaxis = np.zeros(shape,dtype=np.float32) 
        self.rotZaxis = np.zeros(shape,dtype=np.float32) 
        #now do so using curl definition
        #sigma1 = rotation around xaxis (rotxAxis)
        #sigma2 = rotation around yaxis (rotyAxis)
        #sigma3 = rotation around zaxis (rotzAxis)
        #self.rotXaxis = 
        
        pass
    
    ### plots the requested rotation based on the current plot parameters
    def plotRotations(self):
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
    
        for line in range(data.shape[1]):
            axes.plot(time, z + scale*data[:,0,line], lw=1)
            axes.text(t2-5, z, str(round((np.abs(data[:,0,line])).max()*scale,6)) + '{:.2e}'.format(1/scale) +" rads/s", fontsize=10)
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
        fig.savefig(outpath+name+'.'+line_string+'.plot_wf.png')

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
        "LOCATION_INFORMATION":{"1X3X6.SS_ESSI_100M.cycle=0000.essi":{'x':(500,700),'y':(1500,1700),"depth":200},
        "1X3X6.SS_ESSI_SMALL.cycle=0000.essi":{'x':(500,700),'y':(1500,1700),"depth":200}}
        }

### Main
sw4Outputs=[]
essi = []
hdf5File = ''
sw4Outputs = os.listdir("./"+parameters["MODEL_DIR"])
essi = [i for i in sw4Outputs if ('essi') in i]
### if multiprocessing bitches about this it is probably an issue with the paths, try using fully qualified complete paths
essi = ['./'+parameters["MODEL_DIR"]+'/'+i for i in essi]
"""
just do this from within the compute rotations class
#make a directory for sac images, essi images and for sw4images
os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/frames",exist_ok=True)
"""


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
    break
    rotAtan.vel.integrateAll()
    rotAtan.computeSurfaceRotationsAtan()
    #compare predicated rotations from atan to curl (make sure they are similar)
    rotAtan.plotArgs['data'] = rotAtan.rotXaxis
    rotAtan.plotArgs['outpath'] = './' +  parameters["MODEL_DIR"] + '/' + parameters["OUTPUTS_FROM_SCRIPT"]
    rotAtan.plotArgs['scale'] = 1E7
    rotAtan.plotArgs["grid_spacing"] = 10
    rotAtan.plotRotations()
    #now compute rotations using the defintion of curl
 


"""
#test differentiate function
test = createTrace(rotCurl.vel.x[:,0,0,0])
plt.plot(rotCurl.vel.t,rotCurl.vel.x[:,0,0,0])
rotCurl.vel.differAll(55)
plt.plot(rotCurl.vel.t,rotCurl.vel.x[:,0,0,0])


you just needed to taper more timesteps!
###test
ntaper=150
self = rotAtan.vel
plt.plot(self.t,self.x[:,0,0,0])
### take the derrivatetive 
data = np.copy(self.x)
data = data[:,0,0,0]
data = np.fft.fft(data)
data = data*2j#*w
#taper the data using a  hamming taper
hammingTaper= np.arange(0,ntaper,1,dtype=np.float32)[::-1]
hammingTaper= 0.50 + 0.50 * np.cos(np.pi * (hammingTaper - 1) / (ntaper - 1))  # Hamming taper
data[0:ntaper] = data[0:ntaper]*hammingTaper
data[-ntaper-1:-1] = data[-ntaper-1:-1]*hammingTaper #[::-1], you dont need this youre data is already going backwards..
#inverse fft
data = np.fft.ifft(data)     
#
plt.plot(self.t,data)
"""