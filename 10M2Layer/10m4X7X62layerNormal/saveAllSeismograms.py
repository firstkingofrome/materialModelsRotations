#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:05:38 2019
outputs velocity plots for all values in the the essi volume at spacing
@author: eeckert
"""
### loads the hdf5 container and gets motions from that, integrates said motions and plots
#import pandas as pd
import h5py
import sys
import os
from os.path import isfile
import subprocess
import multiprocessing
import pySW4 as sw4
import obspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
# Turn interactive plotting off
plt.ioff()

# from artie rodgers sw4_postprocess_plot_wfs.py
#unlike with arties code, everything is in meters
#also I use argument dictionaries when a function requires a ton of complex arguments since I think this makes things easier!, this is just a skeleton
#populate this as you iterate

def createTrace(data,trace_header):
    header = obspy.core.trace.Stats(header=trace_header)
    trace=obspy.core.trace.Trace(data=data,header=header)
    # add the rest of the sac header
    #trace.meta["sac"]=metaDataTraces[0].meta["sac"]
    return trace

#zcoord=0 correspodns to the surface point
def createStream(streamData,trace_header,zcoord=0):
    st = obspy.core.stream.Stream()
    for i in range(streamData[0,:,:,:].shape[0]):
        st.clear()
        for j in range(streamData[0,:,:,:].shape[1]):
            #get everything in this line
            st.append(createTrace(streamData[:,i,j,zcoord],trace_header))
    return st


def plotLine(plotArgs):
    #debug parameters: debug parameters: sac,name,t1,t2,z1,z2,zlabel,g = x,"5M spacing 1Km cube",0,5,0,1000,"X (meters)",9.8
    st=plotArgs["sac"]
    DPI=plotArgs["DPI"]
    t1,t2 = float(st[0].stats["starttime"]),float(st[0].stats["endtime"])
    z1,z2,lineSpacing = plotArgs["lineStart"],plotArgs["lineStop"],plotArgs["lineSpacing"]
    zlabel,name = plotArgs["zlabel"],plotArgs["name"]
    g = plotArgs["g"]
    st_acc = st.copy()
    st_acc.differentiate()
    st_vel = st.copy()
    st_dis = st.copy()
    st_dis.integrate()
    time = np.linspace(0, st[0].stats.npts*st[0].stats.delta, st[0].stats.npts)
    print('Lengths: ', len(st_acc[0].data), len(st_vel[0].data), len(st_dis[0].data), len(time))
    gm_list = ['Acceleration (g)', 'Velocity (m/s)', 'Displacement (m)']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,8), dpi=DPI)
    for i, ax in enumerate(axes):
        ax.set_xlim(t1, t2)
        ax.set_xlabel('Time (sec)')
        ax.set_ylim(z1, z2)
        ax.set_yticks(range(z1,z2,lineSpacing))
        ax.set_ylabel(zlabel)
        ax.grid()
        ax.set_title(gm_list[i])
        # Acceleration
        if i==0:
            z = z1 + lineSpacing
            scale = plotArgs["scale"]
            for tr in st_acc[1:]:
                ax.plot(time, z + scale*tr.data/g, lw=0.25)
                ax.text(t2-5, z, str(round((np.abs(tr.data/g)).max(),2)), fontsize=6)
                z += lineSpacing
                if z > z2:
                    break
            ax.vlines(2,z1+1, z1+1.5, colors='k', lw=2)
            ax.vlines(4, z1+1, z1+2, colors='k', lw=2)
            #ax.text(1, z1+0.8, '1 2 g', fontsize=8, horizontalalignment='left', verticalalignment='top')
        
        # Velocity
        if i==1:
            z = z1 + lineSpacing
            scale = plotArgs["scale"]
            for tr in st_vel[1:]:
                ax.plot(time, z + scale*tr.data, lw=0.25)
                ax.text(t2-5, z, str(round((np.abs(tr.data)).max(),2)), fontsize=6)
                z += lineSpacing
                if z > z2:
                    break
            ax.vlines(2, z1+1, z1+1.5, colors='k', lw=2)
            ax.vlines(4, z1+1, z1+2, colors='k', lw=2)
            #ax.text(1, z1+0.8, '1 2 m/s', fontsize=8, horizontalalignment='left', verticalalignment='top')
        # Displacement
        if i==2:
            z = z1 + lineSpacing
            scale = plotArgs["dispScale"]
            for tr in st_dis[1:]:
                ax.plot(time, z + scale*tr.data, lw=0.25)
                ax.text(t2-5, z, str(round((np.abs(tr.data)).max(),2)), fontsize=6)
                z += lineSpacing
                if z > z2:
                    break
            ax.vlines(2, z1+1, z1+1.5, colors='k', lw=3)
            #ax.text(1, z1+0.8, '1 m', fontsize=8, horizontalalignment='left', verticalalignment='top')        
    #line_string = st[0].stats['channel']+name
    fig.suptitle('Run: '+name+' component: '+st[0].stats['channel'], fontsize=14)
    plt.autoscale(tight=True)
    fig.subplots_adjust(top=0.90)
    fig.savefig(plotArgs["path"]+name+'.'+'.plot_wf.png',dpi=DPI)
    plt.close(fig) # do not display the plot if running with an ipython compatible interpreter

"""
A very inexpertely done parallel loop wrapper (as if I am looping across the depth option)
"""
def parallelPlotWrapper(args):
    #make a plot for xyz at the current depth
    depth,plotArgs,parameters,x,y,z = args
    #now generate the plot for the x direction at the surfac
    plotArgs["lineStart"],plotArgs["lineStop"] = parameters["LOCATION_INFORMATION"][location]['y'][0],parameters["LOCATION_INFORMATION"][location]['y'][1]
    plotArgs["path"] =  "./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/x/"  
    plotArgs["lineSpacing"]=np.int32(gridSpacing)

    st = createStream(hdf5File['vel_1 ijk layout'],trace_header,zcoord=depth)
    plotArgs["sac"] = st
    plotArgs["name"] = "Lines Along X at " + str(depth*gridSpacing) + " Meters depth"
    plotLine(plotArgs)
    #save y plot
    os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/y",exist_ok=True)
    plotArgs["path"] =  "./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/y/"  
    st = createStream(hdf5File['vel_0 ijk layout'],trace_header)
    plotArgs["sac"] = st
    plotArgs["name"] = "Lines Along Y at surface"
    plotLine(plotArgs)
    #save z plot
    os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/z",exist_ok=True)
    plotArgs["path"] =  "./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/z/"  
    st = createStream(hdf5File['vel_2 ijk layout'],trace_header)
    plotArgs["sac"] = st
    plotArgs["name"] = "Lines Along Z at surface"
    plotLine(plotArgs)
    return

### default plot and sac arguments 
trace_header = {
    "starttime":0,
    "endtime":45,
    "sampling_rate":-1,
    "delta":-1, 
    "npts":-1,
    "calib":1.0
}

#default plot arguments 
plotArgs = {
    "sac":None,
    "name":"10M spacing ",
    "lineStart":500,
    "lineStop":700,
    "lineSpacing":0,
    "zlabel":"Y Meters",
    "g":9.8,
    "path":"./",
    "DPI":400,
    "scale":10.0,
    "dispScale":100.0
}

### Main parameters (aka model specific parameters)
parameters = {
        "DATA_TYPE":"VEL",
        "MAX_CPU":35,
        "PLOT_DPI":400,
        "MODEL_DIR":"10m4X7X6Normal/sw4output",
        "OUTPUTS_FROM_SCRIPT":"processed",
        "LOCATION_INFORMATION":{"1X3X6M.SS_800_1000":{'x':(2000,2200),'y':(800,1000),"depth":200},
        "1X3X6.SS_1500_1700":{'x':(2000,2200),'y':(1500,1700),"depth":200},
        "1X3X6.SS_2000_2200":{'x':(2000,2200),'y':(2000,2200),"depth":200},
        "1X3X6M.SS_3000_3200":{'x':(2000,2200),'y':(3000,3200),"depth":200},
        "1X3X6.SS_4000_4200":{'x':(2000,2200),'y':(4000,4200),"depth":200},
        "1X3X6M.SS_5000_5200":{'x':(2000,2200),'y':(5000,5200),"depth":200},
        "1X3X6M.SS_6000_6200":{'x':(2000,2200),'y':(6000,6200),"depth":200}
        }
        }

rootDir = os.getcwd()
sw4Outputs=[]
essi = []
hdf5File = ''
sw4Outputs = os.listdir("./"+parameters["MODEL_DIR"])
essi = [i for i in sw4Outputs if ('essi') in i]
### if multiprocessing bitches about this it is probably an issue with the paths, try using fully qualified complete paths
essi = ['./'+parameters["MODEL_DIR"]+'/'+i for i in essi]
#make a directory for the output seismograms
os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms",exist_ok=True)

#generate outputs for each container
for location in parameters["LOCATION_INFORMATION"]:
    #load up the relevent container
    currentContainer = []
    currentContainer = [i for i in essi if (location.split('.')[1] == i.split('.')[2])][0]
    hdf5File = h5py.File(currentContainer,'r')   
    #important container values
    gridSpacing = hdf5File['ESSI xyz grid spacing'][0]
    dt = hdf5File['timestep'][:][0]
    recordShape = hdf5File['vel_0 ijk layout'].shape #includes the number of time steps
    #save outputs for this this hdf5 output
    os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location,exist_ok=True)
    #save x plot
    os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/x",exist_ok=True)
    #populate sac header
    trace_header["starttime"],trace_header["endtime"] = 0,dt*recordShape[0]
    trace_header["sampling_rate"] = recordShape[0]/(dt*recordShape[0])
    trace_header["delta"],trace_header["npts"]=dt,recordShape[0] #I asusme that the calibration for these will always be 1.0
    #iterate across all depths
    # all plots oriented in y direction
    """
    You really should make this a parallel program
    """

    for depth in range(recordShape[3]):
        #now generate the plot for the x direction at the surfac
        plotArgs["lineStart"],plotArgs["lineStop"] = parameters["LOCATION_INFORMATION"][location]['y'][0],parameters["LOCATION_INFORMATION"][location]['y'][1]
        plotArgs["path"] =  "./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/x/"  
        plotArgs["lineSpacing"]=np.int32(gridSpacing)
        plotArgs["PLOT_DPI"] = parameters["PLOT_DPI"]
        st = createStream(hdf5File['vel_0 ijk layout'],trace_header,zcoord=depth)
        plotArgs["sac"] = st
        plotArgs["name"] = "Lines Along X at " + str(depth*gridSpacing) + " Meters depth"
        plotLine(plotArgs)
        #save y plot
        os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/y",exist_ok=True)
        plotArgs["path"] =  "./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/y/"  
        st = createStream(hdf5File['vel_1 ijk layout'],trace_header,zcoord=depth)
        plotArgs["sac"] = st
        plotArgs["name"] = "Lines Along Y at " + str(depth*gridSpacing) + " Meters depth"
        plotLine(plotArgs)
        #save z plot
        os.makedirs("./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/z",exist_ok=True)
        plotArgs["path"] =  "./"+parameters["OUTPUTS_FROM_SCRIPT"]+"/seismograms/"+location+"/z/"  
        st = createStream(hdf5File['vel_2 ijk layout'],trace_header,zcoord=depth)
        plotArgs["sac"] = st
        plotArgs["name"] = "Lines Along Z at " + str(depth*gridSpacing) + " Meters depth"
        plotLine(plotArgs)
        #only run one line
        break
        
    
    #done
       
