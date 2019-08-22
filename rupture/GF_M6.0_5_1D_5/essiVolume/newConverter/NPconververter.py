#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:52:01 2019
updated the conversion tool to use numpy and dask primitives (since these are much much faster than the primitives used by Houjun in the orgional version)
First just write it to ues numpy, then (only if necessary) rewrite to use dask based parallelization
@author: eeckert
"""

import sys
import h5py
import numpy as np
import datetime
import math
import json
### new 
import numpy as np
import dask
import scipy
from scipy import integrate

ESSI_INFO={
        "essi_x_start":15,
        "essi_x_end":125,
        "essi_y_start":15,
        "essi_y_end":125,
        "essi_z_start":15,
        "essi_z_end":120
        }


#time interval to use in seconds (if -1, all time steps are used)
timeInterval = (-1,-1)

drm_filename = "cubicNewConverter.hdf5"
sw4essiout_hdf5_output = "1X3X6M.GF_18000_18200.essi"
#open up the sw4 hdf5 output

#open up the drm file
drm_file = h5py.File(drm_filename)
drm_x = drm_file['Coordinates'][0::3]
drm_y = drm_file['Coordinates'][1::3]
drm_z = drm_file['Coordinates'][2::3]
#create arrays to hold these results
"""
drm_x = np.zeros(drmXCoordinates.shape[0],dtype=np.float32)
drm_y = np.zeros(drmXCoordinates.shape[0],dtype=np.float32)
drm_z = np.zeros(drmXCoordinates.shape[0],dtype=np.float32)
"""

#open up the hdf5 file output by sw4
sw4_output = h5py.File(sw4essiout_hdf5_output)
# Get parameter values from HDF5 data
h  = sw4_output['ESSI xyz grid spacing'][0]
x0 = sw4_output['ESSI xyz origin'][0]
y0 = sw4_output['ESSI xyz origin'][1]
z0 = sw4_output['ESSI xyz origin'][2]
t0 = sw4_output['time start'][0]
npts = sw4_output['cycle start, end'][1]
dt = sw4_output['timestep'][0]
t1 = dt*(npts-1)
nt = sw4_output['vel_0 ijk layout'].shape[0]
time = np.linspace(t0, t1, npts+1)
#load the sw4 data (into memory!)
vel_x,vel_y = sw4_output['vel_0 ijk layout'][:],sw4_output['vel_1 ijk layout'][:]
vel_z = -1*sw4_output['vel_2 ijk layout'][:]
#create output data sets (as necessary)
if not "Time" in drm_file.keys():
    Time_dset          = drm_file.create_dataset("Time", data=time)
    Accelerations_dset = drm_file.create_dataset("Accelerations", (drm_x.shape[0]*3, nt))
    Displacements_dset = drm_file.create_dataset("Displacements", (drm_x.shape[0]*3, nt))
else:
    Time_dset          = drm_file['Time']
    Accelerations_dset = drm_file['Accelerations']
    Displacements_dset = drm_file['Displacements']
#compute the acceleration and displacement for each drm coordinate using numpy
acc_x = np.gradient(vel_x[:,:,:,:],dt,axis=0)
acc_y = np.gradient(vel_y[:,:,:,:],dt,axis=0)
acc_z = np.gradient(vel_z[:,:,:,:],dt,axis=0)
#compute the displacement for each drm coordinate using numpy
disp_x=scipy.integrate.cumtrapz(y=vel_x[:,:,:,:],dx=dt,initial=0,axis=0)
disp_y=scipy.integrate.cumtrapz(y=vel_y[:,:,:,:],dx=dt,initial=0,axis=0)
disp_z=scipy.integrate.cumtrapz(y=vel_z[:,:,:,:],dx=dt,initial=0,axis=0)
#save this data for the appropriate coordinates
#compute coordinate transforms given the input essi coordinates, in sw4 coordinates!!!
essi_y_start,essi_x_start = ESSI_INFO["essi_y_start"],ESSI_INFO["essi_x_start"]
essi_z_end = ESSI_INFO["essi_z_end"]
loc_x,loc_y = [int((drm_y[i]-essi_y_start)/h) for i in range(drm_y.shape[0])],[int((drm_x[i] - essi_x_start)/h) for i in range(drm_x.shape[0])]
loc_z = [int((essi_z_end - drm_z[i])/h) for i in range(drm_z.shape[0])]
loc_x,loc_y,loc_z = np.array(loc_x),np.array(loc_y),np.array(loc_z)
#assign to the hdf5 container, note that this expects every THIRD item
accelerations = np.hstack((acc_x[:,loc_x,loc_y,loc_z],acc_y[:,loc_x,loc_y,loc_z],acc_z[:,loc_x,loc_y,loc_z]))
displacements = np.hstack((disp_x[:,loc_x,loc_y,loc_z],disp_y[:,loc_x,loc_y,loc_z],disp_z[:,loc_x,loc_y,loc_z]))
accelerations=accelerations.T
displacements=displacements.T
#assignment
Accelerations_dset[...] = accelerations
Displacements_dset[...] = displacements

"""
try:
    drm_file['Accelerations'] = accelerations
    drm_file['Displacements'] = displacements
except RuntimeError:
    #assume that it couldnt overwrite these data sets (because they allready exist)
"""

#done close the container
drm_file.close()
sw4_output.close()

