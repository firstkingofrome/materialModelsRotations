#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:48:57 2019
@author: eeckert
"""
### not flipped

import h5py
import numpy as np
import re
import pandas

sw4_i_start = 8000
sw4_i_end   = 8200
sw4_j_start = 18000
sw4_j_end   = 18200
sw4_k_start = 0
sw4_k_end   = 200
sw4_grid_spacing=5
ESSI_SURFACE = 200





sw4essiout_filename =  "Location5.cycle=00000.essi"
pointsRequested = "outputLocations.csv"
pointsOutput = "outputPoints"
#load the csv file
points = pandas.read_csv(pointsRequested)


#open the hdf5 container for this run and get each out put velocity point
sw4essiout = h5py.File(sw4essiout_filename, 'r')
spacing = sw4essiout['ESSI xyz grid spacing'][:][0]
for index, row in points.iterrows():
    print(row)
    #translate into xyz coordinates
    ### THE ESSI CORDINATES ARE XYZ WHILE SW4 IS YXZ
    i,j,k = row["x (m)"],row["y (m)"],row["z (m)"]
    print("preparing refrence point " + row["point"])
    print("saving x y and z velocities for  exssi point " + "%i,%i,%i" % (i,j,k))
    j,i,k = int((i)/spacing), int((j)/spacing),int((ESSI_SURFACE-k)/spacing)
    print("This corresponds to sw4 point %i,%i,%i"  %(i,j,k))
    print('\n')
    
    #save xyz as csv files for that point
    #save ESSI x y and z!
    
    fname = "_vel_sw4_"+row["point"]+".csv"
    x = sw4essiout['vel_1 ijk layout'][:,i,j,k]
    np.savetxt(fname='x'+fname,X=x)
    #save y
    y = sw4essiout['vel_0 ijk layout'][:,i,j,k]
    np.savetxt(fname='y'+fname,X=y)
    #save z
    z = -1.0*sw4essiout['vel_2 ijk layout'][:,i,j,k]
    np.savetxt(fname='z'+fname,X=z)
