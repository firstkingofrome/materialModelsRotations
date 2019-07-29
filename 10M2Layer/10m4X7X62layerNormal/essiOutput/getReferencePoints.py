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
sw4_i_start = 2000
sw4_i_end   = 2200
sw4_j_start = 6000
sw4_j_end   = 6200
sw4_k_start = 0
sw4_k_end   = 200
sw4_grid_spacing=10
ESSI_SURFACE = 140





sw4essiout_filename =  "1X3X6M.SS_6000_6200.cycle=00000.essi"
pointsRequested = "referencePoints.txt"
pointsOutput = "outputPoints"
with open(pointsRequested ) as f:
    lines = f.readlines()
# ditch white spaces
lines = [x.strip() for x in lines] 
#ditch commentts
lines = [line for line in lines if("#") not in line]
#get all of the valid point pairs that he wants
pointPairs = []
for line in lines:
    pointPairs.append([line.split('(')[1]])
    pointPairs[-1][0] = [[int(re.findall("(\d+)",i)[0]),int(re.findall("(\d+)",i)[1]),int(re.findall("(\d+)",i)[2])] for i in pointPairs[-1]]
    #name of thie point pair
    pointPairs[-1].append(line.split(' ')[0])

#open the hdf5 container for this run and get each out put velocity point
sw4essiout = h5py.File(sw4essiout_filename, 'r')
spacing = sw4essiout['ESSI xyz grid spacing'][:][0]
for pointPair in pointPairs:
    #translate into xyz coordinates
    ### THE ESSI CORDINATES ARE XYZ WHILE SW4 IS YXZ
    i,j,k = int(pointPair[0][0][0]),int(pointPair[0][0][1]),int(pointPair[0][0][2])
    print("preparing refrence point " + pointPair[1])
    print("saving x y and z velocities for  exssi point " + "%i,%i,%i" % (i,j,k))
    j,i,k = int((i)/spacing), int((j)/spacing),int((ESSI_SURFACE-k)/spacing)
    print("This corresponds to sw4 point %i,%i,%i"  %(i,j,k))
    print('\n')
    
    #save xyz as csv files for that point
    #save ESSI x y and z!
    
    fname = "_vel_sw4_"+pointPair[1]+".csv"
    x = sw4essiout['vel_1 ijk layout'][:,i,j,k]
    np.savetxt(fname='x'+fname,X=x)
    #save y
    y = sw4essiout['vel_0 ijk layout'][:,i,j,k]
    np.savetxt(fname='y'+fname,X=y)
    #save z
    z = -1.0*sw4essiout['vel_2 ijk layout'][:,i,j,k]
    np.savetxt(fname='z'+fname,X=z)
