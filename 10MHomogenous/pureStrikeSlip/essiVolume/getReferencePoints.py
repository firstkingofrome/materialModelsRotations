#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:48:57 2019
@author: eeckert
"""
### not flipped

import sys
import h5py
import numpy as np
import re
sw4_x_start = 500
sw4_x_end   = 700
sw4_y_start = 1500
sw4_y_end   = 1700
sw4_z_start = 0
sw4_z_end   = 200

sw4essiout_filename = "1X3X6.SS_ESSI_SMALL.cycle=0000.essi"
pointsRequested = "referencePoints.txt"
pointsOutput = "outputPoints"
with open(pointsRequested ) as f:
    lines = f.readlines()
# ditch white spaces
lines = [x.strip() for x in lines] 
#get all of the valid point pairs that he wants
pointPairs = []
for line in lines:
    #pointPairs.append(line.split('(')[0],float(''.join(x for x in i if x.isdigit() or x in line.split('(')[2].split(','))))))
    pointPairs.append([line.split('(')[0],line.split('(')[2].split(',')])
    #get rid of units and other irritating shit
    pointPairs[-1][1] = [float(re.sub('[^0-9.\-]','',i)) for i in pointPairs[-1][1]]

#open the hdf5 container for this run and get each out put velocity point
sw4essiout = h5py.File(sw4essiout_filename, 'r')
spacing = sw4essiout['ESSI xyz grid spacing'][:][0]
for pointPair in pointPairs:
    #get rid of units commas and other shit
    #get the point
    i,j,k = int(pointPair[1][0]),int(pointPair[1][1]),int(pointPair[1][2])
    print("saving x y and z velocities for point " + "%i,%i,%i" % (i,j,k))
    i,j,k = int((i-sw4_x_start)/spacing), int((j-sw4_y_start)/spacing),int((k*-1)/spacing)
    #save xyz as csv files for that point
    #save x
    fname = "_vel_sw4_"+pointPair[0].split(' ')[0]+".csv"
    x = sw4essiout['vel_0 ijk layout'][:,i,j,k]
    np.savetxt(fname='x'+fname,X=x)
    #save y
    y = sw4essiout['vel_1 ijk layout'][:,i,j,k]
    np.savetxt(fname='y'+fname,X=y)
    #save z
    z = -1.0*sw4essiout['vel_2 ijk layout'][:,i,j,k]
    np.savetxt(fname='z'+fname,X=z)