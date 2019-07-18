
Skip to content
Using University of Nevada Reno Mail with screen readers
17 of 4,304
What time would you guys like to talk tommorow?
Inbox
	x
Eric Eckert
	
	Sun, Jul 14, 12:40 PM (3 days ago)
Hello Suiwen and Houjun, if you both have a little while I think we should Skype for a bit tomorrow morning. Does 9 work well for both of you?
4
Eric Eckert
	
	Mon, Jul 15, 8:34 AM (2 days ago)
Hello Suiwen and Houjun, before we all talk today I wanted to have a brief recap so that we are all on the same page: Suiwen, I don’t know if you saw the email
David Mccallen
	
Mon, Jul 15, 8:53 AM (2 days ago)
	
to me, Suiwen, Houjun
Please go through this very methodically so all are clear, it is essential that the ESSI input motions at the DRM boundary are all defined with the right signs so the effective vector inputs are correct, I suspect if we get this right the good results of the earlier runs for the horizontal components will be realized and hopefully z will look a little better

__________________

David McCallen, Ph.D.
Professor Department of Civil and Environmental Engineering
Director Center for Civil Engineering Earthquake Research
University of Nevada, Reno (dmccallen@unr.edu)
Senior Scientist
Energy Geosciences Division
Lawrence Berkeley National Laboratory (dbmccallen@lbl.gov)

Cell 510 289 3286

Program administration and operations contact (LBNL)
Carol Chien
clchien@lbl.gov
Office 510 486 4724
Eric Eckert
	
	Mon, Jul 15, 9:27 AM (2 days ago)
Ok, so just to restate what I had above more clearly: The sw4 coordinates should look like this: sw4_x=hdf5Container["vel_1"] sw4_y=hdf5Container["vel_0"] sw4_z
Suiwen Wu
	
	AttachmentsMon, Jul 15, 10:26 AM (2 days ago)
Here are slides of the procedures of how to generate the input motions for ESSI model. I am confused about the structure of data storage in SW4. We discuss it a
Houjun Tang
	
	AttachmentsMon, Jul 15, 10:31 AM (2 days ago)
Let's look at this script for our meeting, I changed all the sw4 related coordinates into (i, j, k), instead of mixing with the essi (x, y, z).
Houjun Tang
	
	Mon, Jul 15, 11:57 AM (2 days ago)
Here's the latest converted data, I will send a document that describes how the conversion is done shortly. cubic_model_0715.hdf5
Eric Eckert
	
	Mon, Jul 15, 11:58 AM (2 days ago)
Could you also include the code that generated the container? It is easy to keep them together and that way we will always have it?
Houjun Tang
	
	Mon, Jul 15, 11:59 AM (2 days ago)
Here it is. sw4essi_converter_0715.py
Houjun Tang
	
	Mon, Jul 15, 1:18 PM (2 days ago)
Here is the document describing how the conversion is done, please let me know if you find anything unclear. SW4 motion to ESSI acceleration and displacement Th
Eric Eckert
	
	Mon, Jul 15, 1:30 PM (2 days ago)
Houjun I really think that we should be calling these sw4x,sw4y and sw4z, refering to them as sw4 i,j,k just adds another layer of compelxity and these points a
Eric Eckert
	
	Mon, Jul 15, 1:39 PM (2 days ago)
Also did you compare this container to the one from two weeks ago?
Eric Eckert
	
	Mon, Jul 15, 1:52 PM (2 days ago)
I am concerned because it looks like the -1* the z velocity is missing, was that absent from that container?
Houjun Tang
	
	Mon, Jul 15, 2:02 PM (2 days ago)
There is a minus sign on line 100, k_vel_all = -sw4essiout['vel_2 ijk layout']. Currently I prefer to use i, j, k for the sw4 data, we can change it to sw4 x, y
Eric Eckert
	
	Mon, Jul 15, 2:04 PM (2 days ago)
Ok thanks I did not notice that.
Eric Eckert
	
	Mon, Jul 15, 2:29 PM (2 days ago)
Suiwen, I just wanted to triple check on the z coordinates for ESSI: z=0 is always the bottom and you count up right? (so the essi point z=140 is the surface an
Eric Eckert <eeckert@nevada.unr.edu>
	
AttachmentsMon, Jul 15, 2:40 PM (2 days ago)
	
to Suiwen, Houjun, David
Suiwen, here is the data for the reference points, the reference points script and the output from the script. I treated 140 as the surface and 0 as the bottom. Output is in terms of ESSI coordinates and should match Houjuns conversion methodology. Please let me know if there are any discrepancies or problems. (the output .txt file shows the sw4 essi container INDEXES that I compute for each point)
4 Attachments
Suiwen Wu
	
Mon, Jul 15, 2:49 PM (2 days ago)
	
to me, David, Houjun
Correct.
Suiwen Wu
	
Mon, Jul 15, 2:50 PM (2 days ago)
	
to me, David, Houjun
Got it. Thanks.
	
	
	

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
sw4_x_start = 500
sw4_x_end   = 700
sw4_y_start = 1500
sw4_y_end   = 1700
sw4_z_start = 0
sw4_z_end   = 200
sw4_grid_spacing=10
ESSI_SURFACE = 140

sw4essiout_filename = "1X3X6.SS_ESSI_SMALL.cycle=0000.essi"
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
    
