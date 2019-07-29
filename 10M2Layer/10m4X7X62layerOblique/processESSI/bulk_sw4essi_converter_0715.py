import sys
import h5py
import numpy as np
import datetime
import math
import os
from multiprocessing import Process

 
## use the version in the normal fault directory

def parseSW4InputFile(fname="10m1X10X62layerOblique.sw4input",essiCMD = "essioutput"):
    lines = []
    with open(fname,'r') as file:
        lines = file.readlines()
    #get only the lines with an essi cmd option
    lines = [line for line in lines if(essiCMD in line and "#" not in line)]    
    return lines

def parseOutESSIParameters(essiLine):
    lineOptionsDict={}
    essiLine = essiLine.split(" ")
    for i in essiLine:
        if('=' in i):
            lineOptionsDict[i.split('=')[0]] = i.split('=')[1]
    return lineOptionsDict

def child(essiLine,drm_filename,sw4HDF5OutExt=".essi"):
    #load the essi file and set parameters    
    #loop through these and save each one 
    # assuming:
    # essi_x =  sw4_vel_1
    # essi_y =  sw4_vel_0
    # essi_z = -sw4_vel_2
    #make a copy of the drm outfile so named for this region
    sw4OptionsDict = parseOutESSIParameters(essiLine)
    sw4essiout_filename = sw4OptionsDict["file"] + sw4HDF5OutExt
    # input metadata
    os.system("cp " + drm_filename + " "+sw4OptionsDict["file"].split('.')[0] +sw4OptionsDict["file"].split('.')[1]+ ".cycle=00000.essi")
    drm_filename = sw4OptionsDict["file"].split('.')[0] + ".essiHDF5"
    sw4_i_start = int(sw4OptionsDict["xmin"])
    sw4_i_end   = int(sw4OptionsDict["xmax"])
    sw4_j_start = int(sw4OptionsDict["ymin"])
    sw4_j_end   = int(sw4OptionsDict["ymax"])
    sw4_k_start = 0
    sw4_k_end   = int(sw4OptionsDict["depth"])
    
    essi_x_start = 30
    essi_x_end   = 150
    essi_y_start = 30
    essi_y_end   = 150
    essi_z_start = 30
    essi_z_end   = 140
    
    sw4_i0 = sw4_i_start+50
    sw4_j0 = sw4_j_start+50
    sw4_k0 = 0
    # End of input metadata
    
    print('Input files:', drm_filename, sw4essiout_filename)
    
    # Get the coordinates from DRM file
    drm_file = h5py.File(drm_filename)
    coordinates = drm_file['Coordinates']
    n_coord = int(coordinates.shape[0] / 3)
    print('# of coordinates: ', n_coord)
    drm_x = np.zeros(n_coord)
    drm_y = np.zeros(n_coord)
    drm_z = np.zeros(n_coord)
    
    # Store the coordiates in individual x, y, z arrays
    for i in range(0, n_coord):
        drm_x[i] = coordinates[i*3]
        drm_y[i] = coordinates[i*3+1]
        drm_z[i] = coordinates[i*3+2]
    
    print('First xyz: ', drm_x[0], drm_y[0], drm_z[0])
    print('Last xyz: ', drm_x[n_coord-1], drm_y[n_coord-1], drm_z[n_coord-1])
    
    # Get parameter values from HDF5 data
    sw4essiout = h5py.File(sw4essiout_filename, 'r')
    h  = sw4essiout['ESSI xyz grid spacing'][0]
    x0 = sw4essiout['ESSI xyz origin'][0]
    y0 = sw4essiout['ESSI xyz origin'][1]
    z0 = sw4essiout['ESSI xyz origin'][2]
    t0 = sw4essiout['time start'][0]
    npts = sw4essiout['cycle start, end'][1]
    dt = sw4essiout['timestep'][0]
    t1 = dt*(npts-1)
    print ('grid spacing, h: ', h)
    print ('ESSI origin x0, y0, z0: ', x0, y0, z0)
    print('timing, t0, dt, npts, t1: ', t0, round(dt,6), npts, round(t1,6) )
    print('Shape of HDF5 data: ', sw4essiout['vel_0 ijk layout'].shape)
    
    nt = sw4essiout['vel_0 ijk layout'].shape[0]
    
    time = np.linspace(t0, t1, npts+1)
    
    if not "Time" in drm_file.keys():
        Time_dset          = drm_file.create_dataset("Time", data=time)
        Accelerations_dset = drm_file.create_dataset("Accelerations", (n_coord*3, nt))
        Displacements_dset = drm_file.create_dataset("Displacements", (n_coord*3, nt))
    else:
        Time_dset          = drm_file['Time']
        Accelerations_dset = drm_file['Accelerations']
        Displacements_dset = drm_file['Displacements']
    
    
    i_slice_start = int((sw4_i0-sw4_i_start)/h)
    i_slice_end   = int(i_slice_start + (essi_y_end-essi_y_start)/h + 1)
    j_slice_start = int((sw4_j0-sw4_j_start)/h)
    j_slice_end   = int(j_slice_start + (essi_x_end-essi_x_start)/h + 1)
    k_slice_start = int((sw4_k0-sw4_k_start)/h)
    k_slice_end   = int(k_slice_start + (essi_z_end-essi_z_start)/h + 1)
    
    ni = i_slice_end - i_slice_start
    nj = j_slice_end - j_slice_start
    nk = k_slice_end - k_slice_start
    
    i_vel_all =  sw4essiout['vel_0 ijk layout'][:, i_slice_start:i_slice_end, j_slice_start:j_slice_end, k_slice_start:k_slice_end]
    j_vel_all =  sw4essiout['vel_1 ijk layout'][:, i_slice_start:i_slice_end, j_slice_start:j_slice_end, k_slice_start:k_slice_end]
    k_vel_all = -sw4essiout['vel_2 ijk layout'][:, i_slice_start:i_slice_end, j_slice_start:j_slice_end, k_slice_start:k_slice_end]
    
    print('Shape of vel: ', i_vel_all.shape)
    
    x_disp      = np.zeros(shape=(ni,nj,nk))
    y_disp      = np.zeros(shape=(ni,nj,nk))
    z_disp      = np.zeros(shape=(ni,nj,nk))
    prev_x_disp = np.zeros(shape=(ni,nj,nk))
    prev_y_disp = np.zeros(shape=(ni,nj,nk))
    prev_z_disp = np.zeros(shape=(ni,nj,nk))
    
    start = 1
    for i in range(start, nt-1):
    
        print('Iter ', i)
    
        x_vel = j_vel_all[i, :, :, :]
        y_vel = i_vel_all[i, :, :, :]
        z_vel = k_vel_all[i, :, :, :]
    
        prev_x_vel = j_vel_all[i-1, :, :, :]
        prev_y_vel = i_vel_all[i-1, :, :, :]
        prev_z_vel = k_vel_all[i-1, :, :, :]
    
        next_x_vel = j_vel_all[i+1,  :, :, :]
        next_y_vel = i_vel_all[i+1, :, :, :]
        next_z_vel = k_vel_all[i+1,  :, :, :]
    
        if i > 1:
            prev_x_disp = x_disp
            prev_y_disp = y_disp
            prev_z_disp = z_disp
    
        # Acc(i)=(Vel(i+1)-Vel(i-1))/(2*dt);
        x_acc  = (next_x_vel - prev_x_vel) / (2.0 * dt)
        # Disp(i)=Disp(i-1)+0.5*(Vel(i-1)+Vel(i))*dt;
        x_disp = prev_x_disp + 0.5 * (prev_x_vel + x_vel) * dt
    
        y_acc  = (next_y_vel - prev_y_vel) / (2.0 * dt)
        y_disp = prev_y_disp + 0.5 * (prev_y_vel + y_vel) * dt
    
        z_acc  = (next_z_vel - prev_z_vel) / (2.0 * dt)
        z_disp = prev_z_disp + 0.5 * (prev_z_vel + z_vel) * dt
    
        # Calculate the acceleration and displacement for each DRM coordinate
        for j in range(0, n_coord):
            loc_i = int((drm_y[j] - essi_y_start)/h)
            loc_j = int((drm_x[j] - essi_x_start)/h)
            loc_k = int((essi_z_end - drm_z[j])/h)
           
            Accelerations_dset[j*3, i]   = x_acc[loc_i, loc_j, loc_k]
            Accelerations_dset[j*3+1, i] = y_acc[loc_i, loc_j, loc_k]
            Accelerations_dset[j*3+2, i] = z_acc[loc_i, loc_j, loc_k] 
    
            Displacements_dset[j*3, i]   = x_disp[loc_i, loc_j, loc_k]
            Displacements_dset[j*3+1, i] = y_disp[loc_i, loc_j, loc_k]
            Displacements_dset[j*3+2, i] = z_disp[loc_i, loc_j, loc_k]
        #End for each DRM coordinate
    
        # print('Min     acceleration: ', np.min(Accelerations_dset[:,i]))
        # print('Average acceleration: ', np.average(Accelerations_dset[:,i]))
        # print('Max     acceleration: ', np.max(Accelerations_dset[:,i]))
    
        # print('Min     displacement: ', np.min(Displacements_dset[:,i]))
        # print('Average displacement: ', np.average(Displacements_dset[:,i]))
        # print('Max     displacement: ', np.max(Displacements_dset[:,i]))
        sys.stdout.flush()
    # End for each timestep
    sw4essiout.close()
    drm_file.close()
    print('End time:', datetime.datetime.now().time())

if __name__ == '__main__':
    drm_filename = "cubic_model.hdf5"
    #sw4essiout_filename = "1X3X6M.SS_3000_3200.cycle=00000.essi"
    sw4InputFile = "10m1X10X62layerOblique.sw4input"
    essiOptions = parseSW4InputFile(fname=sw4InputFile,essiCMD = "essioutput")
    sw4Inputs = parseSW4InputFile(fname=sw4InputFile,essiCMD = "essioutput")
    processList = []    
    """
    #prepare all of the processess
    for i in sw4Inputs:
        processList.append(Process(target=child, args=(i,drm_filename)))
    #release
    for process in processList:
        process.start()
    #wait for them all to finish
    for process in processList:
        process.join()
    """