# read_essi.py

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy

os.chdir('../')#Quick eric hack

run = 'GF_M6.0_1_1D_10'

os.chdir(run+'/'+run+'.sw4output')

filename = 'M6.0_ESSI_SRF_X.cycle=00000.essi'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])

# Get parameter values from HDF5 data
h = f['ESSI xyz grid spacing']
x0 = f['ESSI xyz origin'][0]
y0 = f['ESSI xyz origin'][1]
z0 = f['ESSI xyz origin'][2]
print ('grid spacing, h: ', h)
print ('ESSI origin x0, y0, z0: ', x0, y0, z0)

t0 = f['time start'].value[0]
npts = f['cycle start, end'][1]
dt = f['timestep'].value[0]
t1 = dt*(npts-1)
print('timing, t0, dt, npts, t1: ', t0, round(dt,6), npts, round(t1,6) )

print('Shape of HDF5 data: ', f['vel_0 ijk layout'].shape)
nx = f['vel_0 ijk layout'].shape[0]
ny = f['vel_0 ijk layout'].shape[1]
nz = f['vel_0 ijk layout'].shape[2]

time = np.linspace(t0, t1, npts+1)

v = np.zeros((3,npts+1))
v[0,:] = f['vel_0 ijk layout'][0,0,0,:]
v[1,:] = f['vel_1 ijk layout'][0,0,0,:]
v[2,:] = f['vel_2 ijk layout'][0,0,0,:]
vmax = max( np.abs(v.min()), np.abs(v.max()) )

st = obspy.read('S_06_10.?')

#fig = plt.figure(figsize=(6,6), dpi=150)
#fig, (ax0, ax1, ax2) = plt.subplots(2, 2, sharex=True, sharey=True)
fig, ax = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(6,6), dpi=300)

for i in range(0,3):
    ax[i].set_ylim(-vmax, vmax)
    ax[i].plot(time, v[i,:], c='k', lw=1, label='ESSI comp: '+str(i))
    ax[i].plot(time, st[i].data, c='r', lw=1, ls='--', label='SAC')
    ax[i].set_ylabel('Velocity (m/s)')
    if i==2:
        ax[i].set_xlabel('Time (sec)')
    ax[i].legend(loc='upper right')
    if i==0:
        ax[i].set_title('ESSI Mw 6.0 test')
fig.savefig('ESSI_test_plot.png')
    