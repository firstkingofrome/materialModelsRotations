# sw4_postprocess_plot_wfs.py

import sys
import glob
import obspy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#########################################################
# Main program begins here
print ('Number of arguments:', len(sys.argv))
print ('Argument List:', str(sys.argv))

# Makes plot of waveforms along station grid
#
# Arguments:
#   1   run (str)   = run name
#   2   name (str)  run short name
#   3   x1
#   4   x2
#   5   y1
#   6   y2

run = sys.argv[1]
name = sys.argv[2]
x1 = sys.argv[3]
x2 = sys.argv[4]
y1 = sys.argv[5]
y2 = sys.argv[6]

# Report input parameters:
print('run: ', run, ' name: ', name)
print('x1, x2: ', x1, x2)
print('y1, y2: ', y1, y2)

# Fixed parameters
t1 = 0
t2 = 80
g = 9.8

if (x1 != x2) and (y1 != y2):
    print('ERROR - either x1==x2 or y1==y2, exiting ...')
    sys.exit()

for in [ 'x', 'y', 'z' ]:
    if x1==x2:
        yglob = '['+str(y1)[0]+'-'+str(y2)[0]+']?'
        sacfile_glob = 'S_'+str(x1).zfill(2)+'_'+yglob+'.'+comp
        z1 = int(y1) - 1
        z2 = int(y2) + 1
        zlabel = 'X (km)'
        line_string = 'X='+str(x1).zfill(2)
    if y1==y2:
        yglob = '['+str(x1)[0]+'-'+str(x2)[0]+']?'
        sacfile_glob = 'S_'+yglob+'_'+str(y1).zfill(2)+'.'+comp
        z1 = int(x1) - 1
        z2 = int(x2) + 1
        zlabel = 'Y (km)'
        line_string = 'Y='+str(y1).zfill(2)
    sacfiles = glob.glob(sacfile_glob).sort()
    print('sacfiles: ', sacfiles)
    st = obspy.read(sacfile_glob)
    print(st)
    #st.plot()
    
    st_acc = st.copy()
    st_acc.differentiate()
    st_vel = st.copy()
    st_dis = st.copy()
    st_dis.integrate()
    time = np.linspace(0, st[0].stats.npts*st[0].stats.delta, st[0].stats.npts)
    print('Lengths: ', len(st_acc[0].data), len(st_vel[0].data), len(st_dis[0].data), len(time))
    gm_list = ['Acceleration (g)', 'Velocity (m/s)', 'Displacement (m)']
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,8), dpi=200)
    for i, ax in enumerate(axes):
        ax.set_xlim(t1, t2)
        ax.set_xlabel('Time (sec)')
        ax.set_ylim(z1, z2)
        ax.set_yticks(range(z1,z2))
        ax.set_ylabel(zlabel)
        ax.grid()
        ax.set_title(gm_list[i])
        # Acceleration
        if i==0:
            z = z1 + 1
            scale = 1/2
            for tr in st_acc[1:]:
                ax.plot(time, z + scale*tr.data/g, lw=1)
                ax.text(t2-5, z, str(round((np.abs(tr.data/g)).max(),2)), fontsize=6)
                z += 1
                if z > z2:
                    break
            ax.vlines(2,z1+1, z1+1.5, colors='k', lw=3)
            ax.vlines(4, z1+1, z1+2, colors='k', lw=3)
            ax.text(1, z1+0.8, '1 2 g', fontsize=8, horizontalalignment='left', verticalalignment='top')
        # Velocity
        if i==1:
            z = z1 + 1
            scale = 1/2
            for tr in st_vel[1:]:
                ax.plot(time, z + scale*tr.data, lw=1)
                ax.text(t2-5, z, str(round((np.abs(tr.data)).max(),2)), fontsize=6)
                z += 1
                if z > z2:
                    break
            ax.vlines(2, z1+1, z1+1.5, colors='k', lw=3)
            ax.vlines(4, z1+1, z1+2, colors='k', lw=3)
            ax.text(1, z1+0.8, '1 2 m/s', fontsize=8, horizontalalignment='left', verticalalignment='top')
        # Displacement
        if i==2:
            z = z1 + 1
            scale = 1/2
            for tr in st_dis[1:]:
                ax.plot(time, z + scale*tr.data, lw=1)
                ax.text(t2-5, z, str(round((np.abs(tr.data)).max(),2)), fontsize=6)
                z += 1
                if z > z2:
                    break
            ax.vlines(2, z1+1, z1+1.5, colors='k', lw=3)
            ax.text(1, z1+0.8, '1 m', fontsize=8, horizontalalignment='left', verticalalignment='top')        
    fig.suptitle('Run: '+name+' component: '+comp+' '+'Line along '+line_string, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(name+'.'+comp+'.'+line_string+'.plot_wf.png')
    