import numpy as np
import obspy
import os
import matplotlib.pyplot as plt
# Turn interactive plotting off
plt.ioff()
DT=0.0028243268687629447
startTime = 2
endTime =30
startTime = int(startTime/DT)
endTime = int(endTime/DT)

def createTrace(data,trace_header):
    header = obspy.core.trace.Stats(header=trace_header)
    trace=obspy.core.trace.Trace(data=data,header=header)
    # add the rest of the sac header
    #trace.meta["sac"]=metaDataTraces[0].meta["sac"]
    return trace


def plotTraces(st,run="oblique",fname="xvelStations.png"):
    fig, ax = plt.subplots(len(st), 1, sharex='col', sharey='row', figsize=(6,20), dpi=600)
    time = np.linspace(0,st[0].stats['delta']*(len(st[0].data)),st[0].stats['npts'])
    for i in range(0,len(st)):
        label = st[i].stats["station"]
        #vmax = np.max(st[i].data)
        #ax[i].set_ylim(-vmax, vmax)
        #ax[i].plot(time, st[i].data, c='k', lw=1, label='ESSI comp: '+str(i))
        ax[i].plot(time, st[i].data, c='r', lw=0.25, label=label)
        #ax[i].set_ylabel(label)
        if i==2:
            ax[i].set_xlabel('Time (sec)')
        ax[i].legend(loc='upper right')
        if i==0:
            ax[i].set_title(run)
    fig.savefig(fname)

metaDataSacFile = "metaData.x"
#loads up the comparision points and plots each one
#makes a plot at the bottom of the selected ESSI container
files = os.listdir('.')
files = [i for i in files if (".csv") in i]
#os.mkdirs("./png",exists_ok=True)
#create a stream for every csv file out of the trace
#load meta data from the meta data trace
meta = obspy.read(metaDataSacFile) 
#load the traces and plot x vels
xvels = [i for i in files if("x_vel") in i]
st = obspy.core.stream.Stream()
for i in xvels:
    traceData = np.loadtxt(i)
    st.append(createTrace(traceData,meta[0].stats))
    st[-1].stats["station"] = i.split('.')[0]
    st[-1].stats["channel"] = ''
    label = i.split('.')[0] 

plotTraces(st,run="reverse",fname="xvelStations.png")
st.integrate()
plotTraces(st,run="reverse",fname="xDispStations.png")
st.differentiate()
st.differentiate()
plotTraces(st,run="reverse",fname="xAccelStations.png")
# now do y and z
yvels = [i for i in files if("y_vel") in i]
st = obspy.core.stream.Stream()
for i in yvels:
    traceData = np.loadtxt(i)
    st.append(createTrace(traceData,meta[0].stats))
    st[-1].stats["station"] = i.split('.')[0]
    st[-1].stats["channel"] = ''
    label = i.split('.')[0] 

plotTraces(st,run="reverse",fname="yvelStations.png")
st.integrate()
plotTraces(st,run="reverse",fname="yDispStations.png")
st.differentiate()
st.differentiate()
plotTraces(st,run="reverse",fname="yAccelStations.png")
#z
zvels = [i for i in files if("z_vel") in i]
st = obspy.core.stream.Stream()
for i in zvels:
    traceData = np.loadtxt(i)
    st.append(createTrace(traceData,meta[0].stats))
    st[-1].stats["station"] = i.split('.')[0]
    st[-1].stats["channel"] = ''
    label = i.split('.')[0] 
plotTraces(st,run="reverse",fname="zvelStations.png")
st.integrate()
plotTraces(st,run="reverse",fname="zDispStations.png")
st.differentiate()
st.differentiate()
plotTraces(st,run="reverse",fname="zAccelStations.png")
