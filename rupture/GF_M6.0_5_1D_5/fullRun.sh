#!/bin/csh
#SBATCH --account m3354
#SBATCH -N 1024
#SBATCH -S 4
#SBATCH -t 720
##SBATCH -p regular 
#SBATCH -q regular 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH 
#SBATCH -J mag6Rupture
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=eeckert@nevada.unr.edu
## burst buffer request
#BB create_persistent name=sw4output capacity=18000GB access_mode=striped type=scratch 
#DW persistentdw name=sw4output
#DW stage_out source=$DW_PERSISTENT_STRIPED_sw4output destination=/global/cscratch1/sd/eeckert/largeRuns/GF_M6.0_5_1D_5 type=directory 
## stage in the sw4 input files etc.
# Set total number of nodes request (must match -N above)
set NODES = 1024
#set the output directory and rupture directory (if applicable)
set RUN = GF_M6.0_5_1D_5 
set RUPTURE = m6.0-12.5x8.0.s005.v5.1.srf

# Set number of threads per node
# Set number of OpenMP threads per node
# The product of these two number must equal 64 
#setenv OMP_NUM_THREADS 1
#set PROCPERNODE = 64
setenv OMP_NUM_THREADS 2
setenv OMP_NUM_THREADS 2
set PROCPERNODE = 32
#setenv OMP_NUM_THREADS 8
#set PROCPERNODE = 8 

# Always use these values:
setenv OMP_PLACES threads
setenv OMP_PROC_BIND spread

echo NODES: $NODES
echo PROCPERNODE: $PROCPERNODE
echo OMP_NUM_THREADS: $OMP_NUM_THREADS
#set SW4BIN = /global/project/projectdirs/m2545/sw4/cori-knl/optimize
#set SW4FILE = sw4-nov-6-2018
set SW4BIN =  /global/project/projectdirs/m3354/tang/sw4/optimize_mp/
set SW4FILE = sw4

# Note that $OMP_NUM_THREADS * $PROCPERNODE must equal 64
set TASKS = ` echo $NODES $PROCPERNODE | awk '{ print $1 * $2 }' `
echo TASKS: $TASKS

# Number of Logical cores, for argument of -c in srun (below)
set NUMLC = ` echo $OMP_NUM_THREADS | awk '{ print $1 * 4 }' ` 
echo NUMLC: $NUMLC
echo "Running on ${NODES} nodes with ${TASKS} MPI ranks and OMP_NUM_THREADS=${OMP_NUM_THREADS}"

#modify the sw4 input file to output to the burst buffer
echo $DW_PERSISTENT_STRIPED_sw4output
#python modsw4input.py $DW_JOB_STRIPED $RUN.sw4input
#sed -i -e "s/path=/path=${TEST}/g" burstBufferTest.sw4input
#include the rupture file
echo RUN: $RUN
cp $RUN.sw4input $RUN
cp $RUPTURE $RUN
cd $RUN

#make sure that sw4 saves to the burst buffer
sed -i -e "s#path=#path=$DW_PERSISTENT_STRIPED_sw4output #" $RUN.sw4input
#make sure that sw4 reads the rupture file
sed -i -e "s#rupture file=#rupture file=/tmp/$RUPTURE #" $RUN.sw4input

# SBCAST files to assigned nodes
sbcast -f -F2 -t 600 --compress=lz4 $SW4BIN/$SW4FILE /tmp/$SW4FILE
sbcast -f -F2 -t 600 --compress=lz4 ./$RUN.sw4input /tmp/$RUN.sw4input
sbcast -f -F2 -t 600 --compress=lz4 ./$RUPTURE /tmp/$RUPTURE 


echo "Done sbcasting, preparing "

date
srun -N $NODES -n $TASKS -c $NUMLC --cpu_bind=cores  /tmp/$SW4FILE /tmp/$RUN.sw4input >! $RUN.output
date

cd ..
