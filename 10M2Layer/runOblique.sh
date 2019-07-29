#!/bin/csh
#SBATCH --account m3354
#SBATCH -N 100
#SBATCH -S 4
#SBATCH -t 800
##SBATCH -p regular 
#SBATCH -q regular
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH 
#SBATCH -J 10MdomainforESSI
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=eeckert@nevada.unr.edu
## burst buffer request
##DW jobdw capacity=10GB access_mode=striped type=scratch 
##DW stage_out source=$DW_JOB_STRIPED/dirname destination=/data/home/eeckert/git/materialModelsRotations/10MHomogenous/10m1X3X6/10M1x3x6.sw4output type=directory
# Set total number of nodes request (must match -N above)
set NODES = 100

#set the output directory and rupture directory (if applicable)
set RUN = 10m4X7X62layerOblique

# Set number of threads per node
# Set number of OpenMP threads per node
# The product of these two number must equal 64 
#setenv OMP_NUM_THREADS 1
#set PROCPERNODE = 64
setenv OMP_NUM_THREADS 2
set PROCPERNODE = 32
#setenv OMP_NUM_THREADS 8
#set PROCPERNODE = 8 

# Always use these values:
setenv OMP_PLACES threads
setenv OMP_PROC_BIND spread

echo
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
echo
echo "Running on ${NODES} nodes with ${TASKS} MPI ranks and OMP_NUM_THREADS=${OMP_NUM_THREADS}"


#set RUN = GF_M6.0_1_1D_10
#set RUPTURE = m6.0-12.5x8.0.s001.v5.1.srf
#set RUN = GF_M6.5_1_1D_10
#set RUPTURE = m6.5-26.4x12.0.s001.v5.1.srf
echo RUN: $RUN
cp $RUN.sw4input $RUN
cd $RUN

# Remove any old output file
/bin/rm -r -f $RUN.output
# Stripe output directory
if ( -d $RUN.sw4output ) then
  /bin/rm -r -f $RUN.sw4output
endif
mkdir $RUN.sw4output
stripe_small $RUN.sw4output

# SBCAST files to assigned nodes
sbcast -f -F2 -t 300 --compress=lz4 $SW4BIN/$SW4FILE /tmp/$SW4FILE
sbcast -f -F2 -t 300 --compress=lz4 ./$RUN.sw4input /tmp/$RUN.sw4input
#sbcast -f -F2 -t 300 --compress=lz4 ./$RUPTURE /tmp/$RUPTURE not using a rupture file right now

date
srun -N $NODES -n $TASKS -c $NUMLC --cpu_bind=cores $SW4BIN/$SW4FILE $RUN.sw4input >! $RUN.output
date

cd ..
