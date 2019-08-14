#!/bin/csh
#SBATCH --account m3354
#SBATCH -N 1 
#SBATCH -S 4
#SBATCH -t 1
##SBATCH -p regular 
#SBATCH -q debug 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH 
#SBATCH -J rmBB
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=eeckert@nevada.unr.edu
## burst buffer destroy
#BB destroy_persistent name=sw4output
#cancel this job (I only submitted to issue burst buffer commands)
scancel $SLURM_JOB_ID
