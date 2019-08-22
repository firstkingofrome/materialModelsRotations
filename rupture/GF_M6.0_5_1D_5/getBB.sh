#!/bin/csh
#SBATCH --account m3354
#SBATCH -N 1 
#SBATCH -S 4
#SBATCH -t 1
##SBATCH -p regular 
#SBATCH -q debug 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH 
#SBATCH -J getBB
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=eeckert@nevada.unr.edu
## burst buffer request, use a persistent request to avoid the failures with the que
#BB create_persistent name=sw4output capacity=10000GB access_mode=striped type=scratch 
#DW persistentdw name=sw4output
mkdir $DW_PERSISTENT_STRIPED_sw4output/test1
echo started


