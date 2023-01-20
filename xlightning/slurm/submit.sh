#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=6
#SBATCH --job-name=algo360
#SBATCH --output=algo360.txt
#SBATCH --partition=gpu


#SBATCH -A algo360

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_BLOCKING_WAIT=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

source $HOME/proxy.sh BF5V @cAnada2030

module load cuda

srun singularity exec --nv --home /nethome/algo360/mestrado/tmp_home/robinson.garcia --bind /nethome/algo360/mestrado:/nethome/algo360/mestrado /nethome/algo360/mestrado/torch.sif bash /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/slurm/run.sh 



