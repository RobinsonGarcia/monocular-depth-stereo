#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --job-name=algo360
#SBATCH --output=algo360.txt
#SBATCH --partition=sd_gpu
#SBATCH --mem=128G
#SBATCH --array=0-47
#SBATCH -A algo360

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_BLOCKING_WAIT=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
#module load NCCL/2.4.7-1-cuda.10.0

# run script from above
#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_DISABLE=1


FILES=(/petrobr/algo360/current/MultiGPU-lightning/configs/*)
srun --exclusive bash /petrobr/algo360/current/MultiGPU-lightning/slurm_run.sh ${FILES[$SLURM_ARRAY_TASK_ID]}



