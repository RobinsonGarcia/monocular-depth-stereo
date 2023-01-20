#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --job-name=algo360
#SBATCH --output=algo360.txt
#SBATCH --partition=sd_gpu
#SBATCH --mem=128G

#SBATCH -A algo360

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_BLOCKING_WAIT=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo


# module avail CUDA

export SLURM_ARRAY_TASK_ID=1 ###=====> HACK REMOVE IT
echo $SLURM_JOB_ID

module load cuda/11.1

srun singularity run --bind /petrobr/algo360/current:/petrobr/algo360/current --nv /petrobr/algo360/current/singularity-images/torch.sif python train_pl.py --lr=1e-5 --disparity=false --extend_3d=false --extended=false --l2_reg=0 --loss=scale --max_dist=150 --min_dist=.02 --nearest_up=true --accumulate_grad_batches=6 --mask_background=false --gpus=4 --strategy=ddp --pretrained_kitti=true --pred_log=false --resize_H_to=384 --SIZE 384 --batch_size=4 --disparity2=false &>> logs/$SLURM_JOB_ID.txt



