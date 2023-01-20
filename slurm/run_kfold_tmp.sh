#!/bin/bash

FOLD=$1
JOBID=$SLURM_JOB_ID
LOG="logs/kfold_${JOBID}_${FOLD}.txt"
singularity run --bind /petrobr/algo360/current:/petrobr/algo360/current --nv /petrobr/algo360/current/singularity-images/torch.sif python /petrobr/algo360/current/MultiGPU-lightning/train_pl_tmp.py --lr=1e-5 --SIZE 800 --batch_size=1 --disparity=false --extend_3d=false --extended=true --l2_reg=0 --loss=l1 --max_dist=15 --min_dist=.05 --resize_H_to=800 --nearest_up=true  --accumulate_grad_batches=6 --mask_background=false --gpus=4 --strategy=ddp &>> $LOG


#train_pl.py --loss=l1 --min_dist=0.05 --l2_reg=0 --batch_size=6 --disparity=false --SIZE=384 --pretrained_kitti=true --resize_H_to=384 --accumulate_grad_batches=5 --mask_background=true --gpus=4 --strategy=ddp --extended=true &>> $LOG
