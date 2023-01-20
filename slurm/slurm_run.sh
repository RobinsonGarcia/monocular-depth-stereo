#!/bin/bash

CONFIG=$1
source ${CONFIG}

JOBID=$SLURM_ARRAY_JOB_ID
TESTID=$SLURM_ARRAY_TASK_ID
LOG="${JOBID}_${TESTID}"
echo $LOG
#--max_dist=$MAX_DIST 
singularity run --bind /petrobr/algo360/current:/petrobr/algo360/current --nv /petrobr/algo360/current/singularity-images/torch.sif python /petrobr/algo360/current/MultiGPU-lightning/train_seg2.py --disparity=$DISPARITY --max_dist=$MAX_DIST --accumulate_grad_batches=$ACCUMBATCH --auto_lr_find --auto_scale_batch_size=binsearch --path2meta=$PATH2META --freeze_encoder $FREEZE --gans_wrapper=$GANS_WRAPPER --pretrained_kitti=$PRETRAINED --accelerator=$ACCELERATOR --log_scale=$LOGSCALE --add_dropout=$ADD_DROP --npz_dir=$NPZ_DIR --optimizer=$OPTIMIZER --validation_module=$FOLD --adjust_depth=$ADJUST_DEPTH --extend_3d=$EXTEND_3D --random_split=$RANDOM_SPLIT --heavy_aug=$HA --model=$MODEL --batch_size=$BATCH_SIZE --SIZE=$SIZE --gpus=$GPUS --num_nodes=$NUM_NODES --sync_batchnorm --min_dist=$MIN_DIST --loss=$LOSS --resize_H_to=$RESIZE_H --l2_reg=$REG --lr=$LR  &>> logs/$LOG.txt
