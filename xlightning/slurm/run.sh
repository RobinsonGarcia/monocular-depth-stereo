#!/bin/bash

python3 /nethome/algo360/mestrado/monocular-depth-estimation/train_pl.py --sync_batchnorm --lr=1e-5 --disparity --extended --nearest_up --accumulate_grad_batches=6 --gpus=8 --strategy=ddp --resize_H_to=800 --SIZE 800 --batch_size=1 &>> logs/$SLURM_JOB_ID.txt