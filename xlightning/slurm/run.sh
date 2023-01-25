#!/bin/bash

python3 /nethome/algo360/mestrado/monocular-depth-stereo/train_pl.py --no_fill_in --no_disparity &>> logs/$SLURM_JOB_ID.txt