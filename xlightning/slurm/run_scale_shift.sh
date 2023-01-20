#!/bin/bash

python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:0" --fold=1 &>> logs/${SLURM_JOB_ID}_cuda0.txt &
python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:1" --fold=2 &>> logs/${SLURM_JOB_ID}_cuda1.txt &
python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:2" --fold=3 &>> logs/${SLURM_JOB_ID}_cuda2.txt &
python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:3" --fold=4 &>> logs/${SLURM_JOB_ID}_cuda3.txt &
python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:4" --fold=5 &>> logs/${SLURM_JOB_ID}_cuda4.txt &
python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:5" --fold=6 &>> logs/${SLURM_JOB_ID}_cuda5.txt &
python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:6" --fold=7 &>> logs/${SLURM_JOB_ID}_cuda6.txt &
python3 /nethome/algo360/mestrado/monocular-depth-estimation/xlightning/scale_shift.py --device="cuda:7" --fold=8 &>> logs/${SLURM_JOB_ID}_cuda7.txt
