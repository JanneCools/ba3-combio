#!/bin/bash
#PBS -N flowsom # job name
#PBS -l nodes=1:ppn=4 # single-node job, 4 cores
#PBS -l walltime=5:00:00 # max. 5h of wall time
#PBS -l vmem=50gb # 50GB of (virtual) memory required

# the program itself
echo Start Job
date
./src/v_measure_score.py
echo End Job