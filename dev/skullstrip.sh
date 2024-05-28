#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 6

#$ -m ea

#$ -l h_rt=24:00:00

conda activate py3.11

python skullstrip.py