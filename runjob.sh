#!/bin/env bash
#SBATCH -A SNIC2022-5-420         # find your project with the "projinfo" command
#SBATCH -p alvis                   # what partition to use (usually not needed)
#SBATCH -t 2-01:10:00              # how long time it will take to run
#SBATCH -C NOGPU -n 1      # choosing no. GPUs and their type
#SBATCH -J Conformal_OPE                # the jobname (not needed)
#SBATCH -o Conformal_OPE_H_$1_29.out  # name of the output file

# Load modules
ml purge
#ml PyTorch/1.8.1-fosscuda-2020b matplotlib/3.3.3-fosscuda-2020b JupyterLab/2.2.8-GCCcore-10.2.0
module load Python/3.9.5-GCCcore-10.3.0
export PYTHONPATH=$PYTHONPATH:/cephyr/users/foffano/Alvis/python39/lib/python3.9/site-packages

# Interactive
#jupyter lab

# or you can instead use
#jupyter notebook

# Non-interactive
python main.py --horizon=$1 --weights_estimation_method=$2 --confidence_method=IS --runs $3 --seed $(($1*$3*4100))

# or you can instead use
#jupyter nbconvert --to python regression-pytorch.ipynb &&
#python regression-pytorch.py
