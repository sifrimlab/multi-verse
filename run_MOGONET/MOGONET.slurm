#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --account=lp_edu_comparativegenomics

source ~/.bashrc
conda activate mogonet_env

python ..\MOGONET\main_mogonet.py > output.log 2>&1