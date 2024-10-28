#!/bin/bash -l
#SBATCH --account=lp_edu_comparativegenomics
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=output_mowgli.log
#SBATCH --error=error.log

source ~/.bashrc
conda activate mowgli-env

echo `pwd`
echo `which python`
chmod u+x mowgli_TEA.py

python mowgli_TEA.py > output.log 2>&1
echo "Slurm job is launched. Check error and output log for more information."
