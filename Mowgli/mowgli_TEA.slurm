#!/bin/bash -l
#SBATCH --account=lp_edu_comparativegenomics
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=output_mowgli.log
#SBATCH --error=error.log

#TEA-seq dataset of PBMC

source ~/.bashrc
conda activate mowgli-env

echo `pwd`
echo `which python`

#Give permission
chmod u+x mowgli_TEA.py


python mowgli_TEA.py > output.log 2>&1
echo "Slurm job is launched. Check error and output log for more information."
