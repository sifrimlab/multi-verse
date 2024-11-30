#!/bin/bash 
#SBATCH --job-name=evaluator_training 
#SBATCH --account=lp_edu_comparativegenomics
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=test.log   
#SBATCH --error=test.err    
#SBATCH --mail-type=END,FAIL 
#SBATCH --mail-user=yuxin.qiu@student.kuleuven.be  

source ~/.bashrc  
conda activate /data/leuven/358/vsc35895/miniconda3 
python /data/leuven/358/vsc35895/IBP_multiverse_test/eval.py