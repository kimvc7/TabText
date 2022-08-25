#!/bin/bash

#SBATCH -o features_out.txt
#SBATCH -e features_err.txt
#SBATCH --nodes=1 # Number of node
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4 # How many threads to assign
#SBATCH --mem=4G # Hom much memory 
#SBATCH --array=0-8

# Initialize the module command first source
source /etc/profile
module unload anaconda
module load anaconda/2022a

# Call your script as you would from the command line
python create_features.py --job_set Validation --job_num ${SLURM_ARRAY_TASK_ID}
