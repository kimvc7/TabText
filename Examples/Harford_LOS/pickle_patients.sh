#!/bin/bash

#SBATCH -o pats_out.txt
#SBATCH -e pats_err.txt
#SBATCH --nodes=1 # Number of node
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=32 # How many threads to assign
#SBATCH --mem=64G # Hom much memory 
#SBATCH --array=0-86

# Initialize the module command first source
source /etc/profile
module unload anaconda
module load anaconda/2022a

# Call your script as you would from the command line
python pickle_patients.py --job_set Training --job_num ${SLURM_ARRAY_TASK_ID}
