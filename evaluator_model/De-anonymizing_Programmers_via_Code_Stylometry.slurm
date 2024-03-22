#!/bin/bash
#SBATCH --job-name=ml_job_test_dataset_code               # Job name
#SBATCH --output=ml_job_train_dataset_code_saving.txt             # Standard output and error log
#SBATCH --partition=single              # Partition name
#SBATCH --nodes=1                       # Run all processes on a single node    
#SBATCH --ntasks=1                      # Run a single task        
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=32gb                      # Job memory request
#SBATCH --time=6:00:00                 # Time limit hrs:min:sec
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=myhomeqrc@gmail.com  # Where to send mail  

source activate .venv                # Activate your environment

# Execute your script
python De-anonymizing_Programmers_via_Code_Stylometry.py

# USAGE:
# $sbatch your-file.slum
# $squeue -u your_username
# $squeue -j jobID
# $scontrol show job jobID
# $squeue -p partition_name
# 28775899