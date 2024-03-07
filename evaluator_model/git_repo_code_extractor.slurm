#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32768
#SBATCH --time=5:00:00 

source .venv/bin/activate
python3 git_repo_code_extractor.py 