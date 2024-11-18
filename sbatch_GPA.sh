#!/bin/bash

#SBATCH --job-name=long-job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# SBATCH --nodelist=node02
#SBATCH --mem-per-cpu=50000
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tung-long.vuong@monash.edu
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log_err/%x-%j.err

source activate /home/long/envs
set -x

python main_PGA_multi.py --M1 16 --M2 16 --threshold .4 --data_root /home/shared/data/DomainBed/ --dataset OfficeHome