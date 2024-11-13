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

python main_SL_multi_weighted_domain_class.py --M1 16 --M2 16 --threshold .4 --data_root /home/long/data/ --dataset ${1} --target ${2}  --output_dir b_IC_new/ --ot_t_weight 0.5 --t_weight 0.5 --self_correct 1 --evaluation_step 50 --prompt_learning_rate 0.001 --w_scale 10.0