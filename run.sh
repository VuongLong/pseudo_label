#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=180GB
#SBATCH --job-name=optim
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/gpt
nvidia-smi
cd /home/hvp2011/implement/pseudo_label/
# export PYTHONPATH=$PWD

python main.py --M1 16 --M2 16 --data_root //vast/hvp2011/data// --dataset OfficeHome --noise .1 --output_dir 'outputs/OH/' --evaluation_step 200 --prompt_learning_rate 0.005 --t_weight 0.5 --enhanced_pseudo_label 1 --OT_clustering 1 --training_mode source-combined