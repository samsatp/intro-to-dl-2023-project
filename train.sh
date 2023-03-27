#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 1:00:00
#SBATCH --mem=8000
#SBATCH --gres=gpu:v100:1
#SBATCH -J samtrain
#SBATCH -o train_log/samtrain.out.%j
#SBATCH -e train_log/samtrain.err.%j
#SBATCH --account=project_2002605
#SBATCH

module purge
pip install -r requirements.linux.txt
module load pytorch/1.13
python train_rnn.py config/conf_1.yaml
