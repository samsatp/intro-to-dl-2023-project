#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mem=8000
#SBATCH --gres=gpu:v100:1
#SBATCH -J samtrain
#SBATCH -o train_log/samtrain.out.%j
#SBATCH -e train_log/samtrain.err.%j
#SBATCH --account=project_2002605
#SBATCH

module purge
cat requirements.txt | xargs -n 1 pip install
module load pytorch/1.13

conf_file=$1
python train_rnn.py $conf_file
