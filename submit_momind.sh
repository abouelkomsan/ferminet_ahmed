#!/bin/bash
#
#SBATCH --job-name=ferminet
#SBATCH --error=/work/submit/ahmed95/logs/inferencerun_%j.txt
#SBATCH --output=/work/submit/ahmed95/logs/inferencerun_res_%j.txt

#
#SBATCH --time=48:00:00
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=10000



source ~/venv/ferminet313/bin/activate
cd /work/submit/ahmed95/ferminet_ahmed/

(
for value in 13 15 17; do
    python ferminet/configs/minimalChern_targetmom.py --config_arg "$value"
done
) 



