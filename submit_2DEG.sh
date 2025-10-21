#!/bin/bash
#
#SBATCH --job-name=ferminet
#SBATCH --error=err_%j.txt
#SBATCH --output=res_%j.txt

#
#SBATCH --time=48:00:00
#SBATCH --partition=submit-gpu
#SBATCH --constraint=[nvidia_a30|Tesla_v100]
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=8192

source ~/venv/ferminet313/bin/activate
cd /work/submit/ahmed95/ferminet_ahmed/
ferminet --config ferminet/configs/2DEG_rs.py 
#python ferminet/RDM.py


