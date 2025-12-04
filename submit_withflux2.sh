#!/bin/bash
#
#SBATCH --job-name=ferminet
#SBATCH --error=/work/submit/ahmed95/logs/fluxrun_%j.txt
#SBATCH --output=/work/submit/ahmed95/logs/fluxrun_%j.out

#
#SBATCH --time=48:00:00
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=10000

# Activate the environment and set the working directory
source ~/venv/ferminet313/bin/activate
cd /work/submit/ahmed95/ferminet_ahmed/

# Generate flux2 values from 0.25 to 3.0 with increments of 0.25
flux2_values=$(seq 0.25 0.25 3.0)

# Submit a separate job for each flux2 value
for flux2 in $flux2_values; do
    sbatch --export=ALL,FLUX2=$flux2 run_flux2_job.sh
done