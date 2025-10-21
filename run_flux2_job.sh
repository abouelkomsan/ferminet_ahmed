#!/bin/bash
#
#SBATCH --job-name=ferminet_flux2
#SBATCH --error=/work/submit/ahmed95/logs/fluxrun_%j.txt
#SBATCH --output=/work/submit/ahmed95/logs/fluxrun_%j.out

#
#SBATCH --time=48:00:00
#SBATCH --partition=submit-gpu
#SBATCH --constraint=[nvidia_a30|Tesla_v100]
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=10000

# Activate the environment and set the working directory
source ~/venv/ferminet313/bin/activate
cd /work/submit/ahmed95/ferminet_ahmed/

flux1=0.0

# Determine the filename based on the value of flux2
filename="/work/submit/ahmed95/minimalChern_NN/ferminet_2025_10_18_10:11:32"


# Call the Python script with the arguments
python ferminet/configs/minimalChern_sweepflux.py --flux1 "$flux1" --flux2 "$FLUX2" --filename "$filename"