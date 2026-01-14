#!/bin/bash
#
#SBATCH --job-name=ferminet
#SBATCH --error=/work/submit/ahmed95/logs/fluxrun_%j.txt

#
#SBATCH --time=48:00:00
#SBATCH --partition=submit-gpu
#SBATCH --constraint=[nvidia_a30|Tesla_v100]
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=10000



source ~/venv/ferminet313/bin/activate
cd /work/submit/ahmed95/ferminet_ahmed/




# Generate interaction strength values 
int_values=$(seq 6.0 3.0 24.0)

# Loop over the flux2 values
for intstrength in $int_values; do


    # Call the Python script with the arguments
    python ferminet/configs/minimalChern_sweepflux.py --flux1 "$flux1" --flux2 "$flux2" --filename "$filename"
done



