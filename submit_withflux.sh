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




# Generate flux2 values from 0.25 to 3.0 with increments of 0.25
flux2_values=$(seq 0.25 0.25 3.0)

# Loop over the flux2 values
for flux2 in $flux2_values; do
    flux1=0.0

    # Determine the filename based on the value of flux2
    if (( $(echo "$flux2 == 0.25" | bc -l) )); then
        filename="/work/submit/ahmed95/minimalChern_NN/ferminet_2025_10_18_10:11:32"
    else
        flux2previous=$(printf "%.2f" "$(echo "$flux2 - 0.25" | bc)")
        filename="/work/submit/ahmed95/data/8particles_withflux/${flux2previous}"
    fi

    # Call the Python script with the arguments
    python ferminet/configs/minimalChern_sweepflux.py --flux1 "$flux1" --flux2 "$flux2" --filename "$filename"
done



