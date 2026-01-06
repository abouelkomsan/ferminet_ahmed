source ~/venv/ferminet311_ahmed/bin/activate
cd ~/ferminet_ahmed/




# Generate flux2 values from 0.25 to 3.0 with increments of 0.25
flux2_values=$(seq 0.25 0.25 3.0)

# Loop over the flux2 values
for flux2 in $flux2_values; do
    flux1=0.0

    # Determine the filename based on the value of flux2
    if (( $(echo "$flux2 == 0.25" | bc -l) )); then
        filename="/data/ahmed95/NN_minimalChern/ferminet_2025_10_18_10:11:32"
    else
        flux2previous=$(printf "%.2f" "$(echo "$flux2 - 0.25" | bc)")
        filename="/data/ahmed95/NN_minimalChern/8particles_withflux/$(printf "%.10g" ${flux2previous})"
    fi

    # Call the Python script with the arguments
    python ferminet/configs/minimalChern_sweepflux2.py --flux1 "$flux1" --flux2 "$flux2" --filename "$filename"
done



