
source ~/venv/ferminet311_ahmed/bin/activate
cd ~/ferminet_ahmed/

(
for value in 13 15 17; do
    python ferminet/configs/minimalChern_targetmom.py --config_arg "$value"
done
) 



