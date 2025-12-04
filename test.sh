source /home/ahmed95/venv/ferminet311_ahmed/bin/activate
pip install -U "jax[cuda12]"
pip install -U nvidia-cusparse-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
               nvidia-cusolver-cu12 nvidia-cufft-cu12 nvidia-cuda-runtime-cu12 \
               nvidia-cuda-nvrtc-cu12 nvidia-nccl-cu12
python - <<'PY'
import jax; print("devices:", jax.devices())
PY