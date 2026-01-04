export PYTHONPATH="./src:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml
