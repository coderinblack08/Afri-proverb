export PYTHONPATH="./src:$PYTHONPATH"

accelerate launch --num_processes=2 --mixed_precision=fp16 \
  -m proverb.commands.evaluate --config configs/default.yaml
