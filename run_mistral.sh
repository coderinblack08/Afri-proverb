export PYTHONPATH="./src:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1
accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/mistral-7b-gen-eng-literal \
  --template_name mistral \
  --model_name_or_path mistralai/Ministral-3-8B-Instruct-2512

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/mistral-7b-gen-eng-fig \
  --template_name mistral \
  --model_name_or_path mistralai/Ministral-3-8B-Instruct-2512

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/mistral-7b-gen-swa-literal \
  --template_name mistral \
  --model_name_or_path mistralai/Ministral-3-8B-Instruct-2512

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/mistral-7b-gen-swa-fig \
  --template_name mistral \
  --model_name_or_path mistralai/Ministral-3-8B-Instruct-2512

# --------------------------------------------
