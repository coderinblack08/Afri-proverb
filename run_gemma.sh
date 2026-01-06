export PYTHONPATH="./src:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

# accelerate launch --num_processes=2 --mixed_precision=bf16 \
#   -m proverb.commands.evaluate --config configs/default.yaml \
#   --task_type gen_eng_literal \
#   --output_dir outputs/gemma3-12b-it-gen-eng-literal \
#   --model_name_or_path google/gemma-3-12b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/gemma3-12b-it-gen-eng-fig \
  --model_name_or_path google/gemma-3-12b-it

# accelerate launch --num_processes=2 --mixed_precision=bf16 \
#   -m proverb.commands.evaluate --config configs/default.yaml \
#   --task_type gen_swa_literal \
#   --output_dir outputs/gemma3-12b-it-gen-swa-literal \
#   --model_name_or_path google/gemma-3-12b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/gemma3-12b-it-gen-swa-fig \
  --model_name_or_path google/gemma-3-12b-it

# --------------------------------------------

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/gemma3-4b-it-gen-eng-literal \
  --model_name_or_path google/gemma-3-4b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/gemma3-4b-it-gen-eng-fig \
  --model_name_or_path google/gemma-3-4b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/gemma3-4b-it-gen-swa-literal \
  --model_name_or_path google/gemma-3-4b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/gemma3-4b-it-gen-swa-fig \
  --model_name_or_path google/gemma-3-4b-it

# --------------------------------------------

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/gemma3-1b-it-gen-eng-literal \
  --model_name_or_path google/gemma-3-1b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/gemma3-1b-it-gen-eng-fig \
  --model_name_or_path google/gemma-3-1b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/gemma3-1b-it-gen-swa-fig \
  --model_name_or_path google/gemma-3-1b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/gemma3-1b-it-gen-swa-literal \
  --model_name_or_path google/gemma-3-1b-it

# --------------------------------------------
