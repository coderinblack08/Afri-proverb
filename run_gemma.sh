export PYTHONPATH="./src:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1
# -------------------------GEMMA 4B-----------------------------------

# accelerate launch --num_processes=2 --mixed_precision=bf16 \
#   -m proverb.commands.evaluate --config configs/default.yaml \
#   --task_type gen_eng_literal \
#   --output_dir outputs/gemma3-4b-it-gen-eng-literal-Somali \
#   --model_name_or_path google/gemma-3-4b-it
#
# accelerate launch --num_processes=2 --mixed_precision=bf16 \
#   -m proverb.commands.evaluate --config configs/default.yaml \
#   --task_type gen_eng_fig \
#   --output_dir outputs/gemma3-4b-it-gen-eng-fig-Somali \
#   --model_name_or_path google/gemma-3-4b-it
#
# accelerate launch --num_processes=2 --mixed_precision=bf16 \
#   -m proverb.commands.evaluate --config configs/default.yaml \
#   --task_type gen_swa_literal \
#   --output_dir outputs/gemma3-4b-it-gen-swa-literal-Somali \
#   --model_name_or_path google/gemma-3-4b-it
#
# accelerate launch --num_processes=2 --mixed_precision=bf16 \
#   -m proverb.commands.evaluate --config configs/default.yaml \
#   --task_type gen_swa_fig \
#   --output_dir outputs/gemma3-4b-it-gen-swa-fig-Somali \
#   --model_name_or_path google/gemma-3-4b-it

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/gemma3-4b-it-gen-eng-literal-Somali \
  --model_name_or_path google/gemma-3-4b-it \
  --location Somali \
  --language somali

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/gemma3-4b-it-gen-eng-fig-Somali \
  --model_name_or_path google/gemma-3-4b-it \
  --location Somali \
  --language somali

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/gemma3-4b-it-gen-swa-literal-Somali \
  --model_name_or_path google/gemma-3-4b-it \
  --location Somali \
  --language somali

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/gemma3-4b-it-gen-swa-fig-Somali \
  --model_name_or_path google/gemma-3-4b-it \
  --location Somali \
  --language somali

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/gemma3-4b-it-gen-eng-literal-Ethiopia \
  --model_name_or_path google/gemma-3-4b-it \
  --location Ethiopia \
  --language borana,burji

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/gemma3-4b-it-gen-eng-fig-Ethiopia \
  --model_name_or_path google/gemma-3-4b-it \
  --location Ethiopia \
  --language borana,burji

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/gemma3-4b-it-gen-swa-literal-Ethiopia \
  --model_name_or_path google/gemma-3-4b-it \
  --location Ethiopia \
  --language borana,burji

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/gemma3-4b-it-gen-swa-fig-Ethiopia \
  --model_name_or_path google/gemma-3-4b-it \
  --location Ethiopia \
  --language borana,burji
