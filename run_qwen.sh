export PYTHONPATH="./src:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1
accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/qwen3-8b-gen-eng-literal \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-8B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/qwen3-8b-gen-eng-fig \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-8B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/qwen3-8b-gen-swa-literal \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-8B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/qwen3-8b-gen-swa-fig \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-8B

# --------------------------------------------

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/qwen3-4b-gen-eng-literal \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/qwen3-4b-gen-eng-fig \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/qwen3-4b-gen-swa-literal \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/qwen3-4b-gen-swa-fig \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

# --------------------------------------------
#
accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/qwen3-1.7b-gen-eng-literal \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-1.7B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/qwen3-1.7b-gen-eng-fig \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-1.7B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/qwen3-1.7b-gen-swa-literal \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-1.7B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/qwen3-1.7b-gen-swa-fig \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-1.7B

# --------------------------------------------
