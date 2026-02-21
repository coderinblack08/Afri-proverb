export PYTHONPATH="./src:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_literal \
  --output_dir outputs/qwen3-4b/qwen3-4b-gen-eng-literal-DRC \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_eng_fig \
  --output_dir outputs/qwen3-4b/qwen3-4b-gen-eng-fig-DRC \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_literal \
  --output_dir outputs/qwen3-4b/qwen3-4b-gen-swa-literal-DRC \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

accelerate launch --num_processes=2 --mixed_precision=bf16 \
  -m proverb.commands.evaluate --config configs/default.yaml \
  --task_type gen_swa_fig \
  --output_dir outputs/qwen3-4b/qwen3-4b-gen-swa-fig-DRC \
  --template_name qwen3 \
  --model_name_or_path Qwen/Qwen3-4B

# --------------------------------------------
