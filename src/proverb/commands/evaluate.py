import sys


sys.path.append("./src")


import json
import os
import time
from pprint import pprint

from proverb.engine.cloud_eval import load_cloud_eval_dataset
from proverb.engine.args import _parse_args
from proverb.data.loader import load_proverb_dataset
from proverb.data.collators import ProverbDataCollator
from proverb.extras.logging import get_logger
from proverb.engine.trainer import CustomTrainer
from proverb.engine.metrics import TranslateMetric, TextTranslateMetric
from proverb.model.loader import load_model, load_tokenizer
from tqdm.auto import tqdm

# Use an explicit proverb namespace so logs remain visible when this module runs as __main__.
logger = get_logger("proverb.commands.evaluate")


def _limit_dataset(dataset, task_args):
    if task_args.my_debug:
        debug_size = min(100, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(debug_size))

    if task_args.max_eval_samples > 0:
        max_samples = min(task_args.max_eval_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return dataset


def _limit_rows(rows, task_args):
    if task_args.my_debug:
        rows = rows[:100]

    if task_args.max_eval_samples > 0:
        rows = rows[: task_args.max_eval_samples]

    return rows


def _preview(text: str, limit: int = 220) -> str:
    flat = (text or "").replace("\n", " ").strip()
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "..."


def _run_local_evaluation(model_args, training_args, data_args, task_args):
    tokenizer = load_tokenizer(model_args)
    tokenizer.padding_side = "left"
    model = load_model(model_args)

    loaded_datasets = load_proverb_dataset(
        tokenizer, data_args, training_args, task_args
    )
    data_collator = ProverbDataCollator(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=TranslateMetric(tokenizer=tokenizer),
    )

    metrics_results = []
    for item in loaded_datasets:
        dataset = _limit_dataset(item["dataset"], task_args)

        predict_output = trainer.predict(test_dataset=dataset)
        metrics_result = predict_output.metrics

        # result = trainer.evaluate(eval_dataset=item["dataset"])
        #
        metrics_results.append(
            {
                "task_type": task_args.task_type,
                "location": item["location"],
                "language": item["language"],
                "results": metrics_result,
            }
        )

        if training_args.local_process_index == 0:
            trainer.save_predictions(
                dataset=dataset,
                predict_results=predict_output,
                file_name=f"generated_predictions_{item['location']}_{item['language']}.jsonl",
            )

        if task_args.my_debug:
            break

    if training_args.local_process_index == 0:
        pprint(metrics_results)
        with open(
            os.path.join(training_args.output_dir, "evaluation_results.json"), "w"
        ) as f:
            json.dump(metrics_results, f, indent=4)


def _append_cloud_prediction(file_path: str, prediction: dict):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        f.flush()


def _run_together_evaluation(model_args, training_args, data_args, task_args):
    from proverb.model.together_client import TogetherGenerationConfig, TogetherGenerator

    if training_args.world_size > 1:
        raise ValueError(
            "Together backend does not support multi-process evaluation. Run with a single process."
        )

    os.makedirs(training_args.output_dir, exist_ok=True)
    generator = TogetherGenerator(
        model_name_or_path=model_args.model_name_or_path,
        api_key=model_args.together_api_key,
        config=TogetherGenerationConfig(
            max_tokens=model_args.together_max_tokens,
            max_tokens_ceiling=model_args.together_max_tokens_ceiling,
            temperature=model_args.together_temperature,
            top_p=model_args.together_top_p,
            max_retries=model_args.together_max_retries,
        ),
    )
    loaded_datasets = load_cloud_eval_dataset(data_args, task_args)
    metric = TextTranslateMetric(include_comet=True)

    rows_per_dataset = [_limit_rows(item["rows"], task_args) for item in loaded_datasets]
    total_prompts = sum(len(rows) for rows in rows_per_dataset)
    logger.info_rank0(
        "Together evaluation starting: model=%s datasets=%d prompts=%d max_tokens=%d max_tokens_ceiling=%d",
        model_args.model_name_or_path,
        len(loaded_datasets),
        total_prompts,
        model_args.together_max_tokens,
        model_args.together_max_tokens_ceiling,
    )

    metrics_results = []
    processed_prompts = 0
    run_start = time.time()
    progress = tqdm(
        total=total_prompts,
        desc="Together eval",
        unit="prompt",
        dynamic_ncols=True,
        leave=True,
    )
    for item, rows in zip(loaded_datasets, rows_per_dataset):
        location = item["location"]
        language = item["language"]
        prediction_file = os.path.join(
            training_args.output_dir,
            f"generated_predictions_{location}_{language}.jsonl",
        )
        if os.path.exists(prediction_file):
            os.remove(prediction_file)

        logger.info_rank0(
            "Dataset start: location=%s language=%s prompts=%d output=%s",
            location,
            language,
            len(rows),
            prediction_file,
        )

        predictions = []
        decoded_preds = []
        decoded_labels = []
        decoded_sources = []
        empty_count = 0
        for idx, row in enumerate(rows, 1):
            row_start = time.time()
            gen_info = generator.generate_with_info(row["prompt"])
            prediction = gen_info["text"]
            if not prediction:
                empty_count += 1

            prediction_row = {
                "prompt": row["prompt"],
                "predict": prediction,
                "label": row["label"],
            }
            predictions.append(prediction_row)
            _append_cloud_prediction(prediction_file, prediction_row)
            decoded_preds.append(prediction)
            decoded_labels.append(row["label"])
            if "source" in row:
                decoded_sources.append(row["source"])

            processed_prompts += 1
            latency = time.time() - row_start
            usage = gen_info.get("usage")
            usage_text = (
                f"p={usage.get('prompt_tokens')} c={usage.get('completion_tokens')} t={usage.get('total_tokens')}"
                if usage
                else "n/a"
            )
            status = "OK" if prediction else "EMPTY"
            progress.update(1)
            progress.set_postfix_str(
                f"{location}/{language} {status} out={_preview(prediction, limit=60)}"
            )
            progress.write(
                (
                    f"[{processed_prompts}/{total_prompts}] "
                    f"[{location}/{language} {idx}/{len(rows)}] "
                    f"{status} finish={gen_info.get('finish_reason')} "
                    f"attempts={gen_info.get('attempts')} "
                    f"max_tokens={gen_info.get('max_tokens_used')} "
                    f"latency={latency:.2f}s usage={usage_text}\n"
                    f"Output: {_preview(prediction)}"
                )
            )
            if not prediction and gen_info.get("reasoning"):
                progress.write(f"Reasoning preview: {_preview(gen_info['reasoning'])}")

        metrics_result = metric(
            predictions=decoded_preds,
            references=decoded_labels,
            sources=decoded_sources if decoded_sources else None,
        )
        metrics_results.append(
            {
                "task_type": task_args.task_type,
                "location": location,
                "language": language,
                "results": metrics_result,
            }
        )
        logger.info_rank0(
            "Dataset done: location=%s language=%s prompts=%d empty_predictions=%d metrics=%s",
            location,
            language,
            len(rows),
            empty_count,
            metrics_result,
        )

        if task_args.my_debug:
            break

    pprint(metrics_results)
    with open(
        os.path.join(training_args.output_dir, "evaluation_results.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_results, f, indent=4)
    progress.close()
    logger.info_rank0(
        "Together evaluation finished in %.2fs. Results saved to %s",
        time.time() - run_start,
        os.path.join(training_args.output_dir, "evaluation_results.json"),
    )


def main():
    model_args, training_args, data_args, task_args = _parse_args()
    if model_args.inference_backend == "together":
        _run_together_evaluation(model_args, training_args, data_args, task_args)
    else:
        _run_local_evaluation(model_args, training_args, data_args, task_args)


if __name__ == "__main__":
    main()
