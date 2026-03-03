import os
from typing import Dict, List

from pandas import read_csv

from ..data.prompts import get_few_shots_prompt_by_task, get_prompt_by_task
from .args import DataArguments, TaskArguments


def _normalize_column_name(name: str) -> str:
    return name.strip().lower()


def _get_label_column_name(task_type: str) -> str:
    if task_type == "gen_swa_literal":
        return "swa_literal"
    elif task_type == "gen_eng_literal":
        return "eng_literal"
    elif task_type == "gen_swa_fig":
        return "swa_figurative"
    elif task_type == "gen_eng_fig":
        return "eng_figurative"
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def load_cloud_eval_dataset(
    data_args: DataArguments,
    task_args: TaskArguments,
) -> List[Dict]:
    ret = []

    for loc, langs in data_args.location_langauge_paris:
        for lang in langs:
            file_path = os.path.join(
                data_args.dataset_dir,
                loc,
                f"{lang}_prov.csv",
            )
            csv_file = read_csv(file_path).fillna("")
            csv_file.columns = [
                _normalize_column_name(str(column)) for column in csv_file.columns
            ]

            source_column = _normalize_column_name(f"{lang}_prov")
            label_column = _normalize_column_name(
                _get_label_column_name(task_args.task_type)
            )

            if source_column not in csv_file.columns:
                raise ValueError(
                    f"Missing source column '{source_column}' in file '{file_path}'. "
                    f"Available columns: {list(csv_file.columns)}"
                )

            if label_column not in csv_file.columns:
                raise ValueError(
                    f"Missing label column '{label_column}' in file '{file_path}'. "
                    f"Available columns: {list(csv_file.columns)}"
                )

            few_shot_inputs = []
            few_shot_outputs = []
            if data_args.few_shot_num > 0:
                few_shots = csv_file.head(data_args.few_shot_num)
                few_shot_inputs = [str(x) for x in few_shots[source_column].tolist()]
                few_shot_outputs = [str(x) for x in few_shots[label_column].tolist()]

            rows = []
            for _, row in csv_file.iterrows():
                source = str(row[source_column])
                label = str(row[label_column])

                if data_args.few_shot_num > 0:
                    prompt = get_few_shots_prompt_by_task(
                        task_type=task_args.task_type,
                        source_language=lang,
                        proverb=source,
                        example_inputs=few_shot_inputs,
                        example_outputs=few_shot_outputs,
                    )
                else:
                    prompt = get_prompt_by_task(
                        task_type=task_args.task_type,
                        source_language=lang,
                        proverb=source,
                    )

                item = {
                    "prompt": prompt,
                    "label": label,
                }
                if "literal" in task_args.task_type:
                    item["source"] = source

                rows.append(item)

            ret.append({"location": loc, "language": lang, "rows": rows})

    return ret
