from typing import Optional

GENERATE_PROMPT_LITERAL = (
    "I will provide you with a proverb in {source_language}. \n"
    "Please give its figurative meaning in {target_language}. \n"
    "Respond only with the meaning, without any additional explanations. \n"
    "Proverb: {proverb}"
)

GENERATE_PROMPT_FIGURATIVE = (
    "I will provide you with a proverb in {source_language}. \n"
    "Please give its literal meaning in {target_language}. \n"
    "Respond only with the meaning, without any additional explanations. \n"
    "Proverb: {proverb}"
)


def get_prompt_by_task(
    task_type: str,
    source_language: Optional[str],
    proverb: Optional[str] = None,
) -> str:
    if task_type == "gen_swa_literal":
        return GENERATE_PROMPT_LITERAL.format(
            source_language=source_language,
            target_language="Swahili",
            proverb=proverb,
        )
    elif task_type == "gen_eng_literal":
        return GENERATE_PROMPT_LITERAL.format(
            source_language=source_language,
            target_language="English",
            proverb=proverb,
        )
    elif task_type == "gen_swa_fig":
        return GENERATE_PROMPT_FIGURATIVE.format(
            source_language=source_language,
            target_language="Swahili",
            proverb=proverb,
        )
    elif task_type == "gen_eng_fig":
        return GENERATE_PROMPT_FIGURATIVE.format(
            source_language=source_language,
            target_language="English",
            proverb=proverb,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")
