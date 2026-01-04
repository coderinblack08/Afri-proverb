from typing import Optional

# GENERATE_PROMPT_LITERAL = (
#     "I will provide you with a proverb in {source_language}. \n"
#     "Please give its figurative meaning in {target_language}. \n"
#     "Respond only with the meaning, without any additional explanations. \n"
#     "Proverb: {proverb}"
# )

GENERATE_PROMPT_LITERAL = (
    "**Task:**\nYou are a professional translator specializing in proverbs \n"
    "Your task is to translate the following proverb from {source_language} to {target_language} literally. \n\n"
    "**Input**:\n"
    "A proverb in {source_language}\n\n"
    "**Output**:\n"
    "The literal translation of the proverb in {target_language}.\n\n"
    "**Input**:\n{proverb}\n\n"
    "**Output**:\n"
)
GENERATE_PROMPT_FIGURATIVE = (
    "**Task:**\nYou are a professional translator specializing in proverbs \n"
    "Your task is to infer the figurative meaning of the following proverb from {source_language} to {target_language}. \n\n"
    "**Input**:\n"
    "A proverb in {source_language}\n\n"
    "**Output**:\n"
    "The figurative meaning of the proverb in {target_language}.\n\n"
    "**Input**:\n{proverb}\n\n"
    "**Output**:\n"
)

# GENERATE_PROMPT_FIGURATIVE = (
#     "I will provide you with a proverb in {source_language}. \n"
#     "Please give its literal meaning in {target_language}. \n"
#     "Respond only with the meaning, without any additional explanations. \n"
#     "Proverb: {proverb}"
# )


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
