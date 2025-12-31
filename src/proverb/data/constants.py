from enum import Enum


GENERATE_PROMPT = (
    "I will provide you with a proverb in {source_language}. \n"
    "Please give its figurative meaning in {target_language}. \n"
    "Respond only with the meaning, without any additional explanations. \n"
    "Proverb: {proverb}\n"
)

IGNORE_INDEX = -100


class ChoiceTokens(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
