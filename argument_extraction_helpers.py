import re
from llm_sdk import Small_LLM_Model
from .classes import FunctionDefinition, Prompt


def remove_used(options: list[str], used: list[str]) -> list[str]:
    """Give back the options, minus one copy of each value already used."""
    left = list(options)
    for value in used:
        if value in left:
            left.remove(value)
    return left


def extract_candidates(prompt: str, parameter_type: str) -> list[str]:
    if parameter_type == "integer":
        candidates = re.findall(r"-?\d+", prompt)
    elif parameter_type == "number":
        candidates = re.findall(r"-?(?:\d*[.])?\d+", prompt)
    elif parameter_type == "string":
        matches = re.findall(r"""(['"])(.*?)\1""", prompt)
        candidates = [match[1] for match in matches]
    else:
        candidates = []
    return candidates


def model_picks(prompt: str, model: Small_LLM_Model, vocab: list[str],
               how_many: int) -> int:
    """The model picks one number between 1 and how_many."""
    valid_digits = [str(n) for n in range(1, how_many + 1)]
    input_ids = model.encode(prompt).flatten().tolist()
    scores = model.get_logits_from_input_ids(input_ids)
    # only look at the tokens that are one of the allowed digits
    allowed = []
    for i in range(len(vocab)):
        if vocab[i].strip() in valid_digits:
            allowed.append(i)
    best = max(allowed, key=lambda i: scores[i])
    return int(vocab[best].strip())
