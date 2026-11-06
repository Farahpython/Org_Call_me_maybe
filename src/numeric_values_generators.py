import re
from llm_sdk import Small_LLM_Model
from .other_values_generators import choose_constrained_token

def extract_candidates(prompt: str, parameter_type: str) -> list[str]:
    """Pull possible values for a parameter type out of the request."""
    if parameter_type == "integer":
        return re.findall(r"-?\d+", prompt)
    if parameter_type == "number":
        return re.findall(r"-?(?:\d*[.])?\d+", prompt)
    if parameter_type == "string":
        matches = re.findall(r"""(['"])(.*?)\1""", prompt)
        return [match[1] for match in matches]
    return []


def remove_used(options: list[str], used: list[str]) -> list[str]:
    """Give back the options, minus one copy of each value already used."""
    left = list(options)
    for value in used:
        if value in left:
            left.remove(value)
    return left


def to_number(text: str) -> int | float:
    """A whole number becomes an int, anything with a dot a float."""
    if "." in text:
        return float(text)
    return int(text)


def choose_option(model: Small_LLM_Model, vocab: list[str],
                  input_ids: list[int], options: list[str]) -> str:
    """Write one of the options token by token, and nothing else.

    A token is allowed only if the text written so far plus that token is
    still the beginning of one of the options. The chosen tokens are
    appended to input_ids so the model sees what it has written.
    """
    text = ""
    while text not in options:
        scores = model.get_logits_from_input_ids(input_ids)
        allowed = [
            i for i, token in enumerate(vocab)
            if token and any(o.startswith(text + token) for o in options)
        ]
        best = choose_constrained_token(scores, allowed)
        text += vocab[best]
        input_ids.append(best)
    return text


def generate_number(model: Small_LLM_Model, vocab: list[str],
                    input_ids: list[int]) -> str:
    """Fallback: write a number token by token, only valid prefixes."""
    text = ""
    stop_ids = [i for i, t in enumerate(vocab) if t in (",", "}")]
    for _ in range(20):
        scores = model.get_logits_from_input_ids(input_ids)
        allowed = []
        for i, token in enumerate(vocab):
            if token and is_number_prefix(text + token):
                allowed.append(i)
        if is_number_complete(text):
            allowed += stop_ids
        if not allowed:
            break
        best = choose_constrained_token(scores, allowed)
        if best in stop_ids:
            break
        text += vocab[best]
        input_ids.append(best)
    if text in ("", "-"):
        text = "0"
    return text
