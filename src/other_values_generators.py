import re
import json
import numpy as np
from llm_sdk import Small_LLM_Model

NUM_PREFIX = re.compile(r"-?(?:(?:0|[1-9]\d*)(?:\.\d*)?)?")
NUM_DONE   = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?")
NORMAL = 0
ESCAPE = 1
REPEAT_PENALTY = 1.0
_string_cache: dict[int, tuple[list[int], set[int]]] = {}


def choose_constrained_token(scores: list[float], allowed_ids: list[int]) -> int:
    """Choose the highest-logit token after masking disallowed tokens."""
    if not allowed_ids:
        raise ValueError("No valid tokens are available for constrained decoding.")

    logits = np.asarray(scores, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError("Token logits must be a one-dimensional sequence.")

    if any(token_id < 0 or token_id >= logits.size for token_id in allowed_ids):
        raise ValueError("An allowed token ID is outside the logits range.")

    allowed_mask = np.zeros(logits.size, dtype=bool)
    allowed_mask[allowed_ids] = True
    logits[~allowed_mask] = -np.inf
    return int(np.argmax(logits))

def is_number_prefix(text: str) -> bool:
    """True if text could still grow into a valid JSON number."""
    return NUM_PREFIX.fullmatch(text) is not None


def is_number_complete(text: str) -> bool:
    """True if text is a finished, valid JSON number."""
    return NUM_DONE.fullmatch(text) is not None


def walk_string(text: str, state: int) -> int | None:
    """Feed text through the string rules, starting at `state`.

    Returns the state reached, or None if some character is not allowed.
    """
    for char in text:
        if state == NORMAL:
            if char == "\\":
                state = ESCAPE
            elif char == '"' or ord(char) < 0x20:
                return None
        else:  # ESCAPE
            if char in '"\\/bfnrt':
                state = NORMAL
            else:
                return None
    return state


def string_options(vocab: list[str], state: int) -> tuple[list[int], set[int]]:
    """Return cached token IDs usable in this string state and closers."""
    if state not in _string_cache:
        usable: list[int] = []
        closing: set[int] = set()
        for token_id, token in enumerate(vocab):
            if not token:
                continue
            if state == NORMAL and token[0] == '"':
                usable.append(token_id)
                closing.add(token_id)
            elif walk_string(token, state) is not None:
                usable.append(token_id)
        _string_cache[state] = (usable, closing)
    return _string_cache[state]


def generate_number(model: Small_LLM_Model, vocab: list[str],
                       input_ids: list[int]) -> float:
    """Generate one number token by token, only allowing valid prefixes."""
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
    return float(text)


def generate_string(model, vocab, input_ids) -> str:
    raw = ""
    state = NORMAL
    counts: dict[int, int] = {}
    for _ in range(100):
        scores = model.get_logits_from_input_ids(input_ids)
        usable, closing = string_options(vocab, state)
        adjusted_scores = [
            score - REPEAT_PENALTY * counts.get(token_id, 0)
            for token_id, score in enumerate(scores)
        ]
        best = choose_constrained_token(adjusted_scores, usable)
        if best in closing:
            break
        raw += vocab[best]
        next_state = walk_string(vocab[best], state)
        if next_state is None:
            break
        state = next_state
        counts[best] = counts.get(best, 0) + 1
        input_ids.append(best)
    try:
        return json.loads('"' + raw + '"')
    except ValueError:
        return raw
