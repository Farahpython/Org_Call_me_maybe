import re


def is_pattern_name(name: str) -> bool:
    """True if a parameter name says it holds a regex pattern."""
    lowered = name.lower()
    return "regex" in lowered or "pattern" in lowered


def regex_problem(pattern: str, source: str | None) -> str | None:
    """Return why the pattern is unusable, or None if it looks fine.

    The pattern is tried on the source text with re.finditer, so we see
    what it really matches, not only whether it compiles.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        return f"does not compile ({error})"
    if source is None:
        return None  # nothing to test it against
    matches = [m.group() for m in compiled.finditer(source)]
    if not matches:
        return "matches nothing in the source text"
    if any(match == "" for match in matches):
        return "matches the empty string"
    return None
