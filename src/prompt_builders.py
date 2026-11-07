import json
from .classes import Prompt, FunctionDefinition


INSTRUCTIONS = (
        "/no_think\n"
        "Choose the listed function that fulfills the request using its "
        "description and the request's meaning. "
        "Allow synonyms and different wording.\n"
        "Extract every required argument from the request. "
        "Copy explicitly supplied string values exactly.\n"
        'Return only compact JSON with keys "prompt", "name", '
        '"parameters" in that order. Preserve the original request '
        "in prompt. Use no whitespace outside strings.\n"
        "Generate the most specific regex that matches only the requested content.\n"
        "Do not include surrounding whitespace unless the request explicitly asks for it.\n"
        "Cut a generated pattern at its first \s when the request never mentions whitespace\n"
    )

EXAMPLES = (
    "Function: fn_repeat_text\n"
    "Description: Repeat a text several times.\n"
    "Parameter 1: text (string)\n"
    "Parameter 2: times (integer)\n"
    'Request: Repeat the word "hi" 3 times\n'
    'Arguments: {"text": "hi", "times": 3}\n'
    "\n"
    "Function: fn_replace_text\n"
    "Description: Replace everything matching a pattern in a text.\n"
    "Parameter 1: text (string)\n"
    "Parameter 2: pattern (string)\n"
    "Parameter 3: new (string)\n"
    "Request: Replace all capital letters in 'aBcD' with dashes\n"
    'Arguments: {"text": "aBcD", "pattern": "[A-Z]", "new": "-"}\n'
    "\n"
    "Function: fn_replace_text\n"
    "Description: Replace everything matching a pattern in a text.\n"
    "Parameter 1: text (string)\n"
    "Parameter 2: pattern (string)\n"
    "Parameter 3: new (string)\n"
    "Request: Replace the letters a and b in 'cab' with underscores\n"
    'Arguments: {"text": "cab", "pattern": "[ab]", "new": "_"}\n'
    "\n"
    "Function: fn_replace_text\n"
    "Description: Replace everything matching a pattern in a text.\n"
    "Parameter 1: text (string)\n"
    "Parameter 2: pattern (string)\n"
    "Parameter 3: new (string)\n"
    "Request: Replace the letters x and z in 'zebra fox' with underscores\n"
    'Arguments: {"text": "zebra fox", "pattern": "[xz]", "new": "_"}\n'
    "\n"
)


def prompt_builder(prompt: Prompt, defs: list[FunctionDefinition]) -> str:
    """This function takes the loaded function definitions and a single prompt
    and returns them as a single text so its passed to the model."""
    model_prompt = ""
    model_prompt += "Available Functions:\n"
    for d in defs:
        model_prompt += f"- {d.name}: {d.description}\n"
    model_prompt += f"User request: {prompt.prompt}\n"
    model_prompt += "Function to call:"
    return model_prompt    


def arguments_prompt_builder(prompt: Prompt, f: FunctionDefinition) -> str:
    """ Prompt whose natural continuation is the JSON arguments object. """
    params = ", ".join(f"{n} ({d['type']})" for n, d in f.parameters.items())
    arguments_prompt = INSTRUCTIONS + EXAMPLES
    arguments_prompt += f"Function: {f.name}\n"
    arguments_prompt += f"Description: {f.description}\n"
    arguments_prompt += f"Parameters: {params}\n"
    arguments_prompt += f"Request: {prompt.prompt}\n"
    arguments_prompt += "Arguments:"
    return arguments_prompt
