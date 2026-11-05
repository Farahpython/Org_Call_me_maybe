import json
from .classes import Prompt, FunctionDefinition

INSTRUCTIONS = (
    "Fill in the arguments of a function call from the user's request.\n"
    "Rules:\n"
    "- Copy names, words and numbers exactly as written in the request.\n"
    "- If a parameter is a regex or pattern, write a regular expression "
    "that matches what the request describes. "
    "Do not copy the matched text itself.\n"
    "- If a parameter is the replacement, write only the replacement text, "
    "with no extra trailing or leading spaces.\n"
    "- A value starts right after the opening quote.\n\n"
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
    """
    This function takes the loaded function definitions and a single prompt
    and returns them as a single text so its passed to the model.
    """
    model_prompt = ""
    model_prompt += "Available Functions:\n"
    for d in defs:
        model_prompt += f"- {d.name}: {d.description}\n"
    model_prompt += f"User request: {prompt.prompt}\n"
    model_prompt += "Function to call:"
    return model_prompt


def arguments_prompt_builder(prompt: Prompt, f: FunctionDefinition) -> str:
    """Prompt whose natural continuation is the JSON arguments object."""
    params = ", ".join(f"{n} ({d['type']})" for n, d in f.parameters.items())
    arguments_prompt = INSTRUCTIONS + EXAMPLES
    arguments_prompt += f"Function: {f.name}\n"
    arguments_prompt += f"Description: {f.description}\n"
    arguments_prompt += f"Parameters: {params}\n"
    arguments_prompt += f"Request: {prompt.prompt}\n"
    arguments_prompt += "Arguments:"
    return arguments_prompt
