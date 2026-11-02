import json
from .classes import Prompt, FunctionDefinition


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


def second_prompt_builder(prompt: Prompt, f: FunctionDefinition,
                          parameter: str, options: list[str]) -> str:
    """Numbered choice prompt; the last number is 'none of the above'."""
    text = f"User request: {prompt.prompt}\n"
    text += f"Function: {f.name} - {f.description}\n"
    for name, details in f.parameters.items():
        text += f"- {name}: {details['type']}\n"
    text += f"Parameter to fill now: {parameter}\n"
    text += "Which option is the value of this parameter?\n"
    for number, option in enumerate(options, start=1):
        text += f"{number}. {option}\n"
    text += f"{len(options) + 1}. None of the above\n"
    text += "Answer: "
    return text


def third_prompt_builder(user_prompt: str, function: FunctionDefinition,
                         parameter: str, parameter_type: str) -> str:
    """Prompt for writing one value from scratch (fallback)."""
    text = f"User request: {user_prompt}\n"
    text += f"Function: {function.name} - {function.description}\n"
    text += f"Value of parameter {parameter} ({parameter_type}): "
    if parameter_type == "string":
        """ That open quote tells the model that what comes next is the
        contents of a string """
        text += '"'
    return text
