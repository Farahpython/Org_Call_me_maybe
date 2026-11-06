import argparse
import re
import sys
import json
from typing import Any
from typing import Type
from pydantic import BaseModel, ValidationError, create_model
from llm_sdk import Small_LLM_Model
from .numeric_values_generators import (
    extract_candidates, remove_used,
    choose_option, generate_number, to_number,
)
from .other_values_generators import generate_string, walk_string
from .loaders import load_function_defs, load_prompt
from .prompt_builders import prompt_builder, arguments_prompt_builder
from .classes import FunctionDefinition, Prompt


type_mapping = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--functions_definition",
                   default="data/input/functions_definition.json")
    p.add_argument("--input",
                   default="data/input/function_calling_tests.json")
    p.add_argument("--output",
                   default="data/output/function_calling_results.json")
    return p.parse_args()


def build_vocab(model: Small_LLM_Model) -> list[str]:
    scores_list = model.get_logits_from_input_ids([0])
    scores = len(scores_list)
    vocab = []
    for i in range(scores):
        vocab.append(model.decode([i]))
    return vocab


def response_function(input_text: str,
                      model: Small_LLM_Model,
                      valid_funcs: list[str],
                      vocab: list[str]) -> str:
    """
        This function receives the input text created by the prompt_builder
        function, the model, and a list of valid function names and returns
        the name of the chosen function.
    """
    token_so_far = ""  # generated so far
    # from tokens to token IDs
    input_ids = model.encode(input_text).flatten().tolist()
    while token_so_far not in valid_funcs:
        scores_list = model.get_logits_from_input_ids(input_ids)
        allowed = []
        for i in range(len(vocab)):
            text = vocab[i]
            if not text:
                continue
            possible_token = token_so_far + text
            if any(name.startswith(possible_token) for name in valid_funcs):
                allowed.append(i)
        if not allowed:
            return token_so_far
        best_token = max(allowed, key=lambda i: scores_list[i])
        token_so_far += vocab[best_token]
        input_ids.append(best_token)
    return token_so_far


def response_arguments(prompt: Prompt, model: Small_LLM_Model,
                       function: FunctionDefinition,
                       vocab: list[str]) -> dict[str, Any]:
    """Fill every parameter of the function, in definition order."""
    arguments_prompt = arguments_prompt_builder(prompt, function)
    input_ids = model.encode(arguments_prompt).flatten().tolist()
    arguments: dict[str, Any] = {}
    used: list[str] = []
    for index, (name, details) in enumerate(function.parameters.items()):
        kind = details["type"]
        lead = " {" if index == 0 else ", "
        opening = '"' if kind == "string" else ""
        fragment = f"{lead}{json.dumps(name)}: {opening}"
        input_ids += model.encode(fragment).flatten().tolist()
        if kind == "number":
            options = remove_used(extract_candidates(prompt.prompt, kind),
                                  used)
            if options:
                text = choose_option(model, vocab, input_ids, options)
            else:
                text = generate_number(model, vocab, input_ids)
            used.append(text)
            arguments[name] = to_number(text)
        elif kind == "string":
            arguments[name] = generate_string(model, vocab, input_ids)
            input_ids += model.encode('"').flatten().tolist()
        else:
            raise ValueError(f"Type not supported yet: {kind}")
    return arguments


def create_dynamic_model(f: FunctionDefinition) -> Type[BaseModel]:
    """
        This function creates and returns a pydantic class out of each
        function definition dynamically based on its name and given parameters.
    """
    fields = {}
    for name, p_type in f.parameters.items():
        python_type = type_mapping[p_type["type"]]
        fields[name] = (python_type, ...)
    ClassCreated = create_model(f.name, **fields)
    return ClassCreated


def main() -> None:
    args = parse_args()
    functions = load_function_defs(args.functions_definition)
    prompts = load_prompt(args.input)
    model = Small_LLM_Model()
    vocab = build_vocab(model)
    valid_defs = []
    for f in functions:
        valid_defs.append(f.name)
    for prompt in prompts:
        input_prompt = prompt_builder(prompt, functions)
        chosen_function = response_function(input_prompt, model, valid_defs, vocab)
        for f in functions:
            if f.name == chosen_function:
                arguments = response_arguments(prompt, model, f, vocab)
                print(prompt.prompt, "->", f.name, arguments)
    """            
                print(chosen_arguments)
                ClassCreated = create_dynamic_model(f)
                try:
                    validated_arguements = ClassCreated(**chosen_arguments)
                except ValidationError as e:
                    print("Invalid arguements: ", e)
    """

if __name__ == "__main__":
    main()
