import re
import sys
import json
from typing import Type
from pydantic import BaseModel, ValidationError, create_model
from llm_sdk import Small_LLM_Model
from .argument_extraction_helpers import remove_used, extract_candidates, model_picks
from .loaders import load_function_defs, load_prompt
from .prompt_builders import prompt_builder, second_prompt_builder
from .classes import FunctionDefinition, Prompt


type_mapping = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool
}


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
                       function: FunctionDefinition, vocab: list[str],
                       candidates_by_parameter: dict[str, list[str]]
                    ) -> dict:
    arguments = {}
    used = []
    for name, details in function.parameters.items():
        parameter_type = details["type"]
        candidates = candidates_by_parameter.get(name, [])
        options = remove_used(candidates, used)
        # determines whether an option was chosen
        value = None
        if options:
            second_prompt = second_prompt_builder(prompt, function, name, options)
            print(second_prompt)
"""
            # returns an int
            choice = model_picks(second_prompt, model, vocab, len(options) + 1)
            if choice <= len(options):
                value = options[choice - 1]
            if value is None:
                third_prompt = third_prompt_builder(prompt, function, name,
                                          parameter_type)
            
            
    return argument
"""

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
    functions = load_function_defs("data/input/functions_definition.json")
    prompts = load_prompt("data/input/function_calling_tests.json")
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
                candidates_by_parameter = {}
                for name, p_type in f.parameters.items():
                    candidates = extract_candidates(prompt.prompt, p_type["type"])
                    candidates_by_parameter[name] = candidates
                arguments = response_arguments(prompt, model, f,
                                       vocab, candidates_by_parameter)
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
