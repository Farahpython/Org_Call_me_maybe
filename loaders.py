import json
from .classes import FunctionDefinition, Prompt

def load_function_defs(file: str) -> list[FunctionDefinition]:
    """
        This function loads the function_definition.json file
        and returns a list of FunctionDefinition
    """
    try:
        with open(file, 'r') as f:
            d = json.load(f)
        return [FunctionDefinition(**item) for item in d]
    except FileNotFoundError:
        print(f"{file} not found...")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print("Invalid JSON syntax: ", e)
        sys.exit(1)
    except ValidationError as e:
        print("Invalid value,", e.errors()[0]["msg"])
        sys.exit(1)


def load_prompt(file: str) -> list[Prompt]:
    try:
        with open(file, 'r') as f:
            d = json.load(f)
        return [Prompt(**item) for item in d]
    except FileNotFoundError:
        print(f"{file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print("Invalid JSON syntax: ", e)
        sys.exit(1)
    except ValidationError as e:
        print("Invalid value,", e.errors()[0]["msg"])
        sys.exit(1)
