"""accuracy.py: measures function-call accuracy and speed. Not submitted."""

import argparse
import time
from typing import Any

from llm_sdk import Small_LLM_Model
from src.__main__ import (
    build_vocab,
    response_arguments,
    response_function,
)
from src.classes import Prompt
from src.loaders import load_function_defs
from src.prompt_builders import prompt_builder

# (prompt, expected function, expected values for keys we check)
CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("What is the sum of 2 and 3?", "fn_add_numbers", {"a": 2, "b": 3}),
    ("What is the sum of 265 and 345?", "fn_add_numbers",
     {"a": 265, "b": 345}),
    ("Add 10.5 and -4", "fn_add_numbers", {"a": 10.5, "b": -4}),
    ("Greet shrek", "fn_greet", {"name": "shrek"}),
    ("Greet john", "fn_greet", {"name": "john"}),
    ("Say hello to Alice", "fn_greet", {"name": "Alice"}),
    ("Reverse the string 'hello'", "fn_reverse_string", {"s": "hello"}),
    ("Reverse the string 'Python is fun'", "fn_reverse_string",
     {"s": "Python is fun"}),
    ("What is the square root of 144?", "fn_get_square_root", {"a": 144}),
    ("Calculate the square root of 2.25", "fn_get_square_root",
     {"a": 2.25}),
    ("Replace all numbers in 'Hello 34 I am 233 years old' with 'NUM'",
     "fn_substitute_string_with_regex",
     {"source_string": "Hello 34 I am 233 years old",
      "replacement": "NUM"}),
    ("Replace all vowels in 'Programming is fun' with '*'",
     "fn_substitute_string_with_regex",
     {"source_string": "Programming is fun", "replacement": "*"}),
]


def parse_args() -> argparse.Namespace:
    """Parse tester options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--functions-definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the accuracy and speed cases and print a compact report."""
    args = parse_args()
    functions = load_function_defs(args.functions_definition)
    functions_by_name = {function.name: function for function in functions}
    valid_names = list(functions_by_name)

    model = Small_LLM_Model()
    vocab = build_vocab(model)

    call_count = [0]
    original_get_logits = model.get_logits_from_input_ids

    def counted_get_logits(input_ids: list[int]) -> list[float]:
        """Count model next-token logit evaluations."""
        call_count[0] += 1
        return original_get_logits(input_ids)

    model.get_logits_from_input_ids = counted_get_logits  # type: ignore[method-assign]

    name_correct = 0
    full_correct = 0
    total_start = time.perf_counter()
    for user_text, expected_name, expected_values in CASES:
        call_count[0] = 0
        case_start = time.perf_counter()
        try:
            prompt = Prompt(prompt=user_text)
            selection_prompt = prompt_builder(prompt, functions)
            selected_name = response_function(
                selection_prompt, model, valid_names, vocab
            )
            function = functions_by_name.get(selected_name)
            if function is None:
                actual_values: dict[str, Any] = {}
                error = f"unknown function selected: {selected_name!r}"
            else:
                actual_values = response_arguments(
                    prompt, model, function, vocab
                )
                error = ""
            elapsed = time.perf_counter() - case_start
            name_ok = selected_name == expected_name
            values_ok = all(
                actual_values.get(key) == expected
                for key, expected in expected_values.items()
            )
            case_ok = name_ok and values_ok
            name_correct += int(name_ok)
            full_correct += int(case_ok)

            status = "PASS" if case_ok else "FAIL"
            print(
                f"{status} {elapsed:5.1f}s {call_count[0]:4d} calls "
                f"{user_text!r} -> {selected_name} {actual_values}"
            )
            if error:
                print(f"  Error: {error}")
            if not values_ok:
                print(f"  Expected checked values: {expected_values}")
        except Exception as exc:  # Keep the remaining cases running.
            elapsed = time.perf_counter() - case_start
            print(
                f"ERROR {elapsed:5.1f}s {call_count[0]:4d} calls "
                f"{user_text!r}: {type(exc).__name__}: {exc}"
            )

    total_elapsed = time.perf_counter() - total_start
    case_count = len(CASES)
    print(
        f"\nFunction-name accuracy: {name_correct}/{case_count} "
        f"= {100 * name_correct / case_count:.0f}%"
    )
    print(
        f"Checked-arguments accuracy: {full_correct}/{case_count} "
        f"= {100 * full_correct / case_count:.0f}%"
    )
    print(
        f"Total {total_elapsed:.1f}s; "
        f"{total_elapsed / case_count:.1f}s per prompt"
    )


if __name__ == "__main__":
    main()
