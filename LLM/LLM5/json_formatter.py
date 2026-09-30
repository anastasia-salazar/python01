"""JSON formatting: builds the final output object and guarantees it is strictly valid JSON."""

import json

from parsing import ResponseError
from validation import validate_structure

OUTPUT_FIELDS = ["input", "models", "model_a_initial_proposal", "model_b_critique",
                 "model_a_revised_response", "final_evaluation"]


def build_result(scenario, model_a, model_b, steps):
    """Assemble the final output in a fixed field order."""
    return {
        "input": scenario,
        "models": {"model_a": model_a.model, "model_b": model_b.model},
        "model_a_initial_proposal": steps["proposal"],
        "model_b_critique": steps["critique"],
        "model_a_revised_response": steps["revision"],
        "final_evaluation": steps["evaluation"],
    }


def validate_result(result):
    """Check the final object has every required field and each part is still valid."""
    missing = [field for field in OUTPUT_FIELDS if field not in result]
    if missing:
        raise ResponseError(f"final output missing field(s): {missing}")
    if not isinstance(result["input"], str) or not result["input"].strip():
        raise ResponseError("final output 'input' must be a non-empty string")
    validate_structure(result["model_a_initial_proposal"], "proposal")
    validate_structure(result["model_b_critique"], "stress_test")
    validate_structure(result["model_a_revised_response"], "revision")
    validate_structure(result["final_evaluation"], "evaluation")
    return result


def to_json(data):
    """Serialize to JSON, then parse it back to prove the text is strictly valid JSON."""
    # allow_nan=False rejects NaN/Infinity, which Python allows but the JSON standard doesn't
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    json.loads(text)
    return text


def format_result(result):
    return to_json(validate_result(result))


def format_error(scenario, error, log_path):
    """Failures are also reported as JSON, so the output is always valid JSON."""
    return to_json({"input": scenario, "error": str(error), "log_file": str(log_path)})
