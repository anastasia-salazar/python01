"""Response parsing: turn a model's raw reply text into a Python dict."""

import json


class ResponseError(Exception):
    """A model's reply is malformed, breaks the required structure, or is off-topic."""


def parse_json_response(text):
    """Parse the reply as a JSON object, rejecting any text outside it."""
    text = text.strip()
    # Thinking models (qwen3) may put their reasoning in <think>...</think> first
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    # Some models wrap JSON in ```json fences despite instructions; allow only that
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ResponseError(f"response is not valid JSON ({error}); "
                            f"it started with {text[:80]!r}")
    if not isinstance(data, dict):
        raise ResponseError(f"response must be a JSON object, not a {type(data).__name__}")
    return data
