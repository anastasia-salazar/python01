"""Article Analysis System.

Sends an article to the Jade AI Models endpoint (Ollama-compatible API) and
gets back a validated JSON analysis: summary, important points, key themes and
target audience.

Credentials come from environment variables, or from a .env file next to this
script (copy .env.example to .env and fill it in; .env is ignored by git):
    LLM_USERNAME=...
    LLM_PASSWORD=...
Optional: LLM_API_URL (default: the Jade endpoint), LLM_MODEL (default: llama3.1:8b)

Usage:
    python llm2.py article.txt      # analyze a text file
    python llm2.py                  # paste an article, then press Ctrl+D
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

MAX_SUMMARY_WORDS = 150
MIN_POINTS, MAX_POINTS = 5, 10
MIN_THEMES, MAX_THEMES = 3, 5
MAX_THEME_WORDS = 6  # themes must be short phrases, not sentences

DEFAULT_API_URL = "https://aimodels.jadeglobal.com:8082"
DEFAULT_MODEL = "llama3.1:8b"  # fastest; qwen3:14b and gpt-oss:20b also available

MAX_ATTEMPTS = 3       # how many times to ask the LLM before giving up
REQUEST_TIMEOUT = 300  # seconds; the API guide recommends 120-300 for long generations


class LLMAPIError(Exception):
    """The API call itself failed (network error, bad status, bad payload)."""


class ValidationError(Exception):
    """The LLM replied, but its JSON is missing or breaks the rules."""


# ---------------------------------------------------------------
# 1. Build the prompt
# ---------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an expert article analyst. You read an article and return a
structured analysis as a single JSON object.

Return ONLY the JSON object. Do not add any text before or after it, and do not
wrap it in markdown code fences.

The JSON object must have exactly these four fields:
{{
  "summary": string,
  "important_points": [string, ...],
  "key_themes": [string, ...],
  "target_audience": string
}}

Rules:
- "summary": a concise summary of the article, at most {MAX_SUMMARY_WORDS} words.
- "important_points": between {MIN_POINTS} and {MAX_POINTS} items. Each item is one clearly
  written sentence capturing a core idea of the article.
- "key_themes": between {MIN_THEMES} and {MAX_THEMES} items. Each item is a short phrase of
  1 to {MAX_THEME_WORDS} words, NOT a full sentence, with no ending punctuation.
- "target_audience": one or two sentences identifying who the article is most
  relevant to.
- Base everything only on the article. Do not invent facts."""


def build_messages(article):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this article:\n\n<article>\n{article}\n</article>"},
    ]


# ---------------------------------------------------------------
# 2. Call the LLM API
# ---------------------------------------------------------------
def load_env_file(path):
    """Read KEY=VALUE lines from a .env file into os.environ (real env vars win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config():
    load_env_file(Path(__file__).resolve().parent / ".env")
    config = {
        "url": os.environ.get("LLM_API_URL", DEFAULT_API_URL).rstrip("/"),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "username": os.environ.get("LLM_USERNAME"),
        "password": os.environ.get("LLM_PASSWORD"),
    }
    missing = [var for var, name in [("LLM_USERNAME", "username"), ("LLM_PASSWORD", "password")]
               if not config[name]]
    if missing:
        raise LLMAPIError(f"Missing credential(s): {', '.join(missing)}. "
                          "Set them as environment variables or in a .env file.")
    return config


def service_is_up(config):
    """Hit the /health endpoint (no auth needed) to see if the service is running."""
    try:
        return requests.get(f"{config['url']}/health", timeout=10).status_code == 200
    except requests.RequestException:
        return False


def call_llm(messages, config):
    """Send the messages to the API's /api/chat endpoint and return the reply text.

    This is the only function that knows the API's request/response format.
    Network errors, rate limits (429) and server errors (5xx) are retried.
    """
    payload = {
        "model": config["model"],
        "messages": messages,
        "stream": False,    # get one complete JSON reply instead of streamed chunks
        "format": "json",   # Ollama constrains the model's output to valid JSON
        "options": {
            "temperature": 0.2,  # low = focused, consistent answers
            "num_ctx": 16384,    # default 4096 tokens would cut off long articles
        },
    }
    auth = (config["username"], config["password"])  # HTTP Basic Auth

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(f"{config['url']}/api/chat", json=payload, auth=auth,
                                     timeout=REQUEST_TIMEOUT)
        except requests.RequestException as error:
            problem = f"network error: {error}"
        else:
            if response.status_code == 200:
                try:
                    return response.json()["message"]["content"]
                except (ValueError, KeyError, TypeError):
                    raise LLMAPIError(f"Unexpected API response format: {response.text[:300]}")
            if response.status_code == 401:
                raise LLMAPIError("API returned 401: username or password is wrong")
            if response.status_code != 429 and response.status_code < 500:
                # 400/403/404 (e.g. unknown model) won't fix themselves, so don't retry
                raise LLMAPIError(f"API returned {response.status_code}: {response.text[:300]}")
            problem = f"API returned {response.status_code}"

        if attempt < MAX_ATTEMPTS:
            wait = 2 ** attempt
            print(f"  {problem}; retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)

    status = "is up" if service_is_up(config) else "appears to be DOWN (or you're off the VPN)"
    raise LLMAPIError(f"API call failed after {MAX_ATTEMPTS} attempts ({problem}). "
                      f"Health check: service {status}.")


# ---------------------------------------------------------------
# 3. Parse and validate the response
# ---------------------------------------------------------------
def parse_json(text):
    """Turn the reply text into a Python dict, rejecting anything outside the JSON."""
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
        raise ValidationError(f"Response is not valid JSON ({error}). "
                              f"Response started with: {text[:100]!r}")
    if not isinstance(data, dict):
        raise ValidationError("Response JSON must be an object, not a "
                              f"{type(data).__name__}")
    return data


def is_string_list(value):
    return isinstance(value, list) and all(isinstance(item, str) and item.strip()
                                           for item in value)


def validate(data):
    """Check the structure and constraints; raise ValidationError listing every problem."""
    errors = []
    expected = {"summary", "important_points", "key_themes", "target_audience"}

    missing = expected - data.keys()
    extra = data.keys() - expected
    if missing:
        errors.append(f"missing field(s): {sorted(missing)}")
    if extra:
        errors.append(f"unexpected field(s): {sorted(extra)}")

    summary = data.get("summary")
    if "summary" in data:
        if not isinstance(summary, str) or not summary.strip():
            errors.append("summary must be a non-empty string")
        elif len(summary.split()) > MAX_SUMMARY_WORDS:
            errors.append(f"summary has {len(summary.split())} words "
                          f"(max {MAX_SUMMARY_WORDS})")

    points = data.get("important_points")
    if "important_points" in data:
        if not is_string_list(points):
            errors.append("important_points must be an array of non-empty strings")
        elif not MIN_POINTS <= len(points) <= MAX_POINTS:
            errors.append(f"important_points has {len(points)} items "
                          f"(must be {MIN_POINTS}-{MAX_POINTS})")

    themes = data.get("key_themes")
    if "key_themes" in data:
        if not is_string_list(themes):
            errors.append("key_themes must be an array of non-empty strings")
        else:
            if not MIN_THEMES <= len(themes) <= MAX_THEMES:
                errors.append(f"key_themes has {len(themes)} items "
                              f"(must be {MIN_THEMES}-{MAX_THEMES})")
            for theme in themes:
                if len(theme.split()) > MAX_THEME_WORDS or theme.rstrip()[-1] in ".!?":
                    errors.append(f"key theme {theme!r} is a sentence, "
                                  f"not a short phrase")

    audience = data.get("target_audience")
    if "target_audience" in data:
        if not isinstance(audience, str) or not audience.strip():
            errors.append("target_audience must be a non-empty string")

    if errors:
        raise ValidationError("; ".join(errors))
    return data


# ---------------------------------------------------------------
# 4. Put it together: ask, validate, and retry with feedback
# ---------------------------------------------------------------
def analyze_article(article, config):
    """Return a validated analysis dict, or raise LLMAPIError / ValidationError."""
    messages = build_messages(article)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = call_llm(messages, config)
        try:
            return validate(parse_json(reply))
        except ValidationError as error:
            print(f"  Attempt {attempt}: invalid response - {error}", file=sys.stderr)
            if attempt == MAX_ATTEMPTS:
                raise ValidationError(f"No valid response after {MAX_ATTEMPTS} attempts. "
                                      f"Last problem: {error}")
            # Show the LLM its own answer and what was wrong, and ask it to fix it
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": f"Your response was invalid: {error}. "
                                            "Return the corrected JSON object only."},
            ]


def print_analysis(analysis):
    print("\nSUMMARY")
    print(analysis["summary"])
    print("\nIMPORTANT POINTS")
    for number, point in enumerate(analysis["important_points"], start=1):
        print(f"  {number}. {point}")
    print("\nKEY THEMES")
    print("  " + " | ".join(analysis["key_themes"]))
    print("\nTARGET AUDIENCE")
    print(analysis["target_audience"])


def read_article():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).read_text(encoding="utf-8")
    print("Paste the article, then press Ctrl+D (a new line first) when done:")
    return sys.stdin.read()


def main():
    try:
        article = read_article().strip()
    except OSError as error:
        print(f"Could not read article: {error}", file=sys.stderr)
        return 1
    if not article:
        print("No article text provided.", file=sys.stderr)
        return 1

    try:
        config = load_config()
        print("Analyzing article...", file=sys.stderr)
        analysis = analyze_article(article, config)
    except (LLMAPIError, ValidationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_analysis(analysis)

    output_path = Path(__file__).resolve().parent / "analysis.json"
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nJSON saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
