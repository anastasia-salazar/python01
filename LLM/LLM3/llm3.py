"""Multi-Model Interaction System.

Two LLMs on the company server discuss a topic in three structured turns
(Model A -> Model B -> Model A), then a moderator call synthesizes a conclusion.
The result is printed as a single JSON object - nothing else goes to stdout.
Progress messages go to stderr, and every prompt and raw output is logged
to logs/discussion_<timestamp>.jsonl.

Usage:
    python llm3.py "Should AI be used to grade school exams?"
    python llm3.py                  # you'll be asked for the topic
    python llm3.py "..." > discussion.json   # save the JSON to a file

Modules:
    config.py              credentials and model settings (.env)
    clients.py             API clients for Model A and Model B
    prompts.py             prompt construction for each step
    orchestrator.py        runs the A -> B -> A discussion and the synthesis
    parsing.py             turns raw replies into JSON objects
    validation.py          structure, relevance and final-output checks
    interaction_logger.py  logs all prompts and raw outputs
"""

import json
import sys
from pathlib import Path

from clients import LLMAPIError, create_model_a_client, create_model_b_client
from config import ConfigError, load_model_configs
from interaction_logger import InteractionLogger
from orchestrator import run_discussion
from parsing import ResponseError


def read_topic():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    print("Enter a discussion topic: ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip()


def print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    topic = read_topic()
    if not topic:
        print_json({"error": "No topic provided."})
        return 1

    logger = InteractionLogger(Path(__file__).resolve().parent / "logs")
    try:
        config_a, config_b = load_model_configs()
        model_a = create_model_a_client(config_a, logger)
        model_b = create_model_b_client(config_b, logger)
        result = run_discussion(topic, model_a, model_b, logger)
    except (ConfigError, LLMAPIError, ResponseError) as error:
        # Even failures are reported as JSON, so stdout is always valid JSON
        logger.log("failed", error=str(error))
        print_json({"topic": topic, "error": str(error), "log_file": str(logger.path)})
        return 1

    print_json(result)
    print(f"Log saved to {logger.path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
