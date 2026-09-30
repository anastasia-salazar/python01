"""Multi-Model Adversarial Reasoning System.

Two LLMs on the company server review a scenario adversarially:
  Turn 1  Model A proposes a solution or position, with reasoning
  Turn 2  Model B stress-tests it: weaknesses, risks, edge cases, counterarguments
  Turn 3  Model A revises or defends the proposal in response
  Final   an evaluation call summarizes robustness and remaining risks
The result is printed as a single JSON object - nothing else goes to stdout.
Progress messages go to stderr, and every prompt and raw output is logged to
logs/run_<timestamp>.jsonl.

Usage:
    python llm5.py sample_scenario.txt            # scenario from a text file
    python llm5.py "Should we ... ?"              # scenario typed as an argument
    python llm5.py                                # paste a scenario, then Ctrl+D
    python llm5.py sample_scenario.txt > result.json   # save the JSON to a file

Modules:
    config.py              credentials and model settings (.env)
    clients.py             API clients for Model A and Model B
    prompts.py             prompt construction for each step
    orchestrator.py        runs A -> B -> A and the final evaluation
    parsing.py             turns raw replies into JSON objects
    validation.py          structure and relevance checks
    interaction_logger.py  logs all prompts and raw outputs
    json_formatter.py      builds and verifies the final JSON output
"""

import sys
from pathlib import Path

from clients import LLMAPIError, create_model_a_client, create_model_b_client
from config import ConfigError, load_model_configs
from interaction_logger import InteractionLogger
from json_formatter import build_result, format_error, format_result
from orchestrator import run_adversarial_review
from parsing import ResponseError

MIN_WORDS = 5       # shorter than this isn't a real scenario
MAX_WORDS = 3000    # keep the whole discussion within the models' context window


def read_scenario():
    """Scenario from a file path, from the command-line text, or from pasted input."""
    if len(sys.argv) > 1:
        argument = " ".join(sys.argv[1:]).strip()
        path = Path(argument)
        if path.suffix and path.is_file():
            return path.read_text(encoding="utf-8").strip()
        return argument
    print("Paste the scenario, then press Enter and Ctrl+D when done:", file=sys.stderr)
    return sys.stdin.read().strip()


def check_scenario(scenario):
    words = len(scenario.split())
    if words < MIN_WORDS:
        raise ResponseError(f"Scenario is too short ({words} words); describe the "
                            f"situation in at least {MIN_WORDS} words.")
    if words > MAX_WORDS:
        raise ResponseError(f"Scenario is too long ({words} words, max {MAX_WORDS}).")


def main():
    logger = InteractionLogger(Path(__file__).resolve().parent / "logs")
    scenario = ""
    try:
        scenario = read_scenario()
        check_scenario(scenario)
        config_a, config_b = load_model_configs()
        model_a = create_model_a_client(config_a, logger)
        model_b = create_model_b_client(config_b, logger)
        steps = run_adversarial_review(scenario, model_a, model_b, logger)
        output = format_result(build_result(scenario, model_a, model_b, steps))
    except (OSError, ConfigError, LLMAPIError, ResponseError) as error:
        logger.log("failed", error=str(error))
        print(format_error(scenario, error, logger.path))
        return 1

    logger.log("final_output", output=output)
    print(output)
    print(f"Log saved to {logger.path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
