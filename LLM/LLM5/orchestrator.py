"""Interaction orchestration: runs A (propose) -> B (stress-test) -> A (revise) -> evaluation."""

import sys

import prompts
from parsing import ResponseError, parse_json_response
from validation import check_relevance, validate_structure

MAX_RESPONSE_ATTEMPTS = 3  # how many times to ask a model for a valid, relevant response


def ask_for_structured_response(client, messages, step, scenario, judge_client):
    """Ask a model until it returns valid, well-structured, relevant JSON.

    If a response is malformed, breaks its schema, or ignores the scenario, the model is
    shown its answer and the problem, and asked to fix it.
    """
    for attempt in range(1, MAX_RESPONSE_ATTEMPTS + 1):
        raw = client.chat(messages, step=step)
        try:
            data = validate_structure(parse_json_response(raw), step)
            check_relevance(scenario, data, judge_client, step)
            client.logger.log("validated", step=step, model=client.name, attempt=attempt)
            return data
        except ResponseError as error:
            client.logger.log("invalid_response", step=step, model=client.name,
                              attempt=attempt, error=str(error))
            print(f"  {client}: invalid response ({error}); asking again...", file=sys.stderr)
            if attempt == MAX_RESPONSE_ATTEMPTS:
                raise ResponseError(f"{client} gave no valid response for step '{step}' "
                                    f"after {MAX_RESPONSE_ATTEMPTS} attempts. "
                                    f"Last problem: {error}")
            messages = messages + [{"role": "assistant", "content": raw},
                                   prompts.build_correction_message(error)]


def run_adversarial_review(scenario, model_a, model_b, logger):
    """Run the full process and return each step's validated response."""
    logger.log("start", scenario=scenario, model_a=model_a.model, model_b=model_b.model)

    # Each model's relevance is judged by the *other* model, to avoid self-grading
    print(f"Turn 1: {model_a} proposes a solution...", file=sys.stderr)
    proposal = ask_for_structured_response(
        model_a, prompts.build_proposal_prompt(scenario), "proposal", scenario,
        judge_client=model_b)

    print(f"Turn 2: {model_b} stress-tests it...", file=sys.stderr)
    critique = ask_for_structured_response(
        model_b, prompts.build_stress_test_prompt(scenario, proposal), "stress_test", scenario,
        judge_client=model_a)

    print(f"Turn 3: {model_a} revises or defends it...", file=sys.stderr)
    revision = ask_for_structured_response(
        model_a, prompts.build_revision_prompt(scenario, proposal, critique), "revision",
        scenario, judge_client=model_b)

    # A final orchestration call evaluates the outcome as an impartial evaluator
    print(f"Evaluation: {model_b} evaluates robustness as an impartial evaluator...",
          file=sys.stderr)
    evaluation = ask_for_structured_response(
        model_b, prompts.build_evaluation_prompt(scenario, proposal, critique, revision),
        "evaluation", scenario, judge_client=model_a)

    return {"proposal": proposal, "critique": critique, "revision": revision,
            "evaluation": evaluation}
