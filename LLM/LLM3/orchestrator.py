"""Interaction orchestration: runs the A -> B -> A discussion and the final synthesis."""

import sys

import prompts
from parsing import ResponseError, parse_json_response
from validation import check_relevance, validate_final_output, validate_structure

MAX_RESPONSE_ATTEMPTS = 3  # how many times to ask a model for a valid, on-topic response


def ask_for_structured_response(client, messages, step, topic, judge_client):
    """Ask a model until it returns valid, well-structured, on-topic JSON.

    If a response is malformed, breaks the structure, or drifts off-topic,
    the model is shown its answer and the problem, and asked to fix it.
    """
    for attempt in range(1, MAX_RESPONSE_ATTEMPTS + 1):
        raw = client.chat(messages, step=step)
        try:
            data = validate_structure(parse_json_response(raw), step)
            check_relevance(topic, data, judge_client, step)
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


def run_discussion(topic, model_a, model_b, logger):
    """Run the full discussion and return the validated final output dict."""
    logger.log("start", topic=topic, model_a=model_a.model, model_b=model_b.model)

    # Each model's relevance is judged by the *other* model, to avoid self-grading
    print(f"Turn 1: {model_a} states its position...", file=sys.stderr)
    initial = ask_for_structured_response(
        model_a, prompts.build_initial_prompt(topic), "initial", topic, judge_client=model_b)

    print(f"Turn 2: {model_b} critiques it...", file=sys.stderr)
    critique = ask_for_structured_response(
        model_b, prompts.build_critique_prompt(topic, initial), "critique", topic,
        judge_client=model_a)

    print(f"Turn 3: {model_a} replies...", file=sys.stderr)
    reply = ask_for_structured_response(
        model_a, prompts.build_reply_prompt(topic, initial, critique), "reply", topic,
        judge_client=model_b)

    # A neutral moderator call synthesizes the conclusion (using Model B's client)
    print(f"Synthesis: {model_b} writes the conclusion as moderator...", file=sys.stderr)
    conclusion = ask_for_structured_response(
        model_b, prompts.build_conclusion_prompt(topic, initial, critique, reply),
        "conclusion", topic, judge_client=model_a)

    result = {
        "topic": topic,
        "models": {"model_a": model_a.model, "model_b": model_b.model},
        "model_a_initial_response": initial,
        "model_b_critique": critique,
        "model_a_final_reply": reply,
        "conclusion": conclusion["conclusion"],
    }
    validate_final_output(result)
    logger.log("final_output", result=result)
    return result
