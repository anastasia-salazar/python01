"""Prompt construction for each step of the discussion.

Every prompt tells the model the exact JSON shape to return, and every
follow-up prompt includes the previous responses as context.
"""

import json

MODEL_A_SYSTEM = """You are Model A in a structured two-model discussion. You take a clear,
well-reasoned position on the topic and defend it with evidence and logic,
while staying open to good criticism.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""

MODEL_B_SYSTEM = """You are Model B in a structured two-model discussion. Your role is a
critical reviewer: you examine Model A's argument, point out weaknesses,
missing perspectives and unsupported claims, ask probing questions, and
expand on ideas Model A overlooked. Be rigorous but fair.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""

MODERATOR_SYSTEM = """You are a neutral moderator summarizing a discussion between two AI models.
You do not take sides; you weigh both perspectives fairly.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""

JUDGE_SYSTEM = """You check whether a response is actually about a given topic.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""


def as_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_initial_prompt(topic):
    """Turn 1 (Model A): state a position on the topic."""
    user = f"""Topic: "{topic}"

Give your position on this topic, explain your reasoning, and list your key arguments.

Return JSON with exactly these fields:
{{
  "position": "your stance on the topic in one sentence",
  "explanation": "your reasoning, at most 150 words",
  "key_arguments": ["3 to 5 arguments, one sentence each"]
}}"""
    return [{"role": "system", "content": MODEL_A_SYSTEM}, {"role": "user", "content": user}]


def build_critique_prompt(topic, initial_response):
    """Turn 2 (Model B): critique, question or expand on Model A's response."""
    user = f"""Topic: "{topic}"

Model A gave this response:
{as_json(initial_response)}

Critique Model A's response: point out weaknesses or gaps, offer counterpoints,
ask probing questions, and add important perspectives it missed.

Return JSON with exactly these fields:
{{
  "assessment": "one-sentence overall assessment of Model A's argument",
  "critique": "your critique, at most 150 words",
  "counterpoints": ["2 to 5 counterpoints or missing perspectives, one sentence each"],
  "questions": ["1 to 3 probing questions for Model A"]
}}"""
    return [{"role": "system", "content": MODEL_B_SYSTEM}, {"role": "user", "content": user}]


def build_reply_prompt(topic, initial_response, critique):
    """Turn 3 (Model A): respond to Model B's critique."""
    user = f"""Topic: "{topic}"

Your original response was:
{as_json(initial_response)}

Model B critiqued it:
{as_json(critique)}

Reply to Model B: answer its questions, defend the points you still believe,
and honestly concede where its criticism is valid.

Return JSON with exactly these fields:
{{
  "response_to_critique": "your reply to Model B, at most 150 words",
  "concessions": ["0 to 3 points where Model B is right (empty list if none)"],
  "final_position": "your final stance on the topic in one sentence"
}}"""
    return [{"role": "system", "content": MODEL_A_SYSTEM}, {"role": "user", "content": user}]


def build_conclusion_prompt(topic, initial_response, critique, reply):
    """Final orchestration call: synthesize the whole discussion."""
    user = f"""Topic: "{topic}"

Model A's initial response:
{as_json(initial_response)}

Model B's critique:
{as_json(critique)}

Model A's final reply:
{as_json(reply)}

Write a short, balanced conclusion that synthesizes the discussion: where the
models agree, where they still differ, and the most reasonable takeaway.

Return JSON with exactly this field:
{{
  "conclusion": "the synthesized conclusion, at most 100 words"
}}"""
    return [{"role": "system", "content": MODERATOR_SYSTEM}, {"role": "user", "content": user}]


def build_relevance_prompt(topic, response):
    """Ask a model to judge whether a response is on-topic."""
    user = f"""Topic: "{topic}"

Response to check:
{as_json(response)}

Is this response actually about the topic above (directly or through closely
related ideas), rather than drifting to something else?

Return JSON with exactly these fields:
{{
  "relevant": true or false,
  "reason": "one short sentence explaining your decision"
}}"""
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def build_correction_message(error):
    """Follow-up asking the model to fix an invalid response."""
    return {"role": "user",
            "content": f"Your response was invalid: {error}. "
                       "Return the corrected JSON object only, with exactly the requested fields."}
