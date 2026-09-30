"""Prompt construction for each step of the adversarial reasoning process.

Every prompt includes the original scenario, the previous responses as context, and the
exact JSON shape the model must return.
"""

import json

MODEL_A_SYSTEM = """You are Model A, a senior problem-solver. Given a scenario, you propose a
concrete, well-reasoned solution or position. When your proposal is stress-tested, you
revise it where the criticism is valid and defend it where it is not, honestly.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""

MODEL_B_SYSTEM = """You are Model B, an adversarial red-team reviewer. Your job is to
stress-test a proposal: find its weaknesses, risks, edge cases and strongest
counterarguments. Be rigorous, specific and fair. Do not rewrite the proposal yourself
and do not simply agree with it.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""

EVALUATOR_SYSTEM = """You are an impartial evaluator reviewing an adversarial review process.
You judge how robust the final proposal is after critique, and which risks remain.
You do not take sides.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""

JUDGE_SYSTEM = """You check whether a response actually addresses a given scenario.
Respond ONLY with a single JSON object. No text before or after it, no markdown."""

SEVERITY = '"low" | "medium" | "high"'


def as_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)


def scenario_block(scenario):
    return f"<scenario>\n{scenario}\n</scenario>"


def build_proposal_prompt(scenario):
    """Turn 1 (Model A): propose a solution or position."""
    user = f"""{scenario_block(scenario)}

Propose a solution or take a position on the scenario above, with your reasoning.

Return JSON with exactly these fields:
{{
  "summary": "your proposal in one sentence",
  "proposal": "the proposal in detail, at most 200 words",
  "reasoning": ["3 to 6 reasons supporting it, one sentence each"],
  "assumptions": ["1 to 5 assumptions your proposal relies on"]
}}"""
    return [{"role": "system", "content": MODEL_A_SYSTEM}, {"role": "user", "content": user}]


def build_stress_test_prompt(scenario, proposal):
    """Turn 2 (Model B): stress-test Model A's proposal."""
    user = f"""{scenario_block(scenario)}

Model A proposed:
{as_json(proposal)}

Stress-test this proposal against the scenario: identify its weaknesses, risks (with
severity), edge cases where it breaks, and the strongest counterarguments.

Return JSON with exactly these fields:
{{
  "overall_assessment": "one-sentence verdict on the proposal's soundness",
  "weaknesses": ["2 to 6 flaws in the proposal's logic or design"],
  "risks": [{{"risk": "what could go wrong", "severity": {SEVERITY}}}],
  "edge_cases": ["1 to 5 specific situations where the proposal fails or behaves badly"],
  "counterarguments": ["1 to 5 strongest arguments against the proposal"],
  "questions": ["1 to 3 questions Model A must answer"]
}}
"risks" must contain 2 to 6 items."""
    return [{"role": "system", "content": MODEL_B_SYSTEM}, {"role": "user", "content": user}]


def build_revision_prompt(scenario, proposal, critique):
    """Turn 3 (Model A): revise or defend the proposal in response to the critique."""
    user = f"""{scenario_block(scenario)}

Your original proposal was:
{as_json(proposal)}

Model B stress-tested it:
{as_json(critique)}

Revise your proposal to fix valid criticisms, defend points you still stand by, give
a concrete mitigation for each important risk, and answer Model B's questions.

Return JSON with exactly these fields:
{{
  "revised_proposal": "the improved proposal, at most 200 words",
  "changes_made": ["1 to 6 specific changes from the original proposal"],
  "defended_points": ["0 to 4 criticisms you reject, each with a brief reason"],
  "mitigations": [{{"risk": "a risk Model B raised", "mitigation": "how the revision handles it"}}],
  "answers_to_questions": ["an answer to each of Model B's questions"],
  "remaining_limitations": ["0 to 4 limitations you acknowledge are still unresolved"]
}}
"mitigations" must contain 1 to 6 items."""
    return [{"role": "system", "content": MODEL_A_SYSTEM}, {"role": "user", "content": user}]


def build_evaluation_prompt(scenario, proposal, critique, revision):
    """Final orchestration call: evaluate robustness and remaining risks."""
    user = f"""{scenario_block(scenario)}

Model A's initial proposal:
{as_json(proposal)}

Model B's stress test:
{as_json(critique)}

Model A's revised proposal:
{as_json(revision)}

Evaluate how robust the revised proposal is now. Consider which of Model B's risks were
adequately addressed and which remain.

Return JSON with exactly these fields:
{{
  "robustness_score": an integer from 1 (fragile) to 10 (very robust),
  "verdict": "robust" | "needs_minor_changes" | "needs_major_changes" | "not_viable",
  "summary": "concise evaluation of robustness, at most 100 words",
  "addressed_risks": ["0 to 6 risks the revision handled well"],
  "remaining_risks": [{{"risk": "a risk that is still open", "severity": {SEVERITY}}}]
}}
"remaining_risks" must contain 0 to 5 items."""
    return [{"role": "system", "content": EVALUATOR_SYSTEM}, {"role": "user", "content": user}]


def build_relevance_prompt(scenario, response):
    """Ask a model to judge whether a response addresses the scenario."""
    user = f"""{scenario_block(scenario)}

Response to check:
{as_json(response)}

Does this response actually address the scenario above (its specific situation, goals
and constraints), rather than being generic or about something else?

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
